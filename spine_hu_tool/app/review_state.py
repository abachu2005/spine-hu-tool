"""Review-state backend -- the deterministic logic the GUI is a thin shell over.

Every physician action (accept/reject, adjust radius, nudge center, recompute)
goes through here so it is fully testable without a display. The GUI calls the
exact same methods.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ..core import Volume, ROIResult
from ..config import ROIParams
from ..pipeline import measure_level, process_case
from ..geometry.coords import make_sphere
from ..measurement.hu_stats import compute_hu_stats
from ..measurement.qc import compute_qc
from ..export.writers import AuditTrail


class ReviewState:
    def __init__(self, volume: Volume, seg: np.ndarray, case: dict,
                 reviewer: str = "unknown", audit_path: Optional[str] = None,
                 params: Optional[ROIParams] = None):
        self.volume = volume
        self.seg = seg
        self.case = case
        self.reviewer = reviewer
        self.params = params or ROIParams()
        self.audit = AuditTrail(audit_path) if audit_path else None
        self._undo_stack: list[tuple[str, dict]] = []

    # --- construction ----------------------------------------------------
    @classmethod
    def from_volume(cls, volume, seg, levels=None,
                    mode="centroid_volume_sphere",
                    only_clean=True, compute_comparison=False,
                    apply_calibration=True, progress=None, **kw):
        params = kw.get("params") or ROIParams()
        case = process_case(volume, seg, levels=levels, params=params,
                            mode=mode, only_clean=only_clean,
                            compute_comparison=compute_comparison,
                            apply_calibration=apply_calibration,
                            progress=progress)
        return cls(volume, seg, case, params=params,
                   reviewer=kw.get("reviewer", "unknown"),
                   audit_path=kw.get("audit_path"))

    @property
    def results(self) -> dict[str, ROIResult]:
        return self.case["results"]

    def _audit(self, action, level, detail=None):
        if self.audit:
            self.audit.record(self.reviewer, action, level, detail)

    # --- recomputation helpers -------------------------------------------
    def _recompute_stats(self, r: ROIResult):
        hu_crop = self.volume.hu[r.crop_slices]
        info = {"roi_mask": r.roi_mask, "center_idx": r.center_idx,
                "radius_mm": r.radius_mm, "warnings": []}
        r.stats = compute_hu_stats(hu_crop, r.roi_mask, self.volume.spacing)
        qc = compute_qc(hu_crop, r.body_mask, info, self.volume.spacing,
                        r.stats, self.params)
        # preserve level-status tag
        qc["level_status"] = r.qc.get("level_status")
        r.qc = qc

    # --- undo -------------------------------------------------------------
    def _snapshot(self, level: str) -> Optional[dict]:
        r = self.results.get(level)
        if r is None:
            return None
        return {
            "center_idx": tuple(r.center_idx),
            "radius_mm": r.radius_mm,
            "margin_mm": r.margin_mm,
            "roi_mask": None if r.roi_mask is None else r.roi_mask.copy(),
            "stats": dict(r.stats),
            "qc": dict(r.qc),
            "accepted": r.accepted,
        }

    def push_undo(self, level: str) -> None:
        """Record the current state of `level` so the next edit can be reverted.
        Call this once at the START of an edit gesture (e.g. a drag), not on
        every intermediate update."""
        snap = self._snapshot(level)
        if snap is not None:
            self._undo_stack.append((level, snap))
            if len(self._undo_stack) > 200:     # bound memory
                self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def undo(self) -> Optional[str]:
        """Revert the most recent recorded edit. Returns the affected level."""
        if not self._undo_stack:
            return None
        level, snap = self._undo_stack.pop()
        r = self.results.get(level)
        if r is None:
            return None
        r.center_idx = snap["center_idx"]
        r.radius_mm = snap["radius_mm"]
        r.margin_mm = snap["margin_mm"]
        r.roi_mask = snap["roi_mask"]
        r.stats = snap["stats"]
        r.qc = snap["qc"]
        r.accepted = snap["accepted"]
        self._audit("undo", level, {"center_idx": list(r.center_idx),
                                    "radius_mm": r.radius_mm})
        return level

    # --- physician actions -----------------------------------------------
    def set_decision(self, level: str, accepted: bool):
        r = self.results.get(level)
        if r is None:
            return
        r.accepted = accepted
        self._audit("accept" if accepted else "reject", level,
                    {"mean_HU": r.stats.get("mean_HU")})

    def set_radius(self, level: str, radius_mm: float):
        r = self.results.get(level)
        if r is None:
            return
        radius_mm = max(0.5, float(radius_mm))
        r.radius_mm = radius_mm
        r.margin_mm = r.max_safe_radius_mm - radius_mm
        sphere = make_sphere(r.body_mask.shape, r.center_idx, radius_mm,
                             self.volume.spacing)
        r.roi_mask = sphere & r.body_mask
        self._recompute_stats(r)
        self._audit("set_radius", level, {"radius_mm": radius_mm,
                                          "mean_HU": r.stats.get("mean_HU")})

    def nudge_center(self, level: str, delta_idx):
        r = self.results.get(level)
        if r is None:
            return
        new_c = tuple(int(c + d) for c, d in zip(r.center_idx, delta_idx))
        self.set_center(level, new_c)

    def set_center(self, level: str, center_idx, record: bool = True):
        """Move a level's ROI center. During a live cursor drag pass record=False
        to update the sphere/stats without spamming the audit log; the final
        position (drag release or single click) is recorded with record=True."""
        r = self.results.get(level)
        if r is None:
            return
        new_c = tuple(int(np.clip(int(center_idx[i]), 0, r.body_mask.shape[i] - 1))
                      for i in range(3))
        r.center_idx = new_c
        sphere = make_sphere(r.body_mask.shape, new_c, r.radius_mm,
                             self.volume.spacing)
        r.roi_mask = sphere & r.body_mask
        self._recompute_stats(r)
        if record:
            self._audit("set_center", level, {"center_idx": list(new_c),
                                              "mean_HU": r.stats.get("mean_HU")})

    def recompute_level(self, level: str, mode: Optional[str] = None):
        """Re-run the deterministic ROI engine for one level (discards edits)."""
        res = measure_level(self.volume, self.seg, level, self.params,
                            mode or self.case.get("mode", "centroid_sphere"))
        if res is not None:
            res.qc.setdefault("level_status", self.results.get(level, res).qc.get("level_status")
                              if level in self.results else None)
            self.results[level] = res
            self._audit("recompute", level, {"mean_HU": res.stats.get("mean_HU")})
        return res

    # --- serialization ---------------------------------------------------
    def serialize(self) -> dict:
        return {
            "reviewer": self.reviewer,
            "mode": self.case.get("mode"),
            "params": self.params.to_dict(),
            "level_tags": self.case.get("level_tags", []),
            "results": {lvl: {**r.summary(), "accepted": r.accepted}
                        for lvl, r in self.results.items()},
        }
