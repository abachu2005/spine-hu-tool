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
from ..pipeline import measure_level, process_case, _annotate_calibration
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
        self.compute_comparison = any(
            r.comparison is not None for r in case.get("results", {}).values())

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
        previous_qc = dict(r.qc)
        if r.accepted is True:
            # Inclusion approves the current measurement, not future geometry.
            # A subsequent edit must pass through the default policy again.
            r.accepted = None
        adjunct_stats = {
            k: v for k, v in r.stats.items()
            if k.startswith(("body_", "scout_"))
        }
        hu_crop = self.volume.hu[r.crop_slices]
        info = {"roi_mask": r.roi_mask, "center_idx": r.center_idx,
                "radius_mm": r.radius_mm, "warnings": []}
        r.stats = compute_hu_stats(hu_crop, r.roi_mask, self.volume.spacing)
        qc = compute_qc(hu_crop, r.body_mask, info, self.volume.spacing,
                        r.stats, self.params)
        # Preserve screening provenance independently of the newly computed QC.
        for key in ("level_status", "auto_excluded", "exclusion_reason",
                    "persistent_warnings", "truncated"):
            if key in previous_qc:
                qc[key] = previous_qc[key]
        for warning in previous_qc.get("persistent_warnings", []):
            if warning not in qc["warnings"]:
                qc["warnings"].append(warning)
        self._apply_status_floor(qc)
        if qc["qc_status"] == "fail" and r.accepted is None:
            qc["auto_excluded"] = True
            qc.setdefault("exclusion_reason", "; ".join(qc.get("warnings", []))
                          or "failed quality control")
        r.qc = qc
        r.stats.update(adjunct_stats)
        r.comparison = None
        self._annotate_result(r)

    @staticmethod
    def _apply_status_floor(qc: dict) -> None:
        """Screening/truncation concerns cannot be erased by moving the ROI."""
        if qc.get("qc_status") == "pass" and (
                qc.get("truncated") or qc.get("level_status") == "review"):
            qc["qc_status"] = "review"

    def _annotate_result(self, r: ROIResult) -> None:
        calibration = self.case.get("calibration")
        if calibration:
            _annotate_calibration(
                {r.level: r}, calibration,
                self.case.get("mode", "centroid_volume_sphere"))

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
        if r is None or not r.editable:
            return
        r.accepted = accepted
        self._audit("include" if accepted else "exclude", level,
                    {"mean_HU": r.stats.get("mean_HU")})

    def set_radius(self, level: str, radius_mm: float):
        r = self.results.get(level)
        if r is None or not r.editable:
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
        if r is None or not r.editable:
            return
        new_c = tuple(int(c + d) for c, d in zip(r.center_idx, delta_idx))
        self.set_center(level, new_c)

    def set_center(self, level: str, center_idx, record: bool = True):
        """Move a level's ROI center. During a live cursor drag pass record=False
        to update the sphere/stats without spamming the audit log; the final
        position (drag release or single click) is recorded with record=True."""
        r = self.results.get(level)
        if r is None or not r.editable:
            return
        new_c = tuple(int(np.clip(int(center_idx[i]), 0, r.body_mask.shape[i] - 1))
                      for i in range(3))
        if not r.body_mask[new_c]:
            return
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
        previous = self.results.get(level)
        res = measure_level(self.volume, self.seg, level, self.params,
                            mode or self.case.get("mode", "centroid_sphere"),
                            compute_comparison=self.compute_comparison)
        if res is not None:
            if previous is not None:
                res.stats.update({
                    k: v for k, v in previous.stats.items()
                    if k.startswith(("body_", "scout_"))
                })
                res.qc.setdefault("level_status", previous.qc.get("level_status"))
                for key in ("auto_excluded", "exclusion_reason",
                            "persistent_warnings", "truncated"):
                    if key in previous.qc:
                        res.qc[key] = previous.qc[key]
                for warning in previous.qc.get("persistent_warnings", []):
                    if warning not in res.qc.setdefault("warnings", []):
                        res.qc["warnings"].append(warning)
                if previous.accepted is False:
                    res.accepted = previous.accepted
            self._apply_status_floor(res.qc)
            self._annotate_result(res)
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
