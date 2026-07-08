"""Segmentation consistency gate: well-formed segs pass; scrambled/speck fail."""
import numpy as np

from spine_hu_tool.segmentation.consistency import check_segmentation
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import _phys_grid, _ellipsoid

SPACING = (1.0, 1.0, 1.0)
SHAPE = (60, 60, 200)


def _seg(placements):
    """placements: list of (level_short, center_mm, radii_mm)."""
    seg = np.zeros(SHAPE, dtype=np.int16)
    X, Y, Z = _phys_grid(SHAPE, SPACING)
    for short, c, r in placements:
        seg[_ellipsoid(X, Y, Z, c, r)] = label_id_for(short)
    return seg


def _well_formed():
    # contiguous T11->L1, monotonic SI, plausibly spaced, plausibly sized
    return _seg([
        ("T11", (30, 25, 40), (13, 12, 13)),
        ("T12", (30, 25, 80), (13, 12, 13)),
        ("L1", (30, 25, 120), (13, 12, 13)),
    ])


def test_well_formed_seg_passes():
    out = check_segmentation(_well_formed(), SPACING)
    assert out["seg_status"] == "ok"
    assert all(v["valid"] for v in out["levels"].values())


def test_scrambled_overlap_and_inversion_invalid():
    # T11 & L1 nearly co-located (non-adjacent overlap) and T12 displaced far
    # superiorly -> SI order is non-monotonic with the level labels.
    seg = _seg([
        ("T11", (30, 25, 60), (13, 12, 13)),
        ("T12", (30, 25, 140), (13, 12, 13)),
        ("L1", (30, 25, 70), (13, 12, 13)),
    ])
    out = check_segmentation(seg, SPACING)
    assert out["seg_status"] == "invalid"
    # the SI ordering is broken, so at least the inverted levels are rejected
    n_invalid = sum(1 for v in out["levels"].values() if not v["valid"])
    assert n_invalid >= 2
    assert any("SI order" in r for v in out["levels"].values() for r in v["reasons"])


def test_truncated_edge_level_does_not_condemn_clean_interior():
    # A truncated edge level (touching z=0) whose label bleeds up and overlaps a
    # full, well-formed interior level must NOT invalidate that interior level --
    # only the truncated edge label is blamed.
    seg = _seg([
        ("T12", (30, 25, 150), (13, 12, 13)),    # clean interior
        ("L1", (30, 25, 100), (13, 12, 13)),     # clean interior
        ("L2", (30, 25, 60), (13, 12, 13)),      # clean interior
    ])
    # a truncated "T10" smeared from the top edge down into T12's z-range
    X, Y, Z = _phys_grid(SHAPE, SPACING)
    seg[_ellipsoid(X, Y, Z, (30, 25, 175), (13, 12, 40))] = label_id_for("T10")
    out = check_segmentation(seg, SPACING)
    assert out["levels"]["T12"]["valid"]          # interior vertebra recovered
    assert not out["levels"]["T10"]["valid"]       # truncated edge level blamed
    assert out["levels"]["T10"]["edge_truncated"]


def test_speck_volume_rejected():
    seg = _well_formed()
    # add a tiny "L4" speck (a few voxels) -> implausibly small
    X, Y, Z = _phys_grid(SHAPE, SPACING)
    seg[_ellipsoid(X, Y, Z, (30, 25, 170), (1.5, 1.5, 1.5))] = label_id_for("L4")
    out = check_segmentation(seg, SPACING)
    assert not out["levels"]["L4"]["valid"]
    assert any("small" in r for r in out["levels"]["L4"]["reasons"])


def test_suspect_from_bad_level_carries_per_level_reason_not_global():
    # A single bad level among well-formed ones makes the seg 'suspect' WITHOUT a
    # global reason -- the failure is discoverable only per level. This is the
    # real case (Anon2's speck T8 / ballooned sacrum) the CLI summary must still
    # explain instead of printing a blank reason.
    seg = _well_formed()
    X, Y, Z = _phys_grid(SHAPE, SPACING)
    seg[_ellipsoid(X, Y, Z, (30, 25, 170), (1.5, 1.5, 1.5))] = label_id_for("L4")
    out = check_segmentation(seg, SPACING)
    assert out["seg_status"] == "suspect"
    assert out["global_reasons"] == []
    assert out["levels"]["L4"]["reasons"]
