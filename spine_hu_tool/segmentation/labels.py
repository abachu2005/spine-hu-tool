"""Vendored vertebra/sacrum label map.

These are the TotalSegmentator v2 ``class_map["total"]`` label ids for the
sacrum and every vertebra. We vendor them so the client (which offloads
segmentation to the cloud) does NOT need TotalSegmentator -- and therefore not
torch -- installed at runtime just to look up which integer label is which
vertebra. This keeps the packaged desktop app small (hundreds of MB, not GB).

The cloud/local segmentation masks are produced by TotalSegmentator's "total"
task, so these ids match the integers stored in the cached ``*_seg.nii.gz``.
If TotalSegmentator is installed (the optional ``local-seg`` extra), nothing
here changes -- the ids are identical.
"""
from __future__ import annotations

# id -> structure name, mirroring TotalSegmentator v2 class_map["total"]
# (sacrum + S1..C1). Only the spine subset is needed; organ labels in a mask
# simply aren't found here and are ignored by the vertebra helpers.
VERTEBRA_CLASS_MAP: dict[int, str] = {
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
}
