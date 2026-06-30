"""Segmentation: the only ML component (pretrained TotalSegmentator).

It can run locally (subprocess) or be offloaded to a remote service; use
:func:`segment` to dispatch transparently.
"""
from .totalseg_runner import (run_segmentation, load_segmentation,
                              vertebra_labels, label_id_for, VERT_ORDER)
from .backends import segment, resolve_seg_url, resolve_api_key, DEFAULT_SEG_URL
from .consistency import check_segmentation

__all__ = ["run_segmentation", "load_segmentation", "vertebra_labels",
           "label_id_for", "VERT_ORDER", "segment", "resolve_seg_url",
           "resolve_api_key", "DEFAULT_SEG_URL", "check_segmentation"]
