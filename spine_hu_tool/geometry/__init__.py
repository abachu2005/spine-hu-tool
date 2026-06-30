"""Geometry helpers: coordinate transforms, morphology, per-vertebra axes."""
from .coords import (index_to_physical, physical_to_index, ball, dilate_mm,
                     erode_mm, within_mm, crop_bbox, largest_cc, make_sphere)
from .local_axes import compute_local_axes, project_onto_axis

__all__ = [
    "index_to_physical", "physical_to_index", "ball", "dilate_mm", "erode_mm",
    "within_mm", "crop_bbox", "largest_cc", "make_sphere",
    "compute_local_axes", "project_onto_axis",
]
