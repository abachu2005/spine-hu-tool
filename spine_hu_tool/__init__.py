"""Spine Vertebral-HU Tool.

A deterministic, physician-reviewable pipeline that measures trabecular
Hounsfield Units (HU) in vertebral bodies from CT.

Design philosophy: "ML for the eyes, math for the ruler." The only machine
learning component is vertebra localization/segmentation (pretrained
TotalSegmentator, used off the shelf). Every measurement step -- body
isolation, local axes, ROI placement, HU statistics, and QC -- is deterministic
image processing that a physician can audit and override.

Array convention used throughout:
    arrays are indexed [x, y, z]
        axis 0 = x = Right-Left (columns)
        axis 1 = y = Anterior-Posterior (rows)
        axis 2 = z = Superior-Inferior (slices)
    spacing is (sx, sy, sz) in millimeters, matching the array axes.
This matches nibabel's on-disk order for the LPS axial volumes we work with.
"""

__version__ = "0.2.6"
