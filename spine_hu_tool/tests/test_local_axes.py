import numpy as np
from spine_hu_tool.geometry.local_axes import compute_local_axes
from .synthetic import make_tilted_ellipsoid


def _angle(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(abs(np.clip(a @ b, -1, 1))))


def test_si_axis_recovers_tilt():
    mask, true_si = make_tilted_ellipsoid(tilt_deg=20.0)
    axes = compute_local_axes(mask, (1.0, 1.0, 1.0))
    assert _angle(axes.si, true_si) < 6.0


def test_axes_orthonormal_and_assigned():
    mask, _ = make_tilted_ellipsoid(tilt_deg=10.0)
    axes = compute_local_axes(mask, (1.0, 1.0, 1.0))
    for v in (axes.lr, axes.ap, axes.si):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    # SI should be nearest the global z; LR nearest global x
    assert abs(axes.si[2]) > abs(axes.si[0]) and abs(axes.si[2]) > abs(axes.si[1])
    assert abs(axes.lr[0]) > abs(axes.lr[1]) and abs(axes.lr[0]) > abs(axes.lr[2])
