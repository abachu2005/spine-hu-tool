import numpy as np
from spine_hu_tool.geometry.coords import (make_sphere, within_mm,
                                           index_to_physical, physical_to_index)


def test_sphere_volume_matches_analytic():
    spacing = (1.0, 1.0, 1.0)
    shape = (61, 61, 61)
    r = 10.0
    s = make_sphere(shape, (30, 30, 30), r, spacing)
    analytic = 4.0 / 3.0 * np.pi * r ** 3
    assert abs(s.sum() - analytic) / analytic < 0.05


def test_sphere_anisotropic():
    spacing = (0.5, 0.5, 2.0)
    shape = (81, 81, 41)
    r = 8.0
    s = make_sphere(shape, (40, 40, 20), r, spacing)
    vol = s.sum() * (0.5 * 0.5 * 2.0)
    analytic = 4.0 / 3.0 * np.pi * r ** 3
    assert abs(vol - analytic) / analytic < 0.10


def test_within_mm():
    spacing = (1.0, 1.0, 1.0)
    m = np.zeros((21, 21, 21), bool)
    m[10, 10, 10] = True
    near = within_mm(m, 3.0, spacing)
    assert near[10, 10, 13] and not near[10, 10, 14]


def test_index_physical_roundtrip():
    spacing = (0.7, 0.7, 2.5)
    idx = np.array([10, 20, 5])
    phys = index_to_physical(idx, spacing)
    back = physical_to_index(phys, spacing)
    assert np.allclose(back, idx)
