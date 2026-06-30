import numpy as np
from spine_hu_tool.preprocessing import apply_rescale


def test_rescale_math():
    raw = np.array([0, 100, 1000], dtype=np.int16)
    hu = apply_rescale(raw, slope=1.0, intercept=-1024.0)
    assert hu.tolist() == [-1024.0, -924.0, -24.0]


def test_rescale_slope():
    raw = np.array([10, 20])
    hu = apply_rescale(raw, slope=2.0, intercept=0.0)
    assert hu.tolist() == [20.0, 40.0]
