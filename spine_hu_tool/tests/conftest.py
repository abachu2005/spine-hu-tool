import os
import pytest

# project root = three levels up from this file
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _exists(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


@pytest.fixture
def cached_thoracic():
    vol = os.path.join(ROOT, "work", "thoracic.nii.gz")
    seg = os.path.join(ROOT, "work", "thoracic_seg.nii")
    if not (os.path.exists(vol) and os.path.exists(seg)):
        pytest.skip("cached thoracic NIfTI not available")
    from spine_hu_tool.io import load_volume_from_nifti
    from spine_hu_tool.segmentation import load_segmentation
    return load_volume_from_nifti(vol), load_segmentation(seg)


@pytest.fixture
def test_data_dir():
    d = os.path.join(ROOT, "Test Data")
    if not os.path.isdir(d):
        pytest.skip("Test Data folder not available")
    return d
