import numpy as np
from spine_hu_tool.measurement.level_qc import tag_levels, clean_levels
from spine_hu_tool.segmentation.totalseg_runner import label_id_for


def _two_level_seg():
    seg = np.zeros((20, 20, 24), dtype=np.int16)
    l1 = label_id_for("L1")
    l2 = label_id_for("L2")
    seg[5:15, 5:15, 12:22] = l1     # superior
    seg[5:15, 5:15, 2:12] = l2      # inferior
    return seg, l1, l2


def test_no_metal_all_clean():
    seg, _l1, _l2 = _two_level_seg()
    hu = np.full(seg.shape, 150.0, dtype=np.float32)
    tags = tag_levels(hu, seg, (2.0, 2.0, 2.0))
    assert set(clean_levels(tags)) == {"L1", "L2"}


def test_metal_excludes_and_buffers_neighbor():
    seg, _l1, _l2 = _two_level_seg()
    hu = np.full(seg.shape, 150.0, dtype=np.float32)
    hu[9, 9, 17] = 3000.0    # metal inside L1
    tags = tag_levels(hu, seg, (2.0, 2.0, 2.0))
    by = {t["level"]: t["status"] for t in tags}
    assert by["L1"] == "excluded"
    assert by["L2"] == "review"      # adjacent buffer
    assert "L1" not in clean_levels(tags)
