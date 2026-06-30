import numpy as np
from spine_hu_tool.roi.body_isolation import isolate_body
from .synthetic import make_vertebra_phantom


def test_removes_posterior_elements():
    full, body_true, _hu = make_vertebra_phantom()
    body, flags = isolate_body(full, (1.0, 1.0, 1.0))
    # posterior blob center (y=50) must be gone; anterior body (y=22) kept
    assert not body[30, 50, 25]
    assert body[30, 22, 25]
    # recovered body overlaps the true body strongly
    inter = (body & body_true).sum()
    assert inter / body_true.sum() > 0.85
    # and excludes most of the posterior elements
    assert body.sum() < full.sum() * 0.85


def test_edge_touching_flagged():
    # partial is judged on the isolated body, so the body itself must reach the
    # edge: a thick anterior block (survives erosion) bridging to the x=0 face.
    full, _b, _hu = make_vertebra_phantom()
    full[0:16, 12:32, 16:34] = True
    body, flags = isolate_body(full, (1.0, 1.0, 1.0))
    assert flags.get("partial") is True


def test_transverse_process_at_edge_not_partial():
    # a thin posterior process touching the edge must NOT flag the body partial
    full, _b, _hu = make_vertebra_phantom()
    full[0, 48:54, 22:28] = True     # thin posterior-element sliver at x=0
    body, flags = isolate_body(full, (1.0, 1.0, 1.0))
    assert not flags.get("partial")
