# `tests/` — how we keep the ruler honest

Because the measurement is deterministic, it is **testable** — and that is one of
the main payoffs of the "math for the ruler" philosophy. Run:

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

## Two tiers of tests

1. **Synthetic-phantom unit tests (no ML, always run).** `synthetic.py` builds
   phantoms with *known* geometry and HU, so we can assert exact properties:
   the HU rescale math, local-axis orientation, body isolation, ROI containment
   (fully inside the body, off the endplates), cylinder/sphere shape, QC flags,
   calibration factors, the consistency gate, export round-trips, and a headless
   GUI smoke test. Deterministic geometry means these can check *correctness*,
   not just "it ran".
2. **Integration & series-selection tests (use cached data, skip if absent).**
   These exercise the real pipeline on cached test volumes/masks and validate the
   series chooser against the project's known lumbar/thoracic series. They skip
   gracefully when the data isn't present, so the suite stays green anywhere.

## Why phantoms instead of only real scans

Real scans have no ground truth for HU, so they can't tell you whether a change
*broke* the measurement. A phantom with a known trabecular value and known
cortex/endplates lets a test say "the ROI must sit here, contain this many
voxels, and read this HU" — turning regressions into failing tests instead of
silent drift.

| File | Focus |
|------|-------|
| `synthetic.py`, `conftest.py` | Phantom builders + fixtures. |
| `test_hu_conversion.py` | HU rescale math. |
| `test_local_axes.py`, `test_coords.py` | Local frame + geometry primitives. |
| `test_body_isolation.py` | Posterior-element stripping. |
| `test_roi.py`, `test_modes.py` | ROI placement, containment, modes. |
| `test_qc.py`, `test_level_qc.py` | QC and metal/streak tagging. |
| `test_calibration.py` | kVp/scanner factors + context. |
| `test_consistency.py` | Segmentation consistency gate. |
| `test_export_roundtrip.py` | Export/mask round-trips. |
| `test_review_state.py` | Shared GUI/CLI state model. |
| `test_series_selector.py` | Series ranking/grouping. |
| `test_backends.py`, `test_server.py` | Backend selection + FastAPI service. |
| `test_integration.py`, `test_gui_smoke.py` | End-to-end + headless GUI. |
