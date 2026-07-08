# New-Data Evaluation (Anon1 / Anon2 / Anon3)

Evaluation of three newly-added studies dropped into `Test Data/`, run through the
full pipeline to check result quality and surface weaknesses. This documents what
was tested, what worked, the holes found, the fixes applied, and recommendations.

## TL;DR

- The deterministic measurement pipeline **performed well**: ROI placement is
  well-centered in trabecular bone, metal hardware is correctly quarantined, and
  FOV-clipped end levels are excluded rather than silently measured.
- The biggest hole was operational, not analytical: **full-resolution cloud
  segmentation is far too slow** and a large scan (Anon1, 254 slices) **hard-timed
  out with no result**. Root-caused to Cloud Run CPU-throttling the async worker.
- Fixed the client to never hard-fail (adaptive timeout + automatic fast-mode
  fallback, verified live) and documented the real server-side fix
  (`--no-cpu-throttling`). Also fixed a blank "segmentation suspect" CLI message
  and repaired two tests the new data broke. Added 13 new tests.

## The studies

| Study | Scanner | Series | Slices | Spacing (mm) | Nature |
|-------|---------|--------|--------|--------------|--------|
| Anon1 | GE Revolution CT | 1 axial | 254 | 0.77 x 0.77 x 2.5 | Long C7->sacrum scan; FOV-clipped ends |
| Anon2 | GE BrightSpeed | STANDARD + BONE + 2 reformats | 115 | 0.31 x 0.31 x 2.5 | Lumbar scan with spinal **hardware** |
| Anon3 | GE BrightSpeed | STANDARD + BONE + 2 reformats | 115 | 0.31 x 0.31 x 2.5 | **Byte-identical duplicate of Anon2** |

All three carry placeholder anonymizer metadata (`PatientID="DummyPatID!"`,
`SeriesDescription="DummySeriesDesc!"`, etc.).

## What worked well

1. **Series selection.** On Anon2/Anon3 (4 series each) the STANDARD-kernel axial
   CT is correctly chosen and the BONE-kernel duplicate + the two
   DERIVED/SECONDARY/REFORMATTED recons are rejected. Anon1's single series is
   chosen cleanly.
2. **ROI placement.** Overlays show the centroid sphere landing in central
   trabecular bone, clear of the cortical rind and posterior elements, across
   thoracic and lumbar levels.
3. **Metal quarantine (Anon2).** This is an instrumented spine. L3/L5/S1 (hardware
   in the body) are **excluded and carry no HU**; L2 (neighbor) is downgraded to
   **review** for streak/bloom; the levels are still shown so the hardware is
   visible. Exactly the intended behavior.
4. **FOV-clipped ends (Anon1).** C7/T2/T3 at the top and L5/S1/sacrum at the
   bottom are excluded (speck-sized / overlapping / ballooned labels), not
   passed. Interior T4->L4 measure with a plausible cranio-caudal gradient
   (T4≈198 down to T12≈83 / L1≈86 HU).
5. **No diagnosis emitted.** As designed, every measured level reports a
   calibrated HU + soft L1 context, never an osteoporosis class.

## Holes found and status

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | **Full-res cloud segmentation times out / is impractically slow.** Anon1 (254 sl) hit the 1200 s client cap and hard-failed; Anon2 (115 sl) took ~18 min. Root cause: Cloud Run throttles the async background worker between short status polls. | High | **Fixed** (client) + documented (server) |
| 2 | **Blank "SEGMENTATION SUSPECT:" message.** When a seg is suspect purely from per-level failures (Anon2's speck T8 / ballooned sacrum) `global_reasons` is empty, so the CLI printed an empty reason. | Medium | **Fixed** |
| 3 | **Two tests broke** when new data entered `Test Data/` (they hard-coded "2 studies / 1 patient"). | Medium | **Fixed** (assert invariant, not counts) |
| 4 | **Unknown-scanner calibration.** GE BrightSpeed / Revolution CT are not in the Westerhoff table, so only the (identity) kVp factor applies; no scanner correction. Surfaced honestly in the notes. | Medium | Documented (recommendation) |
| 5 | **Anonymizer PID collision.** All studies share `PatientID="DummyPatID!"`; `list_studies`/`group_series` key on `(patient_id, study_uid)`, so genuinely different patients sharing a placeholder ID would be grouped under one patient header. Studies still separate by study UID. | Low/Med | Documented (recommendation) |
| 6 | **Duplicate re-segmented.** Anon2/Anon3 are byte-identical but have different series UIDs, so the UID-keyed cache segments the same pixels twice. | Low | Documented (recommendation) |

## Fixes applied

- **Timeout robustness** ([`segmentation/backends.py`](../spine_hu_tool/segmentation/backends.py)):
  `segment()` now uses a slice-count-aware poll deadline (`_adaptive_timeout`) and,
  if a full-resolution remote job still times out, transparently **retries once in
  fast (3 mm) mode** (its mask is resampled to the full grid, so ROI placement is
  intact) instead of hard-failing. A clear progress message is surfaced. Verified
  end-to-end against the live cloud (forced full-res timeout -> fast fallback ->
  valid mask).
- **CLI reason surfacing** ([`app/cli.py`](../spine_hu_tool/app/cli.py)): new
  `_seg_status_reasons()` falls back to a per-level summary when there is no
  global reason, so the suspect/invalid banner always explains itself.
- **Deploy guide** ([`deploy/DEPLOY.md`](../deploy/DEPLOY.md)): added
  `--no-cpu-throttling` (the real server-side fix) with an explanation, plus a
  `--roi_subset` speed tip.
- **Tests** repaired to assert the series-collapse invariant instead of magic
  counts ([`test_series_selector.py`](../spine_hu_tool/tests/test_series_selector.py),
  [`test_gui_smoke.py`](../spine_hu_tool/tests/test_gui_smoke.py)).

## Tests added (13)

- `test_backends.py`: adaptive-timeout scaling; full-res->fast fallback on
  timeout; no-fallback-when-already-fast.
- `test_cli.py`: `_seg_status_reasons` prefers global, falls back to per-level,
  handles empty.
- `test_consistency.py`: a single bad level -> `suspect` with a per-level (not
  global) reason (the Anon2 case).
- `test_series_selector.py`: STANDARD axial auto-selected for Anon1/2/3, reformats
  rejected.
- `test_new_data_integration.py` (uses cached NIfTI fixtures, skips if absent):
  instrumented-level quarantine + interior HU band; suspect flag + calibration
  notes; long-scan gradient + clipped-end exclusion; no diagnosis emitted.

Fixtures are generated by
[`work/cache_anon_fixtures.py`](../work/cache_anon_fixtures.py) from a pipeline run
and live in `work/` (git-ignored). All 81 tests pass
(`QT_QPA_PLATFORM=offscreen pytest -q`).

## Reproducibility note (Anon2 vs Anon3)

Anon2 and Anon3 have different `SeriesInstanceUID`s but **identical pixel data**
(max abs diff = 0.0) and produce identical measurements. They are the same scan
re-sent as two studies, so they are a *duplicate*, not a scan-rescan pair, and do
not provide an independent reproducibility check.

## Recommendations (not done, ranked)

1. **Redeploy the segmentation service with `--no-cpu-throttling`** — the single
   highest-impact change; makes full-res fast and removes reliance on the fallback.
2. **Add GE scanner calibration factors** (BrightSpeed, Revolution CT) once
   literature/phantom values are available; until then keep surfacing "not
   calibrated for this scanner" (already done).
3. **Harden patient grouping against placeholder IDs** — when `patient_id` looks
   like a constant/blank/placeholder, fall back to grouping by study UID so
   distinct patients are never merged in the chooser.
4. **Content-hash the volume for the seg cache** (in addition to series UID) so a
   re-sent identical study isn't re-segmented.
5. **Consider `--roi_subset`** (vertebrae + sacrum) to speed up segmentation
   further, after the pilot (it changes cached mask contents).
