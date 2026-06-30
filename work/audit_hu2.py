"""Independent HU audit (live).

Runs the real pipeline once, then for every measured level:
  - pulls the ACTUAL ROI voxels the pipeline used and reports their HU
    distribution + cortical-contamination fraction,
  - independently samples a deep trabecular core (erode the body mask >=5 mm,
    no HU logic at all) and a cortical rim,
and cross-checks the two. Also verifies global HU calibration (air, soft tissue).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from spine_hu_tool.app.analysis import analyze_dataset

FOLDER = "Test Data/100007EC"
state = analyze_dataset(FOLDER, mode="centroid_volume_sphere")
hu, sp = state.volume.hu, state.volume.spacing

print("=== GLOBAL HU CALIBRATION ===")
print(f"shape={hu.shape} spacing={tuple(round(s,2) for s in sp)} "
      f"min={hu.min():.0f} max={hu.max():.0f}")
air = hu[hu < -500]
soft = hu[(hu > -100) & (hu < 100)]
print(f"air median (expect ~ -1000): {np.median(air):.0f}")
print(f"soft-tissue band median (expect ~ 0-60): {np.median(soft):.0f}")

print("\n=== PER-LEVEL: PIPELINE ROI vs INDEPENDENT DEEP CORE ===")
hdr = (f"{'lvl':4} {'stat':6} | {'ROI med':>7} {'mean':>5} {'p5':>4} {'p95':>5} "
       f"{'max':>5} {'%>300':>6} {'%>400':>6} | {'core med':>8} {'rim med':>7}  note")
print(hdr); print("-" * len(hdr))
for lvl, r in state.results.items():
    st = r.qc.get("qc_status")
    if st == "excluded" or not r.stats:
        print(f"{lvl:4} {st:6} | (no ROI)")
        continue
    sl = r.crop_slices
    h = hu[sl]
    roi = r.roi_mask
    rv = h[roi]
    med = np.median(rv); mean = rv.mean()
    p5, p95 = np.percentile(rv, [5, 95]); mx = rv.max()
    f300 = 100.0 * (rv > 300).mean()
    f400 = 100.0 * (rv > 400).mean()

    body = r.body_mask
    dist = distance_transform_edt(body, sampling=sp)
    core = dist >= 5.0
    if core.sum() < 20:
        core = dist >= 3.0
    rim = body & (dist < 1.5)
    core_med = np.median(h[core]) if core.sum() else float("nan")
    rim_med = np.median(h[rim]) if rim.sum() else float("nan")

    note = ""
    if rim_med < core_med:
        note += "RIM<CORE? "
    if abs(med - core_med) > 60:
        note += f"ROIvsCORE Δ{med-core_med:+.0f} "
    if f400 > 2:
        note += "cortical-tail "
    print(f"{lvl:4} {st:6} | {med:7.0f} {mean:5.0f} {p5:4.0f} {p95:5.0f} {mx:5.0f} "
          f"{f300:6.1f} {f400:6.1f} | {core_med:8.0f} {rim_med:7.0f}  {note}")

print("\ncore = voxels >=5 mm inside the vertebral body (independent trabecular")
print("sample, no HU search); rim = outer ~1.5 mm shell (≈cortex, should read higher).")
