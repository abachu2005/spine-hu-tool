import sys, numpy as np, nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion
from totalsegmentator.map_to_binary import class_map
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

cm = class_map["total"]
name_by_id = cm

VERT_ORDER = ["C1","C2","C3","C4","C5","C6","C7",
              "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
              "L1","L2","L3","L4","L5","S1"]

def vert_labels_present(S):
    out = []
    for lab in np.unique(S):
        if lab == 0:
            continue
        nm = name_by_id.get(int(lab), "")
        if nm.startswith("vertebrae_") or nm == "sacrum":
            short = nm.replace("vertebrae_", "")
            out.append((int(lab), short))
    return out

def ellipsoid(mm, sx, sy, sz):
    rx, ry, rz = max(1,int(round(mm/sx))), max(1,int(round(mm/sy))), max(1,int(round(mm/sz)))
    zz, yy, xx = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return ((xx*sx)**2 + (yy*sy)**2 + (zz*sz)**2) <= mm**2

def analyze(study):
    hu = nib.load(f"work/{study}.nii.gz"); H = np.asarray(hu.dataobj).astype(np.float32)
    seg = nib.load(f"work/{study}_seg.nii"); S = np.asarray(seg.dataobj).astype(np.int16)
    sx, sy, sz = [float(v) for v in hu.header.get_zooms()[:3]]
    metal = H > 2500
    buf = binary_dilation(metal, structure=ellipsoid(10, sx, sy, sz)) if metal.any() else metal
    er4 = ellipsoid(4, sx, sy, sz)

    present = vert_labels_present(S)
    present.sort(key=lambda t: VERT_ORDER.index(t[1]) if t[1] in VERT_ORDER else 99)
    rows = []
    for lab, short in present:
        m = S == lab
        n = int(m.sum())
        metal_in = int((m & metal).sum())
        near = float((m & buf).sum()) / n * 100 if n else 0
        core = binary_erosion(m, structure=er4)
        huv = H[core] if core.any() else H[m]
        trab = huv[(huv > -50) & (huv < 400)]
        hu_mean = float(trab.mean()) if trab.size else float("nan")
        hu_sd = float(trab.std()) if trab.size else float("nan")
        status = "CONTAMINATED" if (metal_in > 0 or near > 2) else "clean"
        rows.append((short, n, metal_in, near, hu_mean, hu_sd, status))
    return H, S, metal, sx, sy, sz, rows, present

def render(study, H, S, metal, sx, sy, sz, present):
    # NIfTI axis order: axis0 = R-L (X), axis1 = A-P (Y), axis2 = S-I (Z)
    vmask = S > 0
    xs = np.where(vmask.any(axis=(1, 2)))[0]
    xc = int(round(xs.mean())) if xs.size else H.shape[0] // 2
    # sagittal slice at spine center -> shape (Y, Z); transpose to (Z, Y) so S-I is vertical
    sagH = H[xc, :, :].T
    sagS = S[xc, :, :].T
    sagM = metal[xc, :, :].T
    aspect = sy / sz  # horizontal px = sy mm, vertical px = sz mm
    fig, axes = plt.subplots(1, 2, figsize=(9, 11))
    for ax in axes:
        ax.imshow(sagH, cmap="gray", vmin=-200, vmax=1200, aspect=sz / sy, origin="lower")
    over = np.zeros(sagS.shape)
    for i, (lab, short) in enumerate(present):
        over[sagS == lab] = i + 1
    cmap = plt.get_cmap("tab20", max(2, len(present) + 1))
    axes[1].imshow(np.ma.masked_where(over == 0, over), cmap=cmap, alpha=0.45,
                   aspect=sz / sy, origin="lower", vmin=0, vmax=len(present))
    if sagM.any():
        axes[1].imshow(np.ma.masked_where(~sagM, sagM), cmap="autumn", alpha=1.0,
                       aspect=sz / sy, origin="lower")
    for lab, short in present:
        rr, cc = np.where(sagS == lab)
        if rr.size:
            axes[1].text(cc.mean(), rr.mean(), short, color="cyan", fontsize=8,
                         ha="center", va="center", weight="bold")
    axes[0].set_title(f"{study}: mid-sagittal"); axes[0].axis("off")
    axes[1].set_title(f"{study}: vertebra labels + metal(red)"); axes[1].axis("off")
    fig.tight_layout(); p = f"scratch/{study}_levels.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p

if __name__ == "__main__":
    for study in ["lumbar","thoracic"]:
        H,S,metal,sx,sy,sz,rows,present = analyze(study)
        print(f"\n===== {study.upper()}  (voxel {sx:.2f}x{sy:.2f}x{sz:.2f} mm) =====")
        print(f"{'lvl':5s} {'vox':>7s} {'metal_in':>8s} {'near10mm%':>9s} {'coreHU':>7s} {'sd':>5s}  status")
        for short,n,mi,near,hm,hsd,st in rows:
            print(f"{short:5s} {n:7d} {mi:8d} {near:9.1f} {hm:7.0f} {hsd:5.0f}  {st}")
        p = render(study,H,S,metal,sx,sy,sz,present)
        print("rendered", p)
