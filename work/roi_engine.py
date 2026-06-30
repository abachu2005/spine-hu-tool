"""Vertebral-body isolation + distance-transform ROI engine (prototype).

Works in native anisotropic voxel spacing so HU is never interpolated.
NIfTI axis convention here: axis0 = R-L (X), axis1 = A-P (Y), axis2 = S-I (Z).
Patient is FFS/LPS, so increasing Y (axis1) = posterior -> the vertebral
body is the ANTERIOR (low-Y) portion of the full vertebra mask.
"""
import sys, json
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_erosion, binary_dilation, binary_closing,
                           binary_fill_holes, label, distance_transform_edt,
                           center_of_mass)
from totalsegmentator.map_to_binary import class_map

CM = class_map["total"]
ID_BY_NAME = {v: k for k, v in CM.items()}


def ball(r_mm, spacing):
    sx, sy, sz = spacing
    rx, ry, rz = (max(1, int(round(r_mm / sx))),
                  max(1, int(round(r_mm / sy))),
                  max(1, int(round(r_mm / sz))))
    xx, yy, zz = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    return ((xx * sx) ** 2 + (yy * sy) ** 2 + (zz * sz) ** 2) <= r_mm ** 2


def largest_cc(mask):
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def isolate_body(full_mask, spacing, open_mm=5.0):
    """Strip posterior elements via morphological opening, then keep the most
    anterior large connected component and recover its volume within the mask."""
    M = full_mask.astype(bool)
    b = ball(open_mm, spacing)
    opened = binary_erosion(M, b)
    opened = largest_cc(opened) if opened.any() else opened
    # among eroded components pick the most anterior (lowest mean Y / axis1)
    lab, n = label(binary_erosion(M, b))
    if n >= 1:
        best, best_y = None, 1e9
        for i in range(1, n + 1):
            comp = lab == i
            if comp.sum() < 50:
                continue
            ymean = np.argwhere(comp)[:, 1].mean()
            if ymean < best_y:
                best_y, best = ymean, comp
        if best is not None:
            opened = best
    # recover body volume: dilate back by the opening radius, constrained to M
    body = binary_dilation(opened, b) & M
    body = binary_fill_holes(body)
    body = largest_cc(body)
    return body


def make_inner(body, spacing, margin_mm=3.0):
    dist = distance_transform_edt(body, sampling=spacing)
    inner = dist >= margin_mm
    return inner, dist


def restrict_central_height(mask, body, frac=0.60):
    zs = np.argwhere(body)[:, 2]
    z0, z1 = zs.min(), zs.max()
    h = z1 - z0
    lo = z0 + (1 - frac) / 2 * h
    hi = z1 - (1 - frac) / 2 * h
    out = mask.copy()
    out[:, :, :int(np.floor(lo))] = False
    out[:, :, int(np.ceil(hi)) + 1:] = False
    return out


def sphere_mask(shape, center, radius_mm, spacing):
    sx, sy, sz = spacing
    zz = np.arange(shape[2])
    X, Y, Z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), zz, indexing="ij")
    d2 = ((X - center[0]) * sx) ** 2 + ((Y - center[1]) * sy) ** 2 + ((Z - center[2]) * sz) ** 2
    return d2 <= radius_mm ** 2


def place_roi(body, spacing, target_radius_mm=5.0, margin_mm=3.0, min_radius_mm=3.0):
    inner, dist_body = make_inner(body, spacing, margin_mm)
    central = restrict_central_height(inner, body, 0.60)
    if not central.any():
        return None
    dist_inner = distance_transform_edt(central, sampling=spacing)
    center = np.unravel_index(np.argmax(dist_inner), dist_inner.shape)
    max_safe = float(dist_inner[center])
    radius = min(target_radius_mm, max_safe)
    roi = sphere_mask(body.shape, center, radius, spacing) & inner
    return dict(roi=roi, center=center, radius_mm=radius, max_safe_radius_mm=max_safe,
                inner=inner, dist_body=dist_body, min_radius_mm=min_radius_mm)


def hu_stats(H, roi, spacing):
    v = H[roi]
    vol = float(v.size * np.prod(spacing))
    return dict(mean_HU=float(v.mean()), median_HU=float(np.median(v)),
                sd_HU=float(v.std()), min_HU=float(v.min()), max_HU=float(v.max()),
                p05_HU=float(np.percentile(v, 5)), p95_HU=float(np.percentile(v, 95)),
                voxel_count=int(v.size), volume_mm3=vol)


def qc(H, body, roi, info, spacing):
    warnings = []
    sphere = sphere_mask(body.shape, info["center"], info["radius_mm"], spacing)
    retained = float(roi.sum() / max(1, sphere.sum()))
    min_margin = float(info["dist_body"][roi].min())
    metal = H > 2500
    near_metal = bool((binary_dilation(metal, ball(10, spacing)) & roi).any()) if metal.any() else False
    std_hu = float(H[roi].std())
    if info["max_safe_radius_mm"] < info["min_radius_mm"]:
        warnings.append("max safe radius below minimum")
    if retained < 0.95:
        warnings.append(f"ROI clipped (retained {retained:.2f})")
    if min_margin < spacing[1]:
        warnings.append("ROI touches body boundary")
    if near_metal:
        warnings.append("ROI near metal artifact")
    if std_hu > 150:
        warnings.append("high ROI heterogeneity")
    status = "pass" if not warnings else "review"
    return dict(qc_status=status, warnings=warnings, retained_fraction=retained,
                min_margin_mm=min_margin, near_metal=near_metal)


def crop_bbox(mask, spacing, pad_mm=20.0):
    idx = np.argwhere(mask)
    lo = idx.min(0); hi = idx.max(0) + 1
    pad = np.array([int(round(pad_mm / s)) for s in spacing])
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return sl


def run(study, level):
    hu_img = nib.load(f"work/{study}.nii.gz")
    Hf = np.asarray(hu_img.dataobj).astype(np.float32)
    Sf = np.asarray(nib.load(f"work/{study}_seg.nii").dataobj).astype(np.int16)
    spacing = tuple(float(v) for v in hu_img.header.get_zooms()[:3])
    lab_id = ID_BY_NAME[f"vertebrae_{level}"]
    full_f = Sf == lab_id
    if not full_f.any():
        print(f"{study}/{level}: level not present"); return None
    sl = crop_bbox(full_f, spacing)
    H = Hf[sl]; full = full_f[sl]
    body = isolate_body(full, spacing)
    info = place_roi(body, spacing)
    if info is None:
        print(f"{study}/{level}: ROI placement failed"); return None
    stats = hu_stats(H, info["roi"], spacing)
    q = qc(H, body, info["roi"], info, spacing)
    result = dict(study=study, level=level, spacing_mm=spacing,
                  roi_center_idx=[int(c) for c in info["center"]],
                  roi_radius_mm=round(info["radius_mm"], 2),
                  max_safe_radius_mm=round(info["max_safe_radius_mm"], 2),
                  full_vox=int(full.sum()), body_vox=int(body.sum()),
                  **{k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()},
                  **q)
    return result, H, full, body, info, spacing


if __name__ == "__main__":
    study = sys.argv[1] if len(sys.argv) > 1 else "thoracic"
    level = sys.argv[2] if len(sys.argv) > 2 else "L1"
    out = run(study, level)
    if out:
        print(json.dumps(out[0], indent=2))
