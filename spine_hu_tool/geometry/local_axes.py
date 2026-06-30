"""Per-vertebra local coordinate frame.

Defines a local (LR, AP, SI) frame for an isolated vertebral body so that
"centroid" and "central height" are meaningful on a tilted body (kyphosis,
lordosis, scoliosis) rather than relying on the global image axes.

Method: PCA on the physical coordinates of the body voxels yields three
principal directions. Each is then assigned to the global axis it is most
aligned with (x = LR, y = AP, z = SI) and sign-corrected to point in the
positive global direction. This produces a tilt-corrected frame that is a
small rotation of the global axes -- robust and reproducible.
"""
from __future__ import annotations
import numpy as np

from ..core import LocalAxes


def compute_local_axes(body_mask: np.ndarray, spacing) -> LocalAxes:
    idx = np.argwhere(body_mask)
    if idx.shape[0] < 10:
        # degenerate: fall back to global axes
        c = idx.mean(0) if idx.size else np.array(body_mask.shape) / 2
        return LocalAxes(centroid=np.round(c).astype(int),
                         lr=np.array([1.0, 0, 0]),
                         ap=np.array([0, 1.0, 0]),
                         si=np.array([0, 0, 1.0]))

    phys = idx * np.asarray(spacing)
    centroid_phys = phys.mean(0)
    cov = np.cov((phys - centroid_phys).T)
    _vals, vecs = np.linalg.eigh(cov)        # columns are eigenvectors
    pcs = [vecs[:, i] for i in range(3)]

    # Assign each global axis to the principal component most aligned with it.
    global_axes = {0: np.array([1.0, 0, 0]),   # LR (x)
                   1: np.array([0, 1.0, 0]),    # AP (y)
                   2: np.array([0, 0, 1.0])}    # SI (z)
    assigned = {}
    used = set()
    # greedy assignment by descending alignment strength
    pairs = []
    for g, gax in global_axes.items():
        for j, pc in enumerate(pcs):
            pairs.append((abs(float(np.dot(pc, gax))), g, j))
    pairs.sort(reverse=True)
    for _strength, g, j in pairs:
        if g in assigned or j in used:
            continue
        pc = pcs[j]
        if np.dot(pc, global_axes[g]) < 0:     # sign-correct
            pc = -pc
        assigned[g] = pc
        used.add(j)

    centroid_idx = np.round(centroid_phys / np.asarray(spacing)).astype(int)
    return LocalAxes(centroid=centroid_idx,
                     lr=assigned[0], ap=assigned[1], si=assigned[2])


def project_onto_axis(idx_points: np.ndarray, axis: np.ndarray,
                      centroid_idx: np.ndarray, spacing) -> np.ndarray:
    """Signed physical distance of each voxel index from the centroid along `axis`."""
    phys = (idx_points - centroid_idx) * np.asarray(spacing)
    return phys @ np.asarray(axis)
