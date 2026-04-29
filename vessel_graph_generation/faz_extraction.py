"""
Extract FAZ center and radius from a 3D geometry volume (vessel / tissue mask).

The largest in-plane avascular region (no tissue along the projection axis) is treated as the FAZ,
matching the largest-component idea used for OCTA-based FAZ masks.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import label


def select_largest_component(arr: np.ndarray) -> np.ndarray:
    labeled_arr, num_features = label(arr)
    if num_features == 0:
        return np.zeros_like(arr, dtype=bool)
    sizes = np.bincount(labeled_arr.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    return labeled_arr == largest_label


def faz_area_matched_radius(mask: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    Centroid-centered disk with pixel count matched to the mask; returns radius (px) and
    (mean index axis0, mean index axis1) i.e. (x, y) voxel indices for simulation_space order.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx0, idx1 = np.nonzero(mask)
    c0 = float(idx0.mean())
    c1 = float(idx1.mean())
    h, w = mask.shape
    Y, X = np.ogrid[0:h, 0:w]
    dist_sq = (Y - c0) ** 2 + (X - c1) ** 2
    sorted_d = np.sort(dist_sq.ravel())

    def count_leq(t: float) -> int:
        return int(np.searchsorted(sorted_d, t, side="right"))

    lo, hi_i = 0, len(sorted_d) - 1
    idx_hi = hi_i
    while lo <= hi_i:
        mid = (lo + hi_i) // 2
        if count_leq(sorted_d[mid]) >= n:
            idx_hi = mid
            hi_i = mid - 1
        else:
            lo = mid + 1

    candidates = [sorted_d[idx_hi]]
    if idx_hi > 0:
        candidates.append(sorted_d[idx_hi - 1])
    best_r_sq = min(candidates, key=lambda t: (abs(count_leq(t) - n), t))
    r = float(np.sqrt(best_r_sq))
    return r, (c0, c1)


def load_raw_geometry(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        import nibabel as nib

        return nib.load(path).get_fdata()
    raise ValueError(f"Unsupported geometry format: {path}")


def extract_faz_center_radius_unit(
    raw_geometry: np.ndarray,
    proj_axis: int = 2,
    geometry_size: int | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    raw_geometry : ndarray
        3D label or binary mask; nonzero voxels are treated as tissue / allowed vasculature.
    proj_axis : int
        Axis along which to OR-reduce (typical OCT slab: 2 for the through-depth axis).
    geometry_size : int, optional
        Normalization length; defaults to ``max(raw_geometry.shape)`` (same as SimulationSpace).

    Returns
    -------
    center_xy : (2,) ndarray
        FAZ center in unit cube coordinates (x, y), consistent with ``Greenhouse`` / ``node.position``.
    radius : float
        Area-matched circle radius in the same unit system (distance comparable to ``FAZ_radius`` from config).
    """
    if raw_geometry.ndim != 3:
        raise ValueError(f"Expected 3D geometry, got shape {raw_geometry.shape}")

    ax = proj_axis % 3
    if geometry_size is None:
        geometry_size = int(max(raw_geometry.shape))

    occupied = raw_geometry > 0
    proj_2d = np.any(occupied, axis=ax)
    void = ~proj_2d
    faz_mask = select_largest_component(void)
    r_px, (cx_idx, cy_idx) = faz_area_matched_radius(faz_mask)

    center_unit = np.array([cx_idx, cy_idx], dtype=float) / float(geometry_size)
    radius_unit = float(r_px / float(geometry_size))
    return center_unit, radius_unit
