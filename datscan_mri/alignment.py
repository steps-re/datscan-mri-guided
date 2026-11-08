"""SPECT-to-anatomy alignment check.

The registration QC (ventricle Dice, overlay figure) checks the MRI against
the CT. Nothing there checks the SPECT against the anatomy, and the
posterior/anterior putamen ratio is very sensitive to a front-to-back offset
between the SPECT and the MRI-defined VOIs.

This module searches integer-mm translations of the SPECT (on the 1-mm
analysis grid) for the one that maximises the Pearson correlation between the
patient SPECT and a simulated null (blurred anatomy, no posterior loss) inside
a dilated mask. The caudate-only mask is the primary fit: the caudate is not
part of the posterior-putamen hypothesis, so a putamen pattern cannot pull it.
The putamen-including masks are reported for comparison and ARE biased by the
putamen pattern if there is a real loss.

Convention: a shift d = (dz, dy, dx) voxels moves the SPECT CONTENT by +d,
i.e. shifted[i] = spect[i - d]. On the LPS analysis grid +x = patient left,
+y = posterior, +z = superior. A best shift of +4 in y means the SPECT matches
the anatomy best after being moved 4 mm posterior, i.e. as registered it sits
about 4 mm anterior of the MRI anatomy.

On a 1-mm grid an integer translation of the resampled image equals
resampling the original SPECT with that translation (linear interpolation at
the shifted points), so nothing is approximated here.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage as nd


def shifted_crop(full: np.ndarray, crop: tuple, d, fill=0):
    """`full[crop]` with the content moved by d = (dz, dy, dx) voxels:
    out[i] = full[crop_start + i - d]; `fill` where that falls off the grid."""
    starts = np.array([s.start for s in crop])
    shape = np.array([min(s.stop, n) - s.start for s, n in zip(crop, full.shape)])   # as full[crop]
    return _window(full, starts, shape, np.asarray(d, int), fill)


def _window(full, starts, shape, d, fill):
    out = np.full(tuple(shape), fill, full.dtype)
    src0 = starts - d
    lo = np.maximum(src0, 0)
    hi = np.minimum(src0 + shape, full.shape)
    if np.any(hi <= lo):
        return out
    dst = tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, src0))
    out[dst] = full[tuple(slice(a, b) for a, b in zip(lo, hi))]
    return out


def best_shift(full: np.ndarray, crop: tuple, target: np.ndarray, mask: np.ndarray, search_mm: int):
    """Integer shift in [-search_mm, search_mm]^3 maximising the Pearson
    correlation of the shifted SPECT with `target` (crop frame) over `mask`.
    Returns (shift (dz, dy, dx), correlation, correlation at zero shift)."""
    idx = np.argwhere(mask)
    b0, b1 = idx.min(0), idx.max(0) + 1
    n = b1 - b0
    R = int(search_mm)
    starts = np.array([s.start for s in crop]) + b0
    big = _window(full, starts - R, n + 2 * R, np.zeros(3, int), 0).astype(np.float64)
    m = mask[tuple(slice(a, b) for a, b in zip(b0, b1))]
    U = target[tuple(slice(a, b) for a, b in zip(b0, b1))][m].astype(np.float64)
    Un = (U - U.mean()) / U.std()
    best, c0 = (-np.inf, (0, 0, 0)), np.nan
    for d in itertools.product(range(-R, R + 1), repeat=3):
        w = big[R - d[0]:R - d[0] + n[0], R - d[1]:R - d[1] + n[1], R - d[2]:R - d[2] + n[2]]
        x = w[m]
        sd = x.std()
        c = float(((x - x.mean()) @ Un) / (x.size * sd)) if sd > 0 else -np.inf
        if d == (0, 0, 0):
            c0 = c
        if c > best[0]:
            best = (c, d)
    return tuple(int(v) for v in best[1]), best[0], c0


def check(sg_full, fov_full, crop, lab, targets: dict, measure, exp_ng_low: dict, A: dict) -> dict:
    """Run the translation search for every mask in A["masks"] and every
    simulated null in `targets` ({name: crop-frame volume}); measure P/A at
    each best-fit shift with `measure(s_crop, lab, valid_crop) -> (R, L)`, the
    same function as the main measurement, so at zero shift it is identical.

    exp_ng_low: {side: lowest normal-gradient expectation} for the gap."""
    out = {"method": ("integer-mm translation of the SPECT on the analysis grid maximising the Pearson "
                      "correlation with a simulated null inside the named regions dilated by "
                      f"{A['dilate_mm']} mm (6-connected); search +/-{A['search_mm']} mm per axis. "
                      "Shift = movement applied to the SPECT content, LPS grid: +x left, +y posterior, "
                      "+z superior. P/A is re-measured with the main measurement at the shifted SPECT "
                      "(FOV mask moved with it); the simulated expectation is anatomy-only and unchanged."),
           "primary_mask": A["primary_mask"], "primary_target": A["primary_target"],
           "search_mm": A["search_mm"], "dilate_mm": A["dilate_mm"], "fits": []}
    s0 = sg_full[crop]
    v0 = fov_full[crop]
    m0 = measure(s0, lab, v0)
    out["as_registered"] = {"post_ant": dict(zip("RL", m0)),
                            "gap_vs_normal_gradient_low": {sd: m0[i] - exp_ng_low[sd] for i, sd in enumerate("RL")}}
    pa_cache = {}
    for mname, codes in A["masks"].items():
        mask = nd.binary_dilation(np.isin(lab, codes), iterations=A["dilate_mm"])
        for tname in A["targets"]:
            d, c, c0 = best_shift(sg_full, crop, targets[tname], mask, A["search_mm"])
            if d not in pa_cache:
                pa_cache[d] = measure(shifted_crop(sg_full, crop, d, 0), lab, shifted_crop(fov_full, crop, d, False))
            pa = pa_cache[d]
            out["fits"].append({
                "mask": mname, "mask_labels": list(codes), "target": tname,
                "best_shift_vox_zyx": list(d), "best_shift_mm_xyz": [d[2], d[1], d[0]],
                "best_shift_norm_mm": float(np.linalg.norm(d)), "posterior_shift_mm": d[1],
                "correlation": c, "correlation_at_zero_shift": c0,
                "post_ant": dict(zip("RL", pa)),
                "gap_vs_normal_gradient_low": {sd: pa[i] - exp_ng_low[sd] for i, sd in enumerate("RL")}})
    prim = [f for f in out["fits"] if f["mask"] == A["primary_mask"] and f["target"] == A["primary_target"]][0]
    out["primary"] = prim
    lim = A["warn_shift_mm"]
    out["warn_shift_mm"] = lim
    out["shift_exceeds_limit"] = bool(prim["best_shift_norm_mm"] > lim)
    if out["shift_exceeds_limit"]:
        x, y, z = prim["best_shift_mm_xyz"]
        out["warning"] = (f"SPECT-to-anatomy alignment: the SPECT best matches the simulated uptake pattern "
                          f"({A['primary_mask']} mask) after a ({x:+d}, {y:+d}, {z:+d}) mm (x, y, z) shift, "
                          f"|shift| {prim['best_shift_norm_mm']:.1f} mm > {lim:g} mm. The posterior/anterior "
                          f"ratio is very sensitive to this: as registered R {m0[0]:.3f} / L {m0[1]:.3f}, at the "
                          f"best-fit shift R {prim['post_ant']['R']:.3f} / L {prim['post_ant']['L']:.3f}. Either the "
                          "SPECT is offset from the MRI anatomy or the simulated caudate (VOI placement or its "
                          "uniform-binding model) is displaced; this check cannot tell which. Check the SPECT/CT "
                          "and MRI/CT alignment and figure 2 before reading the ratio.")
    return out

