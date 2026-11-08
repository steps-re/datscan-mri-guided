"""Uptake along the putamen in coronal slabs, and VOI binding ratios.

Specific binding ratio (SBR) = (VOI mean - reference mean) / reference mean.
Reference regions: occipital cortical grey (primary, as in DaTQUANT's default)
and cerebellar grey (secondary). The posterior/anterior putamen ratio
P/A = SBR(posterior) / SBR(anterior) with the split at the coronal AC plane.
"""
from __future__ import annotations

import numpy as np

SIDES = {"R": (3, 5, 1), "L": (4, 6, 2)}  # ant, post, caudate codes
REFS = {"occ": 13, "cer": 14}


def voi_means(img, lab, valid, ids):
    """Mean of `img` over the valid voxels of each region in `ids` (NaN if empty)."""
    L = lab[valid].astype(np.int64)
    n = np.bincount(L, minlength=max(ids) + 1)
    s = np.bincount(L, weights=img[valid].astype(np.float64), minlength=max(ids) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([s[i] / n[i] if n[i] else np.nan for i in ids])


def metrics(values: dict, lab, valid=None) -> dict:
    """values: {label_code: mean activity over the VALID voxels of that region}.
    Returns SBRs per reference and side. The whole-putamen SBR weights the
    anterior and posterior VOIs by their valid voxel counts, so it equals the
    mean over the combined valid putamen voxels."""
    valid = np.ones(lab.shape, bool) if valid is None else valid
    out = {}
    for refn, rc in REFS.items():
        ref = values[rc]
        S = lambda c: (values[c] - ref) / ref  # noqa: E731
        r = {}
        for side, (a_c, p_c, c_c) in SIDES.items():
            a, p, c = S(a_c), S(p_c), S(c_c)
            va, vp = ((lab == a_c) & valid).sum(), ((lab == p_c) & valid).sum()
            put = (a * va + p * vp) / (va + vp)
            r[side] = dict(caud=float(c), ant=float(a), post=float(p), put=float(put),
                           post_ant=float(p / a), put_caud=float(put / c))
        out[refn] = r
    return out


def slab_profile(img, lab, valid, y_index, ac_y, ref_code=13, slab=2, min_voxels=20):
    """Mean SBR of each side's putamen (ant+post) in coronal slabs [y, y+slab),
    slabs aligned to the AC plane. Returns dict with slab centres (mm posterior
    to AC) and per-side SBR (NaN where a slab holds fewer than `min_voxels`
    valid voxels)."""
    ref = img[(lab == ref_code) & valid].mean()
    put_any = np.isin(lab, [3, 4, 5, 6])
    ys = y_index[put_any]
    k0 = int(np.floor((ys.min() - ac_y) / slab))
    k1 = int(np.floor((ys.max() - ac_y) / slab))
    starts = [ac_y + k * slab for k in range(k0, k1 + 1)]
    res = {"mm_from_ac": [s + slab / 2.0 - ac_y for s in starts], "slab_start_index": starts}
    for side, (a_c, p_c, _) in SIDES.items():
        put = np.isin(lab, (a_c, p_c))
        vals, ns = [], []
        for y in starts:
            mk = put & (y_index >= y) & (y_index < y + slab) & valid
            n = int(mk.sum())
            ns.append(n)
            vals.append(float((img[mk].mean() - ref) / ref) if n >= min_voxels else float("nan"))
        res[side] = vals
        res[side + "_n"] = ns
    return res


def trimmed_post_ant(img, lab, valid, y_index, trim_mm, ref_code=13):
    """P/A with the most posterior `trim_mm` coronal planes of each side's
    posterior-putamen VOI removed (tests whether the tail drives the result)."""
    ref = img[(lab == ref_code) & valid].mean()
    out = {}
    for side, (a_c, p_c, _) in SIDES.items():
        post = lab == p_c
        yend = y_index[post].max() + 1          # posterior edge of the VOI
        keep = post & (y_index < yend - trim_mm) & valid
        A = (img[(lab == a_c) & valid].mean() - ref) / ref
        Pv = (img[keep].mean() - ref) / ref
        out[side] = dict(post_ant=float(Pv / A), post=float(Pv), ant=float(A), cut_index=int(yend - trim_mm))
    return out


def scale_to_anterior(meas, sim, x, n):
    """Scale a simulated slab profile to the measured one by the voxel-weighted
    mean over the ANTERIOR window (slab centres < 0 mm, i.e. anterior to the
    AC), where neither curve is NaN. Weights are the slab voxel counts `n`.
    Returns (scaled list, factor); NaN factor if no anterior slab is usable.
    (Scaling to the peak slab would anchor on the front tip.)"""
    meas, sim, x, n = (np.asarray(v, float) for v in (meas, sim, x, n))
    w = np.where((x < 0) & np.isfinite(meas) & np.isfinite(sim), n, 0.0)
    if w.sum() <= 0:
        return [float("nan")] * len(sim), float("nan")
    k = float((w * np.nan_to_num(meas)).sum() / (w * np.nan_to_num(sim)).sum())
    return (sim * k).tolist(), k
