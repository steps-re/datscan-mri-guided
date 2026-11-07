"""Semi-automatic segmentation on the registered MRI (1-mm grid, CT frame).

Method (reproduces the exploratory analysis it was distilled from):

1. Lateral ventricles: smoothed T1 below a CSF threshold inside a box around
   the AC, morphological opening, connected component at a seed point.
2. Putamen, row by row (each axial row y at each slice z, each side):
   scan from 50 mm lateral of the midline toward the midline and remember
   the most medial 'CSF-ish' voxel 22-46 mm from the midline that is not
   ventricle (insular/sylvian CSF). Step 3 mm further medially (skipping the
   insular cortex, claustrum and capsules) and collect the run of voxels in
   the putamen grey range, stopping at white matter or 14 mm from midline,
   at most 13 mm wide. 3-D cleanup: opening, largest component, closing,
   slice-wise hole filling.
3. Caudate: from the lateral wall of the lateral ventricle, step laterally
   collecting up to 5 grey voxels per row; largest component.
4. Intracranial mask from the CT (soft-tissue HU, eroded, seed component,
   re-dilated, holes filled).
5. Region map for analysis and partial-volume correction: caudate,
   anterior/posterior putamen (split at the coronal plane of the AC),
   CSF, deep grey, white matter, cortical grey, extracranial, and the
   occipital-grey and cerebellar reference regions.

The putamen VOI is generous ("oversized"): on the reference case it is
about 7-8 mL per side, against an anatomical putamen of roughly 4-6 mL. The
analysis is NOT insensitive to that. Blur spreads less striatal activity into
an oversized posterior VOI, so the measured posterior/anterior ratio falls
as the VOI grows. The simulation therefore puts binding into a separate,
anatomically sized putamen (`anatomical_putamen`) and measures it through the
VOI exactly as applied to the patient, and the robustness step sweeps the VOI
size (-2 to +6 mm) and the segmentation thresholds to show how much it matters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as nd

LABELS = {1: "caudR", 2: "caudL", 3: "antputR", 4: "antputL", 5: "postputR", 6: "postputL",
          7: "CSF", 8: "deep", 9: "WM", 10: "GMctx", 11: "extracran", 13: "occGM", 14: "cerGM"}
GTM_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]


def _box(shape, ac, b):
    B = np.zeros(shape, bool)
    sl = tuple(slice(max(ac[i] + b[i][0], 0), max(ac[i] + b[i][1], 0)) for i in range(3))
    B[sl] = True
    return B


def _component_at(mask, seed):
    lab, n = nd.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    seed = tuple(int(v) for v in seed)
    l = lab[seed]
    if l == 0:
        idx = np.argwhere(lab > 0)
        l = lab[tuple(idx[((idx - np.array(seed)) ** 2).sum(1).argmin()])]
    return lab == l


def _largest(mask):
    L, n = nd.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    c = np.bincount(L.ravel())
    c[0] = 0
    return L == c.argmax()


def ventricles(mg, ac, P):
    V = P["ventricles"]
    m = nd.gaussian_filter(mg, V["smooth_sigma"])
    csf = (m < V["csf_hi"]) & (m > V["csf_lo"]) & _box(m.shape, ac, V["box"])
    vent = nd.binary_opening(csf, iterations=V["open_iter"])
    return _component_at(vent, np.add(ac, V["seed"]))


def putamen(mg, vent, ac, P):
    """Return uint8 map with 3 = right putamen, 4 = left putamen."""
    Q = P["putamen"]
    m = nd.gaussian_filter(mg, Q["smooth_sigma"])
    dvent = nd.distance_transform_edt(~vent)
    out = np.zeros(m.shape, np.uint8)
    MID = ac[2]
    fallback = Q.get("low_z") is not None          # case-specific left-only workaround
    low_z = ac[0] + Q["low_z"] if fallback else -10 ** 9
    for z in range(ac[0] + Q["z"][0], ac[0] + Q["z"][1] + 1):
        for y in range(ac[1] + Q["y"][0], ac[1] + Q["y"][1] + 1):
            prof = m[z, y]
            for code, step, start in ((3, +1, MID - Q["scan_start"]), (4, -1, MID + Q["scan_start"])):
                lastcsf = None
                for xx in range(start, MID - step * Q["min_mid"], step):
                    if abs(xx - MID) < Q["min_mid"]:
                        break
                    ml = Q["max_lat_low_L"] if (code == 4 and z <= low_z) else Q["max_lat"]
                    if (prof[xx] < Q["csf_thr"] and Q["csf_min_mid"] <= abs(xx - MID) <= ml
                            and dvent[z, y, xx] > Q["csf_dvent"]):
                        lastcsf = xx
                if fallback and lastcsf is None and code == 4 and z <= low_z:
                    lastcsf = MID + Q["max_lat_low_L"] - Q["lat_gap"] + 1
                if lastcsf is None:
                    continue
                xx = lastcsf + step * Q["lat_gap"]
                run = []
                while abs(xx - MID) >= Q["min_mid"] and len(run) < Q["max_w"]:
                    v = prof[xx]
                    if v >= Q["hi"]:
                        break
                    if v > Q["lo"] and dvent[z, y, xx] > Q["dvent"]:
                        run.append(xx)
                    elif run:
                        break
                    xx += step
                if len(run) >= Q["min_w"]:
                    out[z, y, run] = code
    for code in (3, 4):
        mk = out == code
        out[mk] = 0
        mk = _largest(nd.binary_opening(mk, iterations=Q["open_iter"]))
        mk = nd.binary_closing(mk, iterations=Q["close_iter"])
        for z in range(mk.shape[0]):
            mk[z] = nd.binary_fill_holes(mk[z])
        out[mk & (out == 0)] = code
    return out


def caudate(mg, vent, put, ac, P):
    """Add 1 = right caudate, 2 = left caudate to the putamen map."""
    C = P["caudate"]
    m = nd.gaussian_filter(mg, C["smooth_sigma"])
    out = put.copy()
    X = m.shape[2]
    for z in range(ac[0] + C["z"][0], ac[0] + C["z"][1] + 1):
        for y in range(ac[1] + C["y"][0], ac[1] + C["y"][1] + 1):
            xs = np.nonzero(vent[z, y])[0]
            if xs.size == 0:
                continue
            rs = xs[xs < ac[2] + C["wall_split_R"]]
            ls = xs[xs >= ac[2] + C["wall_split_L"]]
            for code, step, wall in ((1, -1, rs.min() if rs.size else None),
                                     (2, +1, ls.max() if ls.size else None)):
                if wall is None:
                    continue
                xx = wall + step
                run = []
                while len(run) < C["max_w"] and 0 <= xx < X:
                    v = m[z, y, xx]
                    if v >= C["hi"] or put[z, y, xx]:
                        break
                    if v > C["lo"]:
                        run.append(xx)
                    elif run:
                        break
                    xx += step
                if len(run) >= C["min_w"] and np.mean(m[z, y, run]) < C["mean_max"]:
                    out[z, y, run] = code
    for code in (1, 2):
        mk = out == code
        out[mk] = 0
        mk = nd.binary_closing(_largest(mk), iterations=C["close_iter"])
        out[mk & (out == 0)] = code
    Z = np.arange(out.shape[0])[:, None, None]
    Yy = np.arange(out.shape[1])[None, :, None]
    dy, dz = C["drop_post_inferior"]
    out[((out == 1) | (out == 2)) & (Yy > ac[1] + dy) & (Z < ac[0] + dz)] = 0
    return out


def intracranial(ct_grid, ac, P):
    I = P["intracranial"]
    cs = nd.median_filter(ct_grid, 3)
    soft = (cs > I["hu_lo"]) & (cs < I["hu_hi"])
    er = nd.binary_erosion(soft, iterations=I["erode"])
    ic = _component_at(er, np.add(ac, I["seed"]))
    ic = nd.binary_dilation(ic, iterations=I["dilate"]) & soft
    for z in range(ic.shape[0]):
        ic[z] = nd.binary_fill_holes(ic[z])
    return ic


def regions(mg, ct_grid, seg, ic, ac, P, split_offset: int = 0):
    """Full label map (codes in LABELS). The anterior/posterior putamen split
    is the coronal plane y = AC + split_offset."""
    G = P["regions"]
    m = nd.gaussian_filter(mg, G["smooth_sigma"])
    ic = ic.copy()
    ic[: max(ac[0] + G["z_min"], 0)] = False
    Z, Y, X = np.indices(m.shape)
    dz, dy, dx = Z - ac[0], Y - ac[1], X - ac[2]
    csf = ic & (m < G["csf_hi"])
    gm = ic & (m >= G["csf_hi"]) & (m < G["gm_hi"])
    wm = ic & (m >= G["gm_hi"])
    head = nd.binary_fill_holes(nd.binary_closing(ct_grid > G["head_hu"], iterations=3))
    split = split_offset
    lab = np.zeros(m.shape, np.uint8)
    lab[seg == 1] = 1
    lab[seg == 2] = 2
    lab[(seg == 3) & (dy < split)] = 3
    lab[(seg == 4) & (dy < split)] = 4
    lab[(seg == 3) & (dy >= split)] = 5
    lab[(seg == 4) & (dy >= split)] = 6
    free = lab == 0
    o, c = G["occ"], G["cer"]
    occ = gm & (dy >= o["y_min"]) & (dz >= o["z"][0]) & (dz <= o["z"][1]) & free
    cer = ic & ~csf & (dy >= c["y_min"]) & (dz <= c["z_max"]) & (np.abs(dx) >= c["mid_gap"]) & free
    lab[occ] = 13
    lab[cer] = 14
    free = lab == 0
    lab[free & csf] = 7
    d = G["deep"]
    deep = (free & ~csf & (dx >= d["x"][0]) & (dx <= d["x"][1]) & (dy >= d["y"][0]) & (dy <= d["y"][1])
            & (dz >= d["z"][0]) & (dz <= d["z"][1]))
    lab[deep] = 8
    free = lab == 0
    lab[free & wm] = 9
    lab[free & gm] = 10
    lab[(lab == 0) & head] = 11
    return lab


def shrink_to_volume(mask, target_vox: int):
    """Shrink a mask uniformly (by distance from its edge) to `target_vox`
    voxels; thin parts (such as a posterior tail) disappear first. Ties in the
    (discrete) edge distance are broken by distance from the mask's centroid.
    Masks already at or below the target are returned unchanged."""
    n = int(mask.sum())
    if n <= target_vox or target_vox <= 0:
        return mask.copy()
    d = nd.distance_transform_edt(mask)
    idx = np.argwhere(mask)
    c = idx.mean(0)
    score = d[mask] - 1e-3 * np.sqrt(((idx - c) ** 2).sum(1))
    keep = idx[np.argsort(-score, kind="stable")[:target_vox]]
    core = np.zeros_like(mask)
    core[tuple(keep.T)] = True
    return _largest(core)


def otsu_threshold(v, bins=128) -> float:
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s0 = np.cumsum(h * c)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def grey_core(mask, m_smooth, floor, wm_floor, wm_frac=0.95, min_core_ml=3.5):
    """The grey-matter part of a VOI: Otsu split of the (smoothed) T1 inside
    the VOI, keeping the darker class above the CSF `floor`, then opening,
    largest component and closing (inside the VOI).

    The split is used ONLY if the brighter class looks like the white matter
    right around this VOI: its mean must be at least `wm_frac` x the median of
    the voxels >= `wm_floor` in a 2-4 mm shell outside the VOI (the capsules).
    Otherwise the VOI holds no white matter to remove (Otsu would just cut the
    putamen in two) and it is returned unchanged. A core smaller than
    `min_core_ml` is not an anatomically plausible putamen (on the reference
    case the split gives 1.3 mL on the right: Otsu cut the putamen itself), so
    it is rejected too. Returns (core, info)."""
    v = m_smooth[mask]
    t = otsu_threshold(v)
    hi = v[v >= t]
    shell = nd.binary_dilation(mask, iterations=4) & ~nd.binary_dilation(mask, iterations=1)
    sv = m_smooth[shell & (m_smooth >= wm_floor)]
    wm = float(np.median(sv)) if sv.size else float("inf")
    info = {"otsu": t, "bright_class_mean": float(hi.mean()) if hi.size else None, "shell_wm_median": wm}
    if hi.size == 0 or v[v < t].size == 0 or hi.mean() < wm_frac * wm:
        info["split"] = False
        return mask.copy(), info
    k = mask & (m_smooth < t) & (m_smooth >= floor)
    k = nd.binary_closing(_largest(nd.binary_opening(k, iterations=1)), iterations=1) & mask
    info["core_ml"] = float(k.sum() / 1000)
    if k.sum() < min_core_ml * 1000:
        info["split"] = False
        info["rejected"] = f"core {k.sum() / 1000:.1f} mL < {min_core_ml} mL"
        return mask.copy(), info
    info["split"] = True
    return k, info


def anatomical_putamen(seg, m_smooth, floor, wm_floor, cap_ml=None, info=None):
    """Best-estimate anatomical putamen FOR THE SIMULATION ONLY.

    The measurement VOI (codes 3 R / 4 L in `seg`) can be larger than the
    putamen: the row scan collects up to 13 mm of voxels in a wide intensity
    range over a fixed box, so it can take in capsule white matter and run
    past the putamen's ends. Filling all of it with striatal binding in the
    simulation makes the expected posterior/anterior ratio RISE as the VOI
    grows, while a real image (binding only in the real putamen) gives a
    FALLING ratio. So the simulation uses, per side:
      1. the grey core of the VOI (`grey_core`; removes white matter only if
         the VOI contains voxels as bright as the white matter around it), then
      2. if `cap_ml` is set, that core shrunk by distance from its edge (thin
         parts such as a posterior tail go first) to at most `cap_ml` mL.
    The cap is an ASSUMPTION about putamen size (adult putamen is of the order
    of 4-6 mL per side and shrinks with age); it is swept. On the randomised
    phantom step 1 alone matches the true putamen with Dice about 0.9 (the
    VOI: about 0.8)."""
    out = seg.copy()
    for code in (3, 4):
        mk = seg == code
        out[mk] = 0
        core, gi = grey_core(mk, m_smooth, floor, wm_floor)
        if info is not None:
            info[{3: "R", 4: "L"}[code]] = gi
        if cap_ml is not None:
            core = shrink_to_volume(core, int(round(cap_ml * 1000)))
        out[core] = code
    return out


def voi_variants(lab, mms, which: str = "whole") -> dict:
    """Measurement-VOI variants: grow (mm > 0) or shrink (mm < 0) the putamen
    VOIs by `mm` (Euclidean), either the whole putamen ("whole": codes 3-6)
    or the posterior VOI only ("posterior": codes 5, 6). Grown voxels come
    only from non-striatal tissue and keep the anterior/posterior label of the
    nearest VOI voxel. Removed voxels become white matter (code 9); only VOI
    membership matters for the uncorrected ratio. Returns {mm: label map}."""
    sets = ((3, 5), (4, 6)) if which == "whole" else ((5,), (6,))
    striatal = np.isin(lab, [1, 2, 3, 4, 5, 6])
    pre = []
    for codes in sets:
        mk = np.isin(lab, codes)
        dout, ind = nd.distance_transform_edt(~mk, return_indices=True)
        pre.append((mk, dout, lab[tuple(ind)], nd.distance_transform_edt(mk)))
    out = {}
    for mm in mms:
        L = lab.copy()
        for mk, dout, near, din in pre:
            if mm > 0:
                grow = (dout <= mm) & ~striatal
                L[grow] = near[grow]
            elif mm < 0:
                L[mk & (din <= -mm)] = 9
        out[mm] = L
    return out


def volumes_ml(lab):
    return {n: float((lab == k).sum() / 1000.0) for k, n in LABELS.items()}
