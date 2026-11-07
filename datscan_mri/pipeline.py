"""End-to-end orchestration: anatomy (registration + segmentation) then
SPECT analysis (profile, simulated nulls, sweeps, GTM, robustness)."""
from __future__ import annotations

import copy
import itertools
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

from . import profile as prof
from . import external, pvc, register, robustness, segment, simulate
from .io import sitk_to_ras_affine, write_clean_nifti
from .segment import GTM_IDS


class QCError(RuntimeError):
    """A quality gate failed; the run must stop (non-zero exit)."""


@dataclass
class Anatomy:
    grid: sitk.Image
    mg: np.ndarray          # N4-corrected MRI on the grid
    ct: np.ndarray          # CT (HU) on the grid
    vent: np.ndarray
    seg: np.ndarray         # 1-4 caudate/putamen (measurement VOIs, before the A/P split)
    ic: np.ndarray
    lab: np.ndarray         # full region map (measurement)
    ac: tuple               # (z, y, x) grid index
    ac_source: str
    transform: sitk.Transform
    reg_info: dict = field(default_factory=dict)
    intensity_scale: float = 1.0
    seg_info: dict = field(default_factory=dict)      # segmenter name/version/model/QC, Dice vs alternatives
    alt_segs: dict = field(default_factory=dict)      # {name: seg (codes 1-4)} other segmentations, for the sweep
    fs_labels: np.ndarray | None = None               # external segmenter's FreeSurfer labels on the grid


def ac_index(grid, P, ct_grid):
    if P.get("ac_lps_mm") is not None:
        x, y, z = grid.TransformPhysicalPointToIndex(tuple(float(v) for v in P["ac_lps_mm"]))
        return (z, y, x), "given"
    # rough estimate: centroid of the largest soft-tissue (brain) component
    I = P["intracranial"]
    cs = nd.median_filter(ct_grid, 3)
    soft = (cs > I["hu_lo"]) & (cs < I["hu_hi"])
    L, _ = nd.label(nd.binary_erosion(soft, iterations=I["erode"]))
    c = np.bincount(L.ravel())
    c[0] = 0
    com = np.array(nd.center_of_mass(L == c.argmax()))
    ac = tuple(int(round(v)) for v in com + np.array(P["ac_from_ic_centroid"]))
    return ac, "ESTIMATED from intracranial centroid - verify on the figures and pass --ac"


def _threshold_seg(mg, vent, ac, P):
    put = segment.putamen(mg, vent, ac, P)
    return segment.caudate(mg, vent, put, ac, P)


def striatal_segmentation(mg, grid, vent, ac, P, workdir=None, log=print):
    """Putamen/caudate (codes 1-4) from the configured segmenter, plus the
    alternatives used for Dice and the segmentation sweep. Quality gates for
    an external segmenter: its QC scores, plausible putamen volume, sides."""
    S = P["segmenter"]
    method = S["method"]
    alts, fs = {}, None
    if method == "threshold":
        seg = _threshold_seg(mg, vent, ac, P)
        info = {"method": "threshold", "tool": "datscan-mri semi-automatic row scan (intensity thresholds)"}
    else:
        wd = workdir or tempfile.mkdtemp(prefix="datscan_seg_")
        try:
            seg, fs, info = external.segment_striatum(mg, grid, ac, S, wd, log)
        except external.SegmenterError as e:
            raise QCError(f"segmenter {method!r}: {e}") from e
        q = info.get("qc_scores") or {}
        low = {k: v for k, v in q.items() if v < S["min_qc_score"]}
        if low:
            raise QCError(f"{info['tool']} QC scores below {S['min_qc_score']}: {low} (segmentation likely "
                          "failed). Check fig2, or lower --min-segmenter-qc.")
        lo, hi = P["qc"]["external_putamen_ml"]
        for code, side in ((3, "R"), (4, "L")):
            ml = float((seg == code).sum() / 1000)
            if not lo <= ml <= hi:
                raise QCError(f"{info['tool']} {side} putamen is {ml:.2f} mL, outside the plausible "
                              f"{lo}-{hi} mL (qc.external_putamen_ml). Check fig2.")
        if S.get("compare_threshold", True):
            alts["threshold"] = _threshold_seg(mg, vent, ac, P)
    for name, path in (S.get("compare_labels_files") or {}).items():
        try:
            a = external.striatum_from_fs(external.resample_to_grid(path, grid))
            external.check_sides(a, ac[2])
        except (external.SegmenterError, RuntimeError, OSError) as e:
            raise QCError(f"comparison labels {name!r}: {e}") from e
        alts[name] = a
    info["putamen_ml"] = {sd: float((seg == c).sum() / 1000) for c, sd in ((3, "R"), (4, "L"))}
    info["caudate_ml"] = {sd: float((seg == c).sum() / 1000) for c, sd in ((1, "R"), (2, "L"))}
    info["dice_vs"] = {n: external.striatal_dice(seg, a) for n, a in alts.items()}
    info["alternatives_ml"] = {n: {"putamen_R": float((a == 3).sum() / 1000), "putamen_L": float((a == 4).sum() / 1000),
                                   "caudate_R": float((a == 1).sum() / 1000), "caudate_L": float((a == 2).sum() / 1000)}
                               for n, a in alts.items()}
    if len(alts) >= 2:
        ks = list(alts)
        info["dice_between_alternatives"] = {f"{a} vs {b}": external.striatal_dice(alts[a], alts[b])
                                             for i, a in enumerate(ks) for b in ks[i + 1:]}
    return seg, alts, fs, info


def prepare_anatomy(ct: sitk.Image, mri: sitk.Image, P: dict, transform: sitk.Transform | None = None,
                    log=print, workdir=None) -> Anatomy:
    grid = register.make_grid(ct)
    c = register.to_grid(ct, grid, None, sitk.sitkLinear, -1000)
    ac, src = ac_index(grid, P, c)
    if src != "given" and not P["qc"]["allow_estimated_ac"]:
        raise QCError("No AC given (--ac / ac_lps_mm). The AC estimate from the intracranial centroid is "
                      "rough; pass --ac x,y,z, or --allow-estimated-ac to run anyway and check the figures.")
    if transform is None:
        log("registering MRI to CT (rigid, mutual information)...")
        transform, info = register.register_mri_to_ct(ct, mri, P)
    else:
        info = {"note": "transform supplied by user", "parameters": list(transform.GetParameters())}
    log("N4 bias correction and resampling to 1-mm grid...")
    cor = register.n4_correct(mri, P)
    mg = register.to_grid(cor, grid, transform, sitk.sitkBSpline, 0)
    log(f"AC grid index (z,y,x) = {ac} [{src}]")
    ic = segment.intracranial(c, ac, P)
    scale = 1.0
    if P["intensity"]["normalize_p90"]:
        er = nd.binary_erosion(ic, iterations=10)
        p90 = float(np.percentile(nd.gaussian_filter(mg, 1.0)[er], 90))
        scale = P["intensity"]["reference_p90"] / p90
        mg = (mg * scale).astype(np.float32)
    log(f"segmenting ventricles, putamen, caudate (segmenter: {P['segmenter']['method']})...")
    vent = segment.ventricles(mg, ac, P)
    seg, alts, fs, seg_info = striatal_segmentation(mg, grid, vent, ac, P, workdir, log)
    lab = segment.regions(mg, c, seg, ic, ac, P)
    dice = register.ventricle_dice(c, vent, np.add(ac, P["ventricles"]["seed"]))
    info["ventricle_dice_ct_vs_mri"] = dice
    if not np.isfinite(dice) or dice < P["qc"]["min_ventricle_dice"]:
        raise QCError(f"CT-vs-MRI ventricle Dice {dice:.2f} is below {P['qc']['min_ventricle_dice']} "
                      "(registration or ventricle segmentation failed). Check fig1, or lower "
                      "--min-ventricle-dice if the CT ventricles are genuinely hard to see.")
    for code, side in ((3, "R"), (4, "L")):
        ml = float((seg == code).sum() / 1000)
        info[f"putamen_voi_ml_{side}"] = ml
        if P["segmenter"]["method"] == "threshold" and ml < P["qc"]["min_putamen_voi_ml"]:
            raise QCError(f"{side} putamen VOI is {ml:.1f} mL (< {P['qc']['min_putamen_voi_ml']} mL): the "
                          "segmentation has probably collapsed (on the phantom this happens when the putamen "
                          "falls below the 'putamen.lo' threshold). Check fig2 and adjust the putamen "
                          "parameters, or lower --min-putamen-ml.")
    return Anatomy(grid, mg, c, vent, seg, ic, lab, ac, src, transform, info, scale, seg_info, alts, fs)


def _crop(lab, margin):
    nz = np.nonzero(lab > 0)
    return tuple(slice(max(a.min() - margin, 0), a.max() + margin + 1) for a in nz)


def _pad(arr_crop, SL, shape):
    out = np.zeros(shape, np.float32)
    out[SL] = arr_crop
    return out


def anatomical_labels(anat: Anatomy, P: dict, cap_ml=None, seg=None, info=None) -> np.ndarray:
    """Region map used ONLY to place activity in the simulation: the measurement
    VOI's putamen replaced by the anatomical putamen (`segment.anatomical_putamen`);
    voxels it gives up are relabelled as CSF / deep grey / WM / cortex by intensity."""
    seg = anat.seg if seg is None else seg
    m = nd.gaussian_filter(anat.mg, 1.0)
    ap = segment.anatomical_putamen(seg, m, P["regions"]["csf_hi"], P["regions"]["gm_hi"], cap_ml, info)
    return segment.regions(anat.mg, anat.ct, ap, anat.ic, anat.ac, P)


def _vkey(v):
    return "no cap" if v is None else f"cap {v:g} mL"


def _pa(values, lab=None, valid=None):
    """Posterior/anterior SBR ratio per side, occipital reference, from VOI means."""
    ref = values[prof.REFS["occ"]]
    out = []
    for side in "RL":
        a_c, p_c, _ = prof.SIDES[side]
        out.append(float((values[p_c] - ref) / (values[a_c] - ref)))
    return tuple(out)


def null_rows(bases: dict, P: dict, name: str, labmap, valid, true_pas, vols=None, fwhms=None,
              backgrounds=None, sbr_levels=None):
    """Expected P/A over a grid of assumptions. true_pas: [None] for uniform binding."""
    S = P["simulation"]
    vols = vols or list(bases)
    fwhms = fwhms or S["fwhm_expected"]
    bgs = backgrounds or {"default": S["background"], **S.get("background_alternatives", {})}
    sbrs = sbr_levels or S["sbr_levels"]
    rows = []
    for vol, fw, (bgn, bg), (sbr, csbr), t in itertools.product(vols, fwhms, bgs.items(), sbrs, true_pas):
        v = bases[vol].voi_values(fw, name, bg, sbr, csbr, t)
        r, l = _pa(v, labmap, valid)
        rows.append({"anatomical_putamen": _vkey(vol), "fwhm_mm": fw, "background": bgn, "true_putamen_sbr": sbr,
                     "true_post_ant": 1.0 if t is None else t, "R": r, "L": l})
    return rows


def _range(rows):
    return {sd: [min(r[sd] for r in rows), max(r[sd] for r in rows)] for sd in "RL"}


def _measured(s, labmap, valid):
    return _pa(dict(zip(GTM_IDS, prof.voi_means(s, labmap, valid, GTM_IDS))), labmap, valid)


def analyze(anat: Anatomy, spect: sitk.Image, P: dict, do_robustness=True, do_sweeps=True, log=print,
            cache: dict | None = None) -> dict:
    """`cache` (a dict) may be passed to reuse the anatomy-only simulation bases
    between SPECTs of the same anatomy (used by the phantom tests)."""
    ac = anat.ac
    S = P["simulation"]
    sg = register.to_grid(spect, anat.grid, None, sitk.sitkLinear, 0)
    fov = register.spect_fov_mask(spect, anat.grid)
    SL = _crop(anat.lab, P["analysis"]["crop_margin"])
    lab = anat.lab[SL]
    s = sg[SL]
    valid = fov[SL]
    vois = np.isin(lab, [1, 2, 3, 4, 5, 6, 13, 14])
    frac_out = {segment.LABELS[c]: float(((lab == c) & ~valid).sum() / max((lab == c).sum(), 1))
                for c in (1, 2, 3, 4, 5, 6, 13, 14)}
    gated = {k: v for k, v in frac_out.items() if k != "cerGM"}   # striatum + primary (occipital) reference
    worst = max(gated, key=gated.get)
    if gated[worst] > P["qc"]["max_voi_outside_fov_frac"]:
        raise QCError(f"{gated[worst]:.1%} of the {worst} VOI lies outside the SPECT reconstruction support "
                      f"(limit {P['qc']['max_voi_outside_fov_frac']:.1%}).")
    out_frac = float((vois & ~valid).sum() / vois.sum())
    Yi = np.indices(lab.shape)[1] + SL[1].start
    dy = (Yi - ac[1]).astype(np.float32)
    ids = GTM_IDS
    obs = prof.voi_means(s, lab, valid, ids)
    res = {"ac_index_zyx": list(map(int, ac)), "ac_source": anat.ac_source,
           "segmentation": anat.seg_info,
           "registration": anat.reg_info, "intensity_scale": anat.intensity_scale,
           "volumes_ml": segment.volumes_ml(anat.lab),
           "voxels_outside_spect_fov_in_vois": int((vois & ~valid).sum()),
           "voi_fraction_outside_spect_fov": out_frac,
           "voi_fraction_outside_spect_fov_by_region": frac_out,
           "uncorrected": prof.metrics(dict(zip(ids, obs)), lab, valid)}
    meas = {sd: res["uncorrected"]["occ"][sd]["post_ant"] for sd in "RL"}

    # ---- measurement-VOI variants (the simulated binding is NOT changed by these)
    mkey = ("measure", tuple(P["sweep"]["voi_mm"]) if do_sweeps else (), do_robustness, lab.shape, int(lab.sum()))
    if cache is not None and mkey in cache:
        measure, sweep_names = cache[mkey]
    else:
        measure = {"base": lab}
        sweep_names = []
        if do_sweeps:
            for which in ("whole", "posterior"):
                for mm, L in segment.voi_variants(lab, P["sweep"]["voi_mm"], which).items():
                    measure[f"{which}{mm:+g}mm"] = L
                    sweep_names.append(f"{which}{mm:+g}mm")
        if do_robustness:
            sh = P["robustness"]["split_shift_mm"]
            for d in (-sh, +sh):
                measure[f"split{d:+d}mm"] = segment.regions(anat.mg, anat.ct, anat.seg, anat.ic, ac, P,
                                                             split_offset=d)[SL]
        if cache is not None:
            cache[mkey] = (measure, sweep_names)

    # ---- simulation bases (anatomy-only; cached across SPECTs of the same anatomy)
    nominal = S["anatomical_putamen_cap_ml"]
    vols = [nominal] + [v for v in S["anatomical_putamen_cap_ml_sweep"] if v != nominal]
    fwhms = sorted(set(S["fwhm_expected"]) | set(P["sweep"]["fwhm"]))
    key = ("bases", tuple(map(str, vols)), tuple(fwhms), tuple(sorted(measure)), int((valid & vois).sum()))
    if cache is not None and key in cache:
        bases, act_labs, core_info = cache[key]
    else:
        log(f"simulation bases: anatomical putamen {[_vkey(v) for v in vols]} x FWHM {fwhms} mm...")
        bases, act_labs, core_info = {}, {}, {}
        for vol in vols:
            al = anatomical_labels(anat, P, vol, info=core_info)[SL]
            act_labs[vol] = al
            bases[vol] = simulate.Basis(al, dy, fwhms, measure, valid, ids)
        if cache is not None:
            cache[key] = (bases, act_labs, core_info)
    t0 = S["normal_gradient_pa"]
    trange = S["normal_gradient_pa_range"]

    uni_rows = null_rows(bases, P, "base", lab, valid, [None])
    ng_rows = null_rows(bases, P, "base", lab, valid, trange)
    exp_uni, exp_ng = _range(uni_rows), _range(ng_rows)
    exp_ng_nom = _range(null_rows(bases, P, "base", lab, valid, [t0], vols=[nominal],
                                  backgrounds={"default": S["background"]}))
    exp_uni_nom = _range(null_rows(bases, P, "base", lab, valid, [None], vols=[nominal],
                                   backgrounds={"default": S["background"]}))
    margin = P["decision"]["margin"]
    res["simulation"] = {
        "note": ("binding placed in an anatomical putamen (VOI shrunk to the given volume), measured through "
                 "the patient's VOI; isotropic Gaussian PSF; no attenuation, scatter, collimator or "
                 "reconstruction modelling"),
        "grey_core": core_info,
        "anatomical_putamen_ml_by_side": {
            _vkey(v): {sd: float(np.isin(act_labs[v], simulate.PUT[sd]).sum() / 1000) for sd in "RL"} for v in vols},
        "normal_gradient_slopes_per_mm": {str(t): bases[nominal].slopes(t) for t in trange},
        "uniform": {"expected_post_ant_range": exp_uni, "nominal_range_over_fwhm": exp_uni_nom,
                    "grid": uni_rows},
        "normal_gradient": {"true_post_ant": t0, "true_post_ant_range": [min(trange), max(trange)],
                            "expected_post_ant_range": exp_ng, "nominal_range_over_fwhm": exp_ng_nom,
                            "grid": ng_rows},
    }
    res["decision"] = {
        "rule": (f"posterior loss called if measured P/A < lowest normal-gradient expectation - {margin}"),
        "margin": margin,
        "gap_vs_normal_gradient_low": {sd: meas[sd] - exp_ng[sd][0] for sd in "RL"},
        "gap_vs_uniform_low": {sd: meas[sd] - exp_uni[sd][0] for sd in "RL"},
        "posterior_loss_called": {sd: bool(meas[sd] < exp_ng[sd][0] - margin) for sd in "RL"},
    }

    # ---- VOI sweep: measured vs both nulls with the simulated binding held fixed
    def sweep_row(name, labmap, bases_, s_img=s):
        """Same assumption grid as the decision (all anatomical sizes in `bases_`,
        backgrounds, binding levels, sweep FWHMs, true P/A 0.80-0.88)."""
        m = _measured(s_img, labmap, valid)
        u = _range(null_rows(bases_, P, name, labmap, valid, [None], fwhms=P["sweep"]["fwhm"]))
        g = _range(null_rows(bases_, P, name, labmap, valid, trange, fwhms=P["sweep"]["fwhm"]))
        return {"variant": name, "measured": dict(zip("RL", m)), "expected_uniform": u,
                "expected_normal_gradient": g,
                "gap_vs_normal_gradient_low": {sd: m[i] - g[sd][0] for i, sd in enumerate("RL")},
                "putamen_voi_ml": {sd: float(np.isin(labmap, simulate.PUT[sd]).sum() / 1000) for sd in "RL"}}

    if do_sweeps:
        rows = [sweep_row("base", lab, bases)] + [sweep_row(n, measure[n], bases) for n in sweep_names]
        if do_robustness:
            rows += [sweep_row(n, measure[n], bases) for n in measure if n.startswith("split")]
        thr_rows = []
        def rebuild_row(name, seg_t, Pt):
            lab_t = segment.regions(anat.mg, anat.ct, seg_t, anat.ic, ac, Pt)[SL]
            b = {v: simulate.Basis(anatomical_labels(anat, Pt, v, seg=seg_t)[SL], dy, P["sweep"]["fwhm"],
                                   {"base": lab_t}, valid, ids) for v in vols}
            r = sweep_row("base", lab_t, b)
            r["variant"] = name
            return r
        seg_rows = []
        for n, seg_t in anat.alt_segs.items():
            log(f"segmentation sweep: VOIs from {n}...")
            seg_rows.append(rebuild_row(f"segmenter {n}", seg_t, P))
        thr = P["sweep"]["threshold"] if P["segmenter"]["method"] == "threshold" else {}
        for k, offs in thr.items():
            for o in offs:
                log(f"threshold sweep: putamen {k} {o:+g}...")
                Pt = copy.deepcopy(P)
                Pt["putamen"][k] = P["putamen"][k] + o
                thr_rows.append(rebuild_row(f"putamen {k} {o:+g}", _threshold_seg(anat.mg, anat.vent, ac, Pt), Pt))
        allg = rows + thr_rows + seg_rows
        res["sweep"] = {
            "note": ("measurement VOI grown/shrunk (whole putamen or posterior VOI only) with the simulated "
                     "binding held in the same anatomical putamen; threshold variants rebuild both. Expected "
                     "ranges use the same assumption grid as the decision"),
            "voi": rows, "threshold": thr_rows, "segmenter": seg_rows,
            "gap_vs_normal_gradient_low_range": {
                sd: [min(r["gap_vs_normal_gradient_low"][sd] for r in allg),
                     max(r["gap_vs_normal_gradient_low"][sd] for r in allg)] for sd in "RL"},
            "measured_range": {sd: [min(r["measured"][sd] for r in allg), max(r["measured"][sd] for r in allg)]
                               for sd in "RL"},
            "expected_normal_gradient_range": {
                sd: [min(r["expected_normal_gradient"][sd][0] for r in allg),
                     max(r["expected_normal_gradient"][sd][1] for r in allg)] for sd in "RL"},
            "posterior_loss_called_in_all_variants": {
                sd: bool(all(r["gap_vs_normal_gradient_low"][sd] < -margin for r in allg)) for sd in "RL"},
        }

    # ---- profile of the patient and of both nulls at the profile FWHM (nominal assumptions)
    al = act_labs[nominal]
    sim_u = simulate.simulate(al, dy, S["fwhm_profile"], S["putamen_sbr"], S["caudate_sbr"], S["background"])
    sim_g = simulate.simulate(al, dy, S["fwhm_profile"], S["putamen_sbr"], S["caudate_sbr"], S["background"],
                              bases[nominal].slopes(t0))
    kw = dict(slab=P["profile"]["slab_mm"], min_voxels=P["profile"]["min_voxels"])
    p = prof.slab_profile(s, lab, valid, Yi, ac[1], **kw)
    for tag, img in (("sim", sim_u), ("simng", sim_g)):
        ps = prof.slab_profile(img, lab, valid, Yi, ac[1], **kw)
        p[f"{tag}_R"], p[f"{tag}_L"] = ps["R"], ps["L"]
        for side in "RL":   # simulation scaled to the patient's peak slab, for display
            pr, sm = np.array(p[side]), np.array(ps[side])
            k = int(np.nanargmax(pr))
            p[f"{tag}_{side}_scaled"] = (sm * pr[k] / sm[k]).tolist()
    p["sim_fwhm_mm"] = S["fwhm_profile"]
    p["simng_true_post_ant"] = t0
    res["profile"] = p

    # ---- posterior tail trimmed
    res["trimmed"] = []
    for t in P["trim_mm"]:
        res["trimmed"].append({"trim_mm": t,
                               "measured": prof.trimmed_post_ant(s, lab, valid, Yi, t),
                               "expected_uniform": prof.trimmed_post_ant(sim_u, lab, valid, Yi, t),
                               "expected_normal_gradient": prof.trimmed_post_ant(sim_g, lab, valid, Yi, t)})

    # ---- GTM partial-volume correction (assumes uniform activity within each VOI)
    res["gtm"] = {}
    for fw in P["gtm"]["fwhm"]:
        log(f"GTM at {fw} mm...")
        T = pvc.gtm(obs, pvc.transfer_matrix(lab, ids, fw, valid))
        res["gtm"][f"{fw:g}"] = prof.metrics(dict(zip(ids, T)), lab, valid)

    if do_robustness:
        log("robustness: 12 rigid perturbations, split +/-4 mm, 1-mm erosion...")
        R = P["robustness"]
        variants = {"base": lab}
        for n in measure:
            if n.startswith("split"):
                variants[n] = measure[n]
        center = anat.grid.TransformIndexToPhysicalPoint((int(ac[2]), int(ac[1]), int(ac[0])))
        rob = robustness.run(spect, anat.grid, SL, lab, valid, variants, center, P)
        rob["post_ant_ranges_uncorrected_occ"] = robustness.ranges(rob, "uncorrected")
        rob["post_ant_ranges_gtm_occ"] = robustness.ranges(rob, f"gtm{R['gtm_fwhm']:g}")
        res["robustness"] = rob
    res["_sg"] = sg  # stripped before JSON
    res["_sim"] = {"uniform": _pad(sim_u, SL, sg.shape), "normal_gradient": _pad(sim_g, SL, sg.shape)}
    res["_fov"] = fov
    return res


def measured_variants(anat: Anatomy, spect: sitk.Image, P: dict, cache: dict) -> dict:
    """Measured P/A for every cached measurement-VOI variant (no simulation):
    cheap, for repeated noise realisations after one `analyze` with `cache`."""
    SL = _crop(anat.lab, P["analysis"]["crop_margin"])
    s = register.to_grid(spect, anat.grid, None, sitk.sitkLinear, 0)[SL]
    valid = register.spect_fov_mask(spect, anat.grid)[SL]
    (measure, _), = [v for k, v in cache.items() if k[0] == "measure"]
    return {n: dict(zip("RL", _measured(s, L, valid))) for n, L in measure.items()}


def quick_metrics(anat: Anatomy, spect: sitk.Image, P: dict, labmap=None) -> dict:
    """Uncorrected VOI SBRs only (no simulation, no GTM): cheap, for repeated noise realisations.
    `labmap` (full grid) replaces the measurement VOIs if given."""
    SL = _crop(anat.lab, P["analysis"]["crop_margin"])
    lab = (anat.lab if labmap is None else labmap)[SL]
    s = register.to_grid(spect, anat.grid, None, sitk.sitkLinear, 0)[SL]
    valid = register.spect_fov_mask(spect, anat.grid)[SL]
    return prof.metrics(dict(zip(GTM_IDS, prof.voi_means(s, lab, valid, GTM_IDS))), lab, valid)


def save_derived(anat: Anatomy, sg: np.ndarray, out: Path, sims: dict | None = None):
    d = Path(out) / "derived"
    d.mkdir(parents=True, exist_ok=True)
    A = sitk_to_ras_affine(anat.grid)
    write_clean_nifti(anat.mg.astype(np.float32), A, d / "mri_grid.nii.gz")
    write_clean_nifti(anat.ct.astype(np.float32), A, d / "ct_grid.nii.gz")
    write_clean_nifti(sg.astype(np.float32), A, d / "spect_grid.nii.gz")
    write_clean_nifti(anat.lab.astype(np.uint8), A, d / "labels_grid.nii.gz")
    write_clean_nifti(anat.ic.astype(np.uint8), A, d / "intracranial_grid.nii.gz")
    write_clean_nifti(anat.vent.astype(np.uint8), A, d / "ventricles_grid.nii.gz")
    for k, v in (sims or {}).items():
        write_clean_nifti(v.astype(np.float32), A, d / f"sim_{k}_grid.nii.gz")
    if anat.fs_labels is not None:   # the external segmenter's full label map, reusable via --seg-labels
        write_clean_nifti(anat.fs_labels.astype(np.int16), A, d / "segmenter_fs_labels_grid.nii.gz")
    for n, a in anat.alt_segs.items():
        write_clean_nifti(a.astype(np.uint8), A, d / f"striatum_{n.replace(' ', '_')}_grid.nii.gz")
    sitk.WriteTransform(anat.transform, str(d / "mri_to_ct.tfm"))
    json.dump({"ac_index_zyx": list(map(int, anat.ac)), "ac_source": anat.ac_source},
              open(d / "anatomy.json", "w"), indent=1)
