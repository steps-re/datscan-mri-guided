"""Perturbation analysis: how much do the ratios move if the SPECT-MRI
alignment, the anterior/posterior split or the VOI size are slightly off?

* 12 rigid perturbations of the SPECT relative to the MRI-defined VOIs:
  +/-2 mm along each axis, +/-2 degrees about each axis (centre = AC).
* Anterior/posterior split moved +/-4 mm.
* VOIs eroded by 1 mm (eroded voxels reassigned to white matter).
* Combined SPECT translations: every integer shift in {-4, -2, 0, 2, 4} mm on
  all three axes at once (125 shifts, measured ratio only).
Uncorrected and GTM-corrected (11 mm) values are reported. These ranges are
of the MEASURED ratio only; the VOI-size sweep against both simulated nulls
is in `pipeline.analyze` (results["sweep"]).
"""
from __future__ import annotations

import itertools

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

from .alignment import shifted_crop
from .profile import metrics, voi_means
from .pvc import gtm, transfer_matrix
from .segment import GTM_IDS


def rigid_set(shift, rot_deg):
    r = np.radians(rot_deg)
    out = []
    for ax, n in enumerate("xyz"):
        for sgn in (+1, -1):
            p = [0.0] * 6
            p[3 + ax] = sgn * shift
            out.append((f"{n}{'+' if sgn > 0 else '-'}{shift:g}mm", p))
    for ax, n in enumerate("xyz"):
        for sgn in (+1, -1):
            p = [0.0] * 6
            p[ax] = sgn * r
            out.append((f"r{n}{'+' if sgn > 0 else '-'}{rot_deg:g}deg", p))
    return out


def run(spect_img, grid_ref, crop, lab, valid, label_variants: dict, center_lps, P):
    """label_variants: {name: cropped label map} (split shifts etc.)."""
    R = P["robustness"]
    fw = R["gtm_fwhm"]
    ids = GTM_IDS
    s = sitk.GetArrayFromImage(sitk.Resample(spect_img, grid_ref, sitk.Transform(), sitk.sitkLinear, -1)).astype(np.float32)[crop]

    def both(img, L, W=None):
        o = voi_means(img, L, valid, ids)
        W = W if W is not None else transfer_matrix(L, ids, fw, valid)
        return {"uncorrected": metrics(dict(zip(ids, o)), L, valid),
                f"gtm{fw:g}": metrics(dict(zip(ids, gtm(o, W))), L, valid)}

    out = {"variants": {}, "rigid": {}}
    for name, L in label_variants.items():
        out["variants"][name] = both(s, L)
    le = lab.copy()
    for i in (1, 2, 3, 4, 5, 6):
        mk = le == i
        le[mk & ~nd.binary_erosion(mk, iterations=R["erode_mm"])] = 9
    out["variants"][f"erode{R['erode_mm']}mm"] = both(s, le)
    W = transfer_matrix(lab, ids, fw, valid)
    for name, par in rigid_set(R["shift_mm"], R["rot_deg"]):
        t = sitk.Euler3DTransform()
        t.SetCenter(center_lps)
        t.SetParameters(par)
        a = sitk.GetArrayFromImage(sitk.Resample(spect_img, grid_ref, t, sitk.sitkLinear, -1)).astype(np.float32)[crop]
        out["rigid"][name] = both(a, lab, W)
    return out


def ranges(rob, key="uncorrected", ref="occ"):
    res = {}
    for side in "RL":
        rig = [v[key][ref][side]["post_ant"] for v in rob["rigid"].values()]
        allv = rig + [v[key][ref][side]["post_ant"] for v in rob["variants"].values()]
        res[side] = {"rigid_only": [min(rig), max(rig)], "all": [min(allv), max(allv)]}
    return res


def combined_shift_grid(sg_full, fov_full, crop, lab, measure, mm: int, step: int) -> dict:
    """Measured P/A over every combined integer translation in
    {-mm..mm step `step`}^3 (all three axes at once)."""
    vals = list(range(-mm, mm + 1, step))
    rows = []
    for d in itertools.product(vals, repeat=3):
        pa = measure(shifted_crop(sg_full, crop, d, 0), lab, shifted_crop(fov_full, crop, d, False))
        rows.append({"shift_mm_xyz": [d[2], d[1], d[0]], "R": pa[0], "L": pa[1]})
    return {"note": f"combined SPECT translation, all axes {-mm}..{mm} mm step {step} ({len(rows)} shifts), "
                    "measured P/A (uncorrected, occipital reference)",
            "range": {sd: [min(r[sd] for r in rows), max(r[sd] for r in rows)] for sd in "RL"},
            "grid": rows}
