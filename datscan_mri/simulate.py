"""Simulated null images on the subject's own anatomy.

Two nulls, both with NO disease-related posterior loss:

* uniform binding: the putamen has the same specific binding ratio (SBR)
  front to back. This is a LENIENT upper bound: real normal putamen is not
  uniform.
* normal gradient (primary comparator): putamen SBR falls linearly from
  front to back so that the true posterior/anterior SBR ratio, split at the
  AC plane, is 0.84 (range 0.80-0.88). 0.84 is the posterior/anterior ratio
  of the DaTQUANT normal values for age 65 (Grisanti 2023, PMC10338803,
  Table 1). That is a different pipeline, template and age, so only the order
  of magnitude transfers.

The binding is placed in an ANATOMICAL putamen (`segment.anatomical_putamen`),
not in the measurement VOI. The image is blurred with an isotropic Gaussian
and then measured through the VOI exactly as applied to the patient. A
generous VOI therefore lowers the expected ratio just as it lowers the
measured one, instead of raising it.

Not modelled: attenuation, scatter, collimator depth-dependent resolution,
reconstruction (OSEM) non-linearity, noise. Background activities (relative
to occipital grey = 1) are assumptions; alternatives are swept.

Blurring is linear, so every null is a weighted sum of a few blurred
"component" images (one per region, plus the putamen's linear gradient term).
`Basis` blurs each component once per FWHM and stores only its mean in each
measurement VOI, which makes the full grid of nulls cheap.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as nd

from .segment import LABELS

FWHM_TO_SIGMA = 1.0 / 2.3548
BG_CODES = (7, 8, 9, 10, 11, 13, 14)
# component order: background regions, caudate R/L, putamen R/L, putamen R/L x dy
COMPONENTS = [LABELS[c] for c in BG_CODES] + ["caudR", "caudL", "putR", "putL", "putR_dy", "putL_dy"]
PUT = {"R": (3, 5), "L": (4, 6)}


def blur(act, fwhm_mm):
    sig = np.asarray(fwhm_mm, float) * FWHM_TO_SIGMA
    return nd.gaussian_filter(act, sig, mode="constant", truncate=3.0)


def gradient_slopes(act_lab, dy, true_pa: float) -> dict:
    """Per side, the slope k (1/mm) of SBR(y) = SBR_AC * (1 - k * dy) that gives
    the requested true posterior/anterior SBR ratio over the anatomical
    putamen split at the AC plane (dy = mm posterior to the AC)."""
    out = {}
    for side, (a, p) in PUT.items():
        ma, mp = dy[act_lab == a].mean(), dy[act_lab == p].mean()
        out[side] = float((1 - true_pa) / (mp - true_pa * ma))
    return out


def components(act_lab, dy):
    """The component images (unblurred), in COMPONENTS order."""
    out = [(act_lab == c).astype(np.float32) for c in BG_CODES]
    out += [(act_lab == 1).astype(np.float32), (act_lab == 2).astype(np.float32)]
    put = [np.isin(act_lab, PUT[s]).astype(np.float32) for s in "RL"]
    return out + put + [(p * dy).astype(np.float32) for p in put]


def coefficients(background: dict, put_sbr, caud_sbr, slopes=None):
    slopes = slopes or {"R": 0.0, "L": 0.0}
    c = [background[LABELS[k]] for k in BG_CODES]
    c += [1 + caud_sbr, 1 + caud_sbr, 1 + put_sbr, 1 + put_sbr,
          -put_sbr * slopes["R"], -put_sbr * slopes["L"]]
    return np.array(c, float)


def activity_map(act_lab, dy, put_sbr, caud_sbr, background: dict, slopes=None):
    c = coefficients(background, put_sbr, caud_sbr, slopes)
    act = np.zeros(act_lab.shape, np.float32)
    for w, comp in zip(c, components(act_lab, dy)):
        if w:
            act += np.float32(w) * comp
    return act


def simulate(act_lab, dy, fwhm_mm, put_sbr, caud_sbr, background, slopes=None):
    return blur(activity_map(act_lab, dy, put_sbr, caud_sbr, background, slopes), fwhm_mm)


class Basis:
    """Blurred-component VOI means for one anatomical label map, several FWHMs
    and several measurement label maps.

    means[fwhm][name] is an array [n_components, n_ids]: the mean of each
    blurred component over each measurement region `ids` (valid voxels only).
    """

    def __init__(self, act_lab, dy, fwhms, measure: dict, valid, ids):
        self.ids = list(ids)
        self.measure = measure
        self.valid = valid
        self.slopes_cache = {}
        self.act_lab, self.dy = act_lab, dy
        idx = {name: [np.flatnonzero(((L == i) & valid).ravel()) for i in self.ids]
               for name, L in measure.items()}
        comps = components(act_lab, dy)
        self.means = {}
        for fw in fwhms:
            M = {name: np.zeros((len(comps), len(self.ids))) for name in measure}
            for k, comp in enumerate(comps):
                if not comp.any():
                    continue
                g = blur(comp, fw).ravel()
                for name in measure:
                    M[name][k] = [g[ix].mean() if ix.size else np.nan for ix in idx[name]]
            self.means[fw] = M

    def slopes(self, true_pa):
        if true_pa is None:
            return {"R": 0.0, "L": 0.0}
        if true_pa not in self.slopes_cache:
            self.slopes_cache[true_pa] = gradient_slopes(self.act_lab, self.dy, true_pa)
        return self.slopes_cache[true_pa]

    def voi_values(self, fwhm, name, background, put_sbr, caud_sbr, true_pa=None) -> dict:
        c = coefficients(background, put_sbr, caud_sbr, self.slopes(true_pa))
        return dict(zip(self.ids, c @ self.means[fwhm][name]))
