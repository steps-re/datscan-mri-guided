"""Geometric transfer matrix (GTM) partial-volume correction.

Rousset OG, Ma Y, Evans AC. Correction for partial volume effects in PET:
principle and validation. J Nucl Med 1998;39:904-911.

W[j, i] = mean over region j of (indicator of region i blurred by the PSF).
Observed regional means t_obs = W @ t_true, solved for t_true. The PSF is an
isotropic Gaussian of the given FWHM (the true SPECT resolution at the
striatum is not known exactly, so several FWHMs are reported).
"""
from __future__ import annotations

import numpy as np

from .simulate import blur


def transfer_matrix(lab, ids, fwhm_mm, valid):
    idx = {i: np.nonzero((lab == i) & valid) for i in ids}
    W = np.zeros((len(ids), len(ids)))
    for a, i in enumerate(ids):
        g = blur((lab == i).astype(np.float32), fwhm_mm)
        for b, j in enumerate(ids):
            W[b, a] = g[idx[j]].mean()
    return W


def gtm(obs, W):
    return np.linalg.solve(W, obs)
