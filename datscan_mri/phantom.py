"""Synthetic digital head phantom (no patient data).

Builds, on a 1-mm "anatomy" grid in the CT frame:
  head with scalp, skull, CSF, cortical grey, white matter; lateral
  ventricles; caudate and putamen; a CSF cleft lateral to each putamen
  (insula/sylvian); cerebellum.
and derives from it
  * a T1-like MRI in its own grid, misaligned by a known rigid motion;
  * a CT (HU) on a 1 x 1 x 2 mm grid;
  * a SPECT (`make_spect`) built differently from the analysis' own
    simulation: anisotropic Gaussian resolution, attenuation by a tissue
    mu-map (skull included) followed by an approximate first-order Chang
    correction that ignores the skull, sampling on 3.9-mm voxels, and Poisson
    noise at a count level calibrated to the reference case.

`build(seed, randomize=True)` draws the anatomy from a seeded random
generator so the tests are not tuned to the segmentation thresholds:
tissue T1 intensities (each +/- ~8 %), putamen and caudate size and position
(per side), and, row by row, the tissue lateral of the putamen: capsule
(white matter, 1-3 mm), insular cortex (grey, 2-4 mm) and sylvian CSF (1-3
mm, missing on 15 % of rows).

Binding scenarios (true specific binding ratio, SBR, at the AC plane):
  uniform          putamen SBR 3.0 front to back, caudate 3.2
  normal_gradient  putamen SBR falls linearly front to back, true P/A 0.84
  posterior_loss   smooth loss: SBR x f(y), f falling from 1 (8 mm anterior
                   to the AC) to 0.30 (18 mm posterior), smoothstep shape
  global_loss      every striatal SBR x 0.6 (age-like uniform reduction)
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

SHAPE = (170, 240, 200)            # z, y, x  (1 mm)
AC = (85, 100, 100)                 # z, y, x index of the anterior commissure
ORIGIN = (-100.0, -120.0, -80.0)    # LPS mm of voxel (x=0, y=0, z=0)

T1 = {"air": 0, "scalp": 120, "skull": 30, "csf": 50, "gm": 135, "wm": 205,
      "caud": 165, "put": 172, "cer": 140}
HU = {"air": -1000, "scalp": 40, "skull": 1000, "csf": 8, "gm": 38, "wm": 30,
      "caud": 36, "put": 36, "cer": 38}
ACT = {"air": 0.0, "scalp": 0.35, "skull": 0.1, "csf": 0.2, "gm": 1.0, "wm": 1.0, "cer": 0.95}
MU = {"air": 0.0, "skull": 0.025}   # 1/mm at 159 keV; everything else soft tissue 0.015
MU_SOFT = 0.015

# rigid motion of the MRI relative to the CT (Euler: rx, ry, rz [rad], tx, ty, tz [mm])
MRI_MOTION = (np.radians(3.0), np.radians(-2.0), np.radians(4.0), 6.0, -8.0, 5.0)
# SPECT resolution (FWHM, mm) of the phantom scanner: [z, y, x]
SPECT_FWHM = (10.5, 12.5, 13.0)
# Counts per unit activity per 3.9-mm voxel. Calibrated so the high-pass noise in
# the occipital reference on the 1-mm grid (SD of image minus its 8-mm box mean,
# over the mean) is about 0.076, the value measured on the reference case (0.96 mCi).
COUNTS_PER_UNIT = 55.0
NORMAL_PA = 0.84


def _ell(d, c, s):
    return sum(((d[i] - c[i]) / s[i]) ** 2 for i in range(3)) <= 1.0


def draw_params(rng=None):
    """Random (or nominal, if rng is None) phantom geometry and intensities."""
    if rng is None:
        return dict(t1=dict(T1), put={"R": ((-2, 1, -24.5), (10, 21, 6)), "L": ((-2, 1, 24.5), (10, 21, 6))},
                    caud={"R": ((12, -8, -18), (7, 14, 3.5)), "L": ((12, -8, 18), (7, 14, 3.5))},
                    cleft_gap=(3, 3), cleft_width=(3, 3), insula_width=(0, 0), cleft_drop=0.0)
    u = rng.uniform
    t1 = {k: (v * u(0.92, 1.08) if v else 0) for k, v in T1.items()}
    t1["put"] = min(t1["put"], 0.9 * t1["wm"])      # keep the tissue order of a T1 image
    t1["caud"] = min(t1["caud"], 0.9 * t1["wm"])
    put, caud = {}, {}
    for s, sg in (("R", -1), ("L", +1)):
        put[s] = ((-2 + u(-1.5, 1.5), 1 + u(-2, 2), sg * (24.5 + u(-1.5, 1.5))),
                  (10 * u(0.9, 1.1), 21 * u(0.88, 1.08), 6 * u(0.85, 1.1)))
        caud[s] = ((12 + u(-1, 1), -8 + u(-1.5, 1.5), sg * (18 + u(-1, 1))),
                   (7 * u(0.9, 1.1), 14 * u(0.9, 1.1), 3.5 * u(0.85, 1.15)))
    return dict(t1=t1, put=put, caud=caud, cleft_gap=(1, 3), insula_width=(2, 4), cleft_width=(1, 3),
                cleft_drop=0.15)


def anatomy(prm=None, rng=None):
    """Return dict of boolean masks and the tissue-name map on the anatomy grid."""
    prm = prm or draw_params(None)
    rng = rng or np.random.default_rng(0)
    Z, Y, X = np.indices(SHAPE, dtype=np.float32)
    d = (Z - AC[0], Y - AC[1], X - AC[2])
    c = (5.0, 15.0, 0.0)
    head = _ell(d, c, (85, 108, 80))
    skull_out = _ell(d, c, (80, 103, 75))
    ic = _ell(d, c, (73, 96, 68))
    brain = _ell(d, c, (71, 94, 66))
    wm = _ell(d, c, (68, 91, 63))
    tissue = np.full(SHAPE, "air", dtype=object)
    tissue[head] = "scalp"
    tissue[skull_out] = "skull"
    tissue[ic] = "csf"
    tissue[brain] = "gm"
    tissue[wm] = "wm"
    cer = _ell(d, (-40, 62, 0), (18, 30, 48)) & brain
    tissue[cer] = "cer"
    caud = {s: _ell(d, *prm["caud"][s]) for s in "RL"}
    put = {s: _ell(d, *prm["put"][s]) for s in "RL"}
    for s in "RL":
        tissue[caud[s]] = "caud"
    for s in "RL":
        tissue[put[s]] = "put"
    vent = (_ell(d, (14, 5, -9), (9, 30, 7)) | _ell(d, (14, 5, 9), (9, 30, 7))
            | ((np.abs(d[2]) <= 3) & (d[0] >= 8) & (d[0] <= 14) & (d[1] >= -5) & (d[1] <= 10)))
    tissue[vent] = "csf"
    for s in "RL":
        put[s] &= ~vent
        caud[s] &= ~vent & ~put[s]
    # Lateral of each putamen row: external/extreme capsule (white matter, `cleft_gap`
    # mm), insular cortex (grey, `insula_width` mm) and sylvian CSF (`cleft_width` mm,
    # missing on a fraction `cleft_drop` of rows). All randomised per row.
    g0, g1 = prm["cleft_gap"]
    w0, w1 = prm["cleft_width"]
    i0, i1 = prm["insula_width"]
    for s, sg in (("R", -1), ("L", +1)):
        zz, yy = np.nonzero(put[s].any(axis=2))
        for z, y in zip(zz, yy):
            xs = np.nonzero(put[s][z, y])[0]
            e = xs.min() if sg < 0 else xs.max()
            k = int(rng.integers(g0, g1 + 1))
            for _ in range(int(rng.integers(i0, i1 + 1))):
                tissue[z, y, e + sg * k] = "gm"
                k += 1
            if rng.random() < prm["cleft_drop"]:
                continue
            for _ in range(int(rng.integers(w0, w1 + 1))):
                tissue[z, y, e + sg * k] = "csf"
                k += 1
    return dict(tissue=tissue, put=put, caud=caud, vent=vent, ic=ic, d=d, prm=prm)


def _vol(arr, origin=ORIGIN, spacing=(1.0, 1.0, 1.0)):
    im = sitk.GetImageFromArray(arr.astype(np.float32))
    im.SetOrigin(origin)
    im.SetSpacing(spacing)
    return im


def _lut(tissue, table, default=0.0):
    out = np.full(SHAPE, default, np.float32)
    for k in np.unique(tissue):
        out[tissue == k] = table.get(k, default)
    return out


def make_mri(anat, rng):
    t1 = nd.gaussian_filter(_lut(anat["tissue"], anat["prm"]["t1"]), 0.5)
    t1 = t1 + rng.normal(0, 4.0, t1.shape).astype(np.float32)
    src = _vol(np.clip(t1, 0, None))
    motion = sitk.Euler3DTransform()
    motion.SetCenter(_vol(t1).TransformIndexToPhysicalPoint((AC[2], AC[1], AC[0])))
    motion.SetParameters(MRI_MOTION)   # maps CT-frame points -> MRI points
    ref = sitk.Image([192, 230, 176], sitk.sitkFloat32)
    ref.SetOrigin((-92.0, -125.0, -95.0))
    ref.SetSpacing((1.0, 1.0, 1.0))
    mri = sitk.Resample(src, ref, motion.GetInverse(), sitk.sitkLinear, 0.0)
    return mri, motion


def make_ct(anat, rng):
    hu = nd.gaussian_filter(_lut(anat["tissue"], HU), 0.5)
    hu = hu + rng.normal(0, 5.0, hu.shape).astype(np.float32)
    ref = sitk.Image([SHAPE[2], SHAPE[1], SHAPE[0] // 2], sitk.sitkFloat32)
    ref.SetOrigin(ORIGIN)
    ref.SetSpacing((1.0, 1.0, 2.0))
    return sitk.Resample(_vol(hu), ref, sitk.Transform(), sitk.sitkLinear, -1000.0)


def flip_y(im: sitk.Image) -> sitk.Image:
    """The SAME physical image stored with its y axis reversed (direction
    cosine -1 along y, origin moved to the far end)."""
    a = sitk.GetArrayFromImage(im)[:, ::-1, :]
    out = sitk.GetImageFromArray(np.ascontiguousarray(a))
    out.SetSpacing(im.GetSpacing())
    n = im.GetSize()
    out.SetOrigin(im.TransformIndexToPhysicalPoint((0, n[1] - 1, 0)))
    D = np.array(im.GetDirection()).reshape(3, 3)
    D[:, 1] *= -1
    out.SetDirection(tuple(D.ravel()))
    return out


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def posterior_loss_factor(dy, post_fraction=0.30, start=-8.0, end=18.0):
    return 1.0 - (1.0 - post_fraction) * smoothstep((dy - start) / (end - start))


def gradient_slope(put, dy, true_pa=NORMAL_PA):
    ma, mp = dy[put & (dy < 0)].mean(), dy[put & (dy >= 0)].mean()
    return (1 - true_pa) / (mp - true_pa * ma)


def putamen_sbr(anat, side, scenario, put_sbr=3.0, post_fraction=0.30, global_factor=0.6):
    """SBR at each voxel of the true putamen (array over put[side])."""
    dy = anat["d"][1]
    p = anat["put"][side]
    y = dy[p]
    f = global_factor if scenario == "global_loss" else 1.0
    if scenario == "normal_gradient":
        return put_sbr * (1 - gradient_slope(p, dy) * y)
    if scenario == "posterior_loss":
        return put_sbr * posterior_loss_factor(y, post_fraction)
    return np.full(y.shape, put_sbr * f, np.float32)


def true_post_ant(anat, scenario):
    dy = anat["d"][1]
    out = {}
    for s in "RL":
        sb = putamen_sbr(anat, s, scenario)
        y = dy[anat["put"][s]]
        out[s] = float(sb[y >= 0].mean() / sb[y < 0].mean())
    return out


def activity(anat, scenario="uniform", caud_sbr=3.2, global_factor=0.6, **kw):
    act = _lut(anat["tissue"], ACT)
    f = global_factor if scenario == "global_loss" else 1.0
    for s in "RL":
        act[anat["caud"][s]] = 1 + caud_sbr * f
        act[anat["put"][s]] = 1 + putamen_sbr(anat, s, scenario, global_factor=global_factor, **kw)
    return act


def attenuation_residual(anat):
    """Gain left in an attenuation-'corrected' image: true attenuation (soft
    tissue + skull) over a first-order Chang correction that assumes uniform
    soft tissue inside the head outline (no skull). Both use the mean of
    exp(-line integral) over the four in-plane directions."""
    t = anat["tissue"]
    head = t != "air"
    mu_true = np.where(head, MU_SOFT, 0.0).astype(np.float32)
    mu_true[t == "skull"] = MU["skull"]
    mu_chang = np.where(head, MU_SOFT, 0.0).astype(np.float32)

    def mean_atten(mu):
        acc = np.zeros(SHAPE, np.float32)
        for ax in (1, 2):
            c = np.cumsum(mu, axis=ax)
            acc += np.exp(-(c - mu / 2))                                   # toward the low-index edge
            acc += np.exp(-(np.take(c, [-1], axis=ax) - c + mu / 2))     # toward the high-index edge
        return acc / 4

    return mean_atten(mu_true) / mean_atten(mu_chang)


def make_spect(anat, scenario, rng, fwhm=SPECT_FWHM, counts_per_unit=COUNTS_PER_UNIT, attenuation=True,
               act=None):
    act = activity(anat, scenario) if act is None else act
    if attenuation:
        if "_att" not in anat:
            anat["_att"] = attenuation_residual(anat)
        act = act * anat["_att"]
    sig = np.asarray(fwhm, float) / 2.3548
    act = nd.gaussian_filter(act, sig, mode="constant")
    vox = 3.9
    n = [int(np.ceil(SHAPE[2] / vox)), int(np.ceil(SHAPE[1] / vox)), int(np.ceil(SHAPE[0] / vox))]
    ref = sitk.Image(n, sitk.sitkFloat32)
    ref.SetOrigin((ORIGIN[0] + 1.2, ORIGIN[1] + 0.7, ORIGIN[2] + 1.6))
    ref.SetSpacing((vox,) * 3)
    s = sitk.GetArrayFromImage(sitk.Resample(_vol(act), ref, sitk.Transform(), sitk.sitkLinear, 0.0))
    s = rng.poisson(np.clip(s, 0, None) * counts_per_unit).astype(np.float32)
    out = sitk.GetImageFromArray(s)
    out.CopyInformation(ref)
    return out


def ac_lps(offset_mm=(0.0, 0.0, 0.0)):
    """True AC in LPS mm, plus an optional (x, y, z) offset (an imperfect AC pick)."""
    return [ORIGIN[0] + AC[2] + offset_mm[0], ORIGIN[1] + AC[1] + offset_mm[1], ORIGIN[2] + AC[0] + offset_mm[2]]


def build(seed=0, randomize=True):
    """Anatomy + MRI + CT (scenario-independent)."""
    rng = np.random.default_rng(seed)
    prm = draw_params(rng if randomize else None)
    anat = anatomy(prm, rng)
    mri, motion = make_mri(anat, rng)
    ct = make_ct(anat, rng)
    return anat, mri, ct, motion
