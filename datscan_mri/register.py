"""Rigid MRI -> CT registration (Mattes mutual information), N4 bias correction,
and resampling of every modality onto one 1-mm analysis grid in the CT frame.

SPECT and CT come from one SPECT/CT session and share a frame, so SPECT is
resampled with the identity transform. Only the MRI is registered.

The analysis grid is ALWAYS axis-aligned in DICOM LPS (identity direction),
whatever the direction cosines of the CT. Array index [z, y, x] therefore
always means (superior, posterior, patient-left) increasing, which the
segmentation and the coronal-slab profile rely on. A CT stored with flipped or
oblique axes is resampled onto this grid; the physical image is unchanged.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def make_grid(ct: sitk.Image) -> sitk.Image:
    """1-mm isotropic grid with IDENTITY (LPS) direction covering the CT's
    physical bounding box: origin at the lowest CT voxel centre, and enough
    voxels per axis to reach both the highest CT voxel centre and the CT's
    footprint (n * spacing). For an identity-direction CT the origin and voxel
    centres are the CT's own; versions before 0.2 used round(n * spacing)
    voxels, which could stop short of the last CT voxel centre (one voxel
    fewer on the reference case, outside the head)."""
    n = np.array(ct.GetSize())
    corners = []
    for c in np.ndindex(2, 2, 2):
        idx = [float(c[i] * (n[i] - 1)) for i in range(3)]
        corners.append(ct.TransformContinuousIndexToPhysicalPoint(idx))
    corners = np.array(corners)
    lo, hi = corners.min(0), corners.max(0)
    D = np.array(ct.GetDirection()).reshape(3, 3)
    # extent along each LPS axis of the CT's voxel footprint (n * spacing for axis-aligned CTs)
    ext = np.abs(D) @ (n * np.array(ct.GetSpacing()))
    size = [max(int(round(e)), int(np.ceil(h - l)) + 1) for e, l, h in zip(ext, lo, hi)]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetOrigin(tuple(float(v) for v in lo))
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection(IDENTITY)
    return ref


def spect_fov_mask(spect: sitk.Image, ref: sitk.Image, min_frac: float = 0.99) -> np.ndarray:
    """Voxels of the analysis grid that lie inside the SPECT's reconstructed
    support: inside the image bounds AND not in the zero padding a
    reconstruction leaves outside its circular/body mask (and in empty end
    slices). The support is, slice by slice, the non-zero voxels with interior
    holes filled. A grid voxel is valid if its linear-interpolated support
    fraction is >= min_frac."""
    a = sitk.GetArrayFromImage(spect) != 0
    sup = np.zeros_like(a)
    for z in range(a.shape[0]):
        if a[z].any():
            sup[z] = nd.binary_fill_holes(nd.binary_closing(a[z], iterations=2))
    im = sitk.GetImageFromArray(sup.astype(np.float32))
    im.CopyInformation(spect)
    f = sitk.GetArrayFromImage(sitk.Resample(im, ref, sitk.Transform(), sitk.sitkLinear, 0.0))
    return f >= min_frac


def register_mri_to_ct(ct: sitk.Image, mri: sitk.Image, P: dict) -> tuple[sitk.Euler3DTransform, dict]:
    """Return the rigid transform mapping CT-frame points to MRI points, plus info."""
    R = P["registration"]
    ctc = sitk.Clamp(ct, sitk.sitkFloat32, *R["ct_clamp_hu"])
    init = sitk.CenteredTransformInitializer(
        ctc, mri, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(R["bins"])
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(R["sampling"], seed=R["seed"])
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        R["learning_rate"], R["min_step"], R["iterations"], relaxationFactor=R["relaxation"])
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(R["shrink"])
    reg.SetSmoothingSigmasPerLevel(R["smooth_mm"])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(ctc, mri)
    if tx.GetName() == "CompositeTransform":
        tx = sitk.CompositeTransform(tx).GetNthTransform(0)
    e = sitk.Euler3DTransform(tx)
    info = dict(metric=float(reg.GetMetricValue()),
                stop=reg.GetOptimizerStopConditionDescription(),
                parameters=[float(v) for v in e.GetParameters()],
                center=[float(v) for v in e.GetCenter()],
                rotation_deg=[float(np.degrees(v)) for v in e.GetParameters()[:3]],
                translation_mm=[float(v) for v in e.GetParameters()[3:]])
    return e, info


def n4_correct(mri: sitk.Image, P: dict) -> sitk.Image:
    mask = sitk.OtsuThreshold(mri, 0, 1, 200)
    k = P["n4"]["shrink"]
    sh = sitk.Shrink(mri, [k] * 3)
    mk = sitk.Shrink(mask, [k] * 3)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(P["n4"]["iterations"])
    n4.Execute(sh, mk)
    lb = n4.GetLogBiasFieldAsImage(mri)
    return mri / sitk.Exp(lb)


def to_grid(im: sitk.Image, ref: sitk.Image, tx=None, interp=sitk.sitkLinear, default=0.0) -> np.ndarray:
    tx = tx if tx is not None else sitk.Transform()
    return sitk.GetArrayFromImage(sitk.Resample(im, ref, tx, interp, default)).astype(np.float32)


def ventricle_dice(ct_grid: np.ndarray, vent_mri: np.ndarray, seed_idx, hu_max: float = 17.0) -> float:
    """QC: overlap of the lateral ventricles seen on CT with those seen on MRI.
    A well-registered pair gives Dice around 0.7-0.9 (CT CSF contrast is poor)."""
    cs = nd.median_filter(ct_grid, 3)
    m = nd.binary_opening(cs < hu_max, iterations=1) & (cs > -50)
    lab, _ = nd.label(m)
    l = lab[tuple(seed_idx)]
    if l == 0:
        # nearest CT-CSF component to any MRI ventricle voxel
        hits = lab[vent_mri & (lab > 0)]
        if hits.size == 0:
            return float("nan")
        l = np.bincount(hits).argmax()
    vc = lab == l
    return float(2 * (vc & vent_mri).sum() / (vc.sum() + vent_mri.sum()))


def qc_figure(ct_grid: np.ndarray, mri_grid: np.ndarray, ac_idx, path: str, title: str = "") -> None:
    """CT (brain and bone windows) with MRI edges / MRI outline overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z0, y0, x0 = ac_idx
    g = nd.gaussian_gradient_magnitude(mri_grid, 1.0)
    pos = mri_grid[mri_grid > 0]
    thr = np.percentile(g[mri_grid > np.percentile(pos, 50)], 92) if pos.size else 0
    lvl = np.percentile(pos, 60) if pos.size else 0
    views = [("axial", lambda a: a[z0 + 10], "upper"),
             ("coronal", lambda a: a[:, y0, :], "lower"),
             ("sagittal", lambda a: a[:, :, x0 + 8], "lower")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for j, (n, f, o) in enumerate(views):
        ax[0, j].imshow(f(ct_grid), cmap="gray", vmin=-20, vmax=90, origin=o)
        ax[0, j].contour(f(g) > thr, levels=[0.5], colors="r", linewidths=0.5, origin=o)
        ax[0, j].set_title(f"{n}: CT brain window + MRI edges (red)", fontsize=9)
        ax[1, j].imshow(f(ct_grid), cmap="gray", vmin=-200, vmax=1500, origin=o)
        ax[1, j].contour(f(mri_grid), levels=[lvl], colors="c", linewidths=0.5, origin=o)
        ax[1, j].set_title(f"{n}: CT bone window + MRI outline (cyan)", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(title or "Registration QC: MRI resampled into the SPECT/CT frame "
                 "(patient right on image left)")
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)
