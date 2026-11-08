"""Viewer-ready, de-identified export.

Full field-of-view volumes (MRI, SPECT, CT, labels) on the common 1-mm grid in
the SPECT/CT frame, so a reader can see the whole study and re-run
registration. De-identification is by construction: each file gets a
brand-new NIfTI header holding geometry only (no description, no intent
name, no extensions), nothing is copied from DICOM, and the JSON written here
holds no file paths. The affine is re-centred so the anterior commissure is
at (0, 0, 0) mm, RAS. MRI and CT are stored as int16 to keep size down.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .io import write_clean_nifti

LABEL_NAMES = {1: "caudate_R", 2: "caudate_L", 3: "putamen_anterior_R", 4: "putamen_anterior_L",
               5: "putamen_posterior_R", 6: "putamen_posterior_L", 7: "CSF", 8: "deep_grey",
               9: "white_matter", 10: "cortical_grey", 11: "extracranial",
               13: "occipital_grey_reference", 14: "cerebellar_grey_reference"}


def _int16(a):
    return np.clip(np.rint(a), -32768, 32767).astype(np.int16)


def viewer_affine(direction, ac):
    """RAS affine of the 1-mm grid with the AC at (0, 0, 0) mm."""
    D = np.array(direction).reshape(3, 3)
    F = np.diag([-1.0, -1.0, 1.0])
    ac_xyz = np.array([ac[2], ac[1], ac[0]], float)
    A = np.eye(4)
    A[:3, :3] = F @ D
    A[:3, 3] = -(F @ D @ ac_xyz)
    return A


def export_viewer(mg, ct, sg, lab, ic, ac, direction, dest, results=None, figures_dir=None, sims=None):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    A = viewer_affine(direction, ac)

    write_clean_nifti(_int16(mg), A, dest / "mri_t1.nii.gz")
    write_clean_nifti(np.where(sg < 0, 0, sg).astype(np.float32), A, dest / "spect.nii.gz")
    write_clean_nifti(_int16(ct), A, dest / "ct.nii.gz")
    write_clean_nifti(lab.astype(np.uint8), A, dest / "labels.nii.gz")
    for k, v in (sims or {}).items():   # derived layers: simulated nulls (activity, occipital grey = 1)
        write_clean_nifti(np.asarray(v, np.float32), A, dest / f"sim_{k}.nii.gz")

    shape = mg.shape
    geom = {"full_fov": True, "shape_xyz": [int(shape[2]), int(shape[1]), int(shape[0])],
            "voxel_mm": 1.0, "orientation": "RAS",
            "origin": "anterior commissure at (0,0,0) mm",
            "frame": "SPECT/CT (MRI rigidly registered into it)",
            "extent_mm_from_ac": {"x_left_positive": [int(-ac[2]), int(shape[2] - 1 - ac[2])],
                                  "y_posterior_positive": [int(-ac[1]), int(shape[1] - 1 - ac[1])],
                                  "z_superior_positive": [int(-ac[0]), int(shape[0] - 1 - ac[0])]},
            "dtypes": {"mri_t1": "int16 (N4-corrected T1, arbitrary units)", "ct": "int16 (HU)",
                       "spect": "float32 (reconstructed counts; 0 outside the SPECT field of view)",
                       "labels": "uint8",
                       "sim_uniform / sim_normal_gradient": ("float32 simulated activity (occipital grey = 1), "
                                                             "derived layer, not patient data")},
            "labels": LABEL_NAMES}
    json.dump(geom, open(dest / "geometry.json", "w"), indent=1)
    if results is not None:
        p = results["profile"]
        prof = {"units": "uncorrected specific binding ratio, occipital grey reference",
                "x_axis": "mm posterior to the anterior commissure (slab centre); negative = anterior",
                "slab_mm": 2, "simulation_fwhm_mm": p["sim_fwhm_mm"],
                "simulations": "uniform binding, and normal gradient (true posterior/anterior "
                               f"{p['simng_true_post_ant']}); binding in the anatomical putamen",
                "slabs": [{"mm_from_ac": p["mm_from_ac"][i],
                           "R_measured": p["R"][i], "L_measured": p["L"][i],
                           "R_simulated": p["sim_R"][i], "L_simulated": p["sim_L"][i],
                           "R_simulated_scaled_to_peak": p.get("sim_R_peak_scaled", p["sim_R_scaled"])[i],
                           "L_simulated_scaled_to_peak": p.get("sim_L_peak_scaled", p["sim_L_scaled"])[i],
                           "R_simulated_scaled_to_anterior_mean": p["sim_R_scaled"][i],
                           "L_simulated_scaled_to_anterior_mean": p["sim_L_scaled"][i],
                           "R_simulated_normal_gradient": p["simng_R"][i],
                           "L_simulated_normal_gradient": p["simng_L"][i],
                           "R_simulated_normal_gradient_scaled_to_peak": p.get("simng_R_peak_scaled",
                                                                                p["simng_R_scaled"])[i],
                           "L_simulated_normal_gradient_scaled_to_peak": p.get("simng_L_peak_scaled",
                                                                                p["simng_L_scaled"])[i],
                           "R_simulated_normal_gradient_scaled_to_anterior_mean": p["simng_R_scaled"][i],
                           "L_simulated_normal_gradient_scaled_to_anterior_mean": p["simng_L_scaled"][i],
                           "R_simulated_normal_gradient_lowest_scaled_to_anterior_mean":
                               p.get("simnglow_R_scaled", [None] * (i + 1))[i],
                           "L_simulated_normal_gradient_lowest_scaled_to_anterior_mean":
                               p.get("simnglow_L_scaled", [None] * (i + 1))[i],
                           "R_voxels": p["R_n"][i], "L_voxels": p["L_n"][i]}
                          for i in range(len(p["mm_from_ac"]))]}
        json.dump(prof, open(dest / "profile.json", "w"), indent=1)
        json.dump(results, open(dest / "results.json", "w"), indent=1)
    if figures_dir is not None:
        for f in sorted(Path(figures_dir).glob("*.png")):
            shutil.copy(f, dest / f.name)
    return geom


def register_flair(flair, mri, mri_to_ct, grid, vent, ic, P):
    """Rigidly register a FLAIR (same MRI session) to the T1, compose with the
    T1 -> CT transform and resample onto the analysis grid.

    QC: Dice between the lateral ventricles on the registered FLAIR (CSF is
    dark on FLAIR: below the 20th percentile of intracranial FLAIR, opened,
    component touching the T1 ventricles) and the T1 ventricle mask.
    Returns (array on grid, info dict)."""
    import SimpleITK as sitk
    from scipy import ndimage as nd

    R = P["registration"]
    init = sitk.CenteredTransformInitializer(mri, flair, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(R["bins"])
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(R["sampling"], seed=R["seed"])
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(R["learning_rate"], R["min_step"], R["iterations"],
                                                 relaxationFactor=R["relaxation"])
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(mri, flair)            # maps T1 points -> FLAIR points
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(tx)                   # applied last: T1 -> FLAIR
    comp.AddTransform(mri_to_ct)            # applied first: CT -> T1
    a = sitk.GetArrayFromImage(sitk.Resample(flair, grid, comp, sitk.sitkLinear, 0.0)).astype(np.float32)
    sm = nd.gaussian_filter(a, 1.0)
    dark = ic & (sm < np.percentile(sm[ic], 20))
    dark = nd.binary_opening(dark, iterations=1)
    L, _ = nd.label(dark)
    hits = np.unique(L[vent & (L > 0)])
    fv = np.isin(L, hits[hits > 0]) & nd.binary_dilation(vent, iterations=5)
    dice = float(2 * (fv & vent).sum() / max(fv.sum() + vent.sum(), 1))
    e = sitk.Euler3DTransform(sitk.CompositeTransform(tx).GetNthTransform(0)) if tx.GetName() == "CompositeTransform" \
        else sitk.Euler3DTransform(tx)
    info = {"metric_mattes_mi": float(reg.GetMetricValue()),
            "flair_to_t1_rotation_deg": [float(np.degrees(v)) for v in e.GetParameters()[:3]],
            "flair_to_t1_translation_mm": [float(v) for v in e.GetParameters()[3:]],
            "ventricle_dice_flair_vs_t1": dice}
    return a, info
