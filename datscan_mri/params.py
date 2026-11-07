"""Method parameters.

All spatial parameters are integer offsets in mm (= voxels of the 1-mm
analysis grid) RELATIVE TO THE ANTERIOR COMMISSURE (AC), ordered
[dz, dy, dx] (superior, posterior, patient-left positive). Boxes are
[[z0, z1], [y0, y1], [x0, x1]] half-open, like Python slices.

Generic defaults vs. the reference case
---------------------------------------
Only one real case has been analysed, so the segmentation thresholds and
boxes below were set by eye on it. They are kept here as STARTING VALUES and
are pinned, together with the case's AC, in `configs/reference_case.json`, so
the reference result never changes when a default is edited.

Intensity thresholds are expressed in "reference units": by default
(`intensity.normalize_p90 = true`) every MRI is first rescaled so the 90th
percentile of its eroded intracranial intensities equals
`intensity.reference_p90` (190.65 = that percentile on the reference case's
N4-corrected MPRAGE; white matter then sits around 170-200, cortical grey
105-158, CSF below 105). This is a unit convention, not a fitted value. A
different scanner or protocol still needs the thresholds checked on
`figures/fig2_segmentation.png`; adjust with a JSON passed to `--params`.

`ac_from_ic_centroid` is only used when no AC is given, and then the run
stops unless `--allow-estimated-ac` is passed. The default offset is a
rounded adult-head value (measured on the reference case).
"""
from __future__ import annotations

import copy
import json

DEFAULTS: dict = {
    # AC location as a physical point (DICOM LPS mm, CT frame). None -> rough
    # estimate from the intracranial centroid (always verify on the figures).
    "ac_lps_mm": None,
    "ac_from_ic_centroid": [8.0, -27.0, 0.0],  # [dz, dy, dx] used only by the estimate
    "intensity": {"normalize_p90": True, "reference_p90": 190.65},
    "registration": {"bins": 64, "sampling": 0.05, "seed": 1, "ct_clamp_hu": [-200, 1500],
                     "learning_rate": 2.0, "min_step": 1e-4, "iterations": 400, "relaxation": 0.6,
                     "shrink": [8, 4, 2, 1], "smooth_mm": [4, 2, 1, 0]},
    "n4": {"iterations": [50, 50, 30], "shrink": 2},
    "ventricles": {"smooth_sigma": 1.0, "csf_lo": 25, "csf_hi": 110, "open_iter": 2,
                   "box": [[-18, 42], [-38, 67], [-39, 41]], "seed": [12, 2, -14]},
    "putamen": {
        "smooth_sigma": 1.0,
        "z": [-16, 10], "y": [-21, 24],          # rows scanned (inclusive)
        "scan_start": 50,                         # start this far lateral of midline
        "csf_thr": 135,                           # 'CSF-ish' (insula / sylvian)
        "csf_min_mid": 22, "max_lat": 46,         # CSF must lie this far from midline
        "csf_dvent": 4,                           # ... and > this from the ventricles
        "lat_gap": 3,                             # skip insular cortex/claustrum/capsule
        "lo": 150, "hi": 194,                     # putamen grey range; >= hi stops (WM)
        "dvent": 3, "min_mid": 14, "max_w": 13, "min_w": 3,
        # Case-specific LEFT-only fallback on low slices where the sylvian CSF was not
        # found (off by default; the reference case uses low_z -7, max_lat_low_L 34).
        # It makes the left and right VOIs differ in construction.
        "low_z": None, "max_lat_low_L": 34,
        "open_iter": 1, "close_iter": 2,
    },
    "caudate": {"smooth_sigma": 1.0, "z": [-9, 14], "y": [-24, 12],
                "wall_split_R": 3, "wall_split_L": -3,
                "lo": 120, "hi": 196, "max_w": 5, "min_w": 2, "mean_max": 190,
                "close_iter": 1, "drop_post_inferior": [0, 4]},  # drop dy>0 & dz<4
    # Striatal segmenter. "threshold" = the semi-automatic row scan below (no external
    # tool). "synthseg" / "fastsurfer" run an established anatomical segmenter from its
    # own environment (see datscan_mri/external.py); "labels_file" loads a precomputed
    # FreeSurfer-LUT label map. Only putamen and caudate come from the segmenter; the
    # ventricles (registration QC) and background regions keep the intensity method.
    # `compare_labels_files` ({name: path}) and, for an external method, the threshold
    # method are run as extra segmentation variants (Dice + sweep rows).
    "segmenter": {"method": "threshold", "labels_file": None, "labels_file_info": None,
                  "synthseg": {"home": None, "robust": False, "threads": 8},
                  "fastsurfer": {"home": None, "threads": 8, "device": "cpu"},
                  "min_qc_score": 0.65,          # SynthSeg QC (Billot 2023): below 0.65 = likely failed
                  "compare_threshold": True, "compare_labels_files": {}},
    "intracranial": {"hu_lo": -30, "hu_hi": 120, "erode": 3, "dilate": 3, "seed": [12, 12, 0]},
    "regions": {
        "smooth_sigma": 0.7, "csf_hi": 105, "gm_hi": 158,
        "z_min": -60,                                   # ignore intracranial below AC-60
        "occ": {"y_min": 77, "z": [-8, 42]},            # occipital cortical grey
        "cer": {"y_min": 47, "z_max": -26, "mid_gap": 8},  # cerebellum
        "deep": {"x": [-29, 31], "y": [-13, 42], "z": [-28, 12]},
        "head_hu": -300,
    },
    "analysis": {"crop_margin": 5, "split_offset": 0},
    "profile": {"slab_mm": 2, "min_voxels": 20},
    # Quality gates: a run that fails one stops with a non-zero exit code.
    "qc": {"min_ventricle_dice": 0.6,           # CT-vs-MRI ventricle overlap after registration
           "max_voi_outside_fov_frac": 0.01,    # per striatal VOI / occipital reference, outside the SPECT support
           "min_putamen_voi_ml": 4.0,           # smaller per-side VOI = collapsed segmentation (threshold method)
           "external_putamen_ml": [2.5, 8.0],   # plausible per-side putamen from an anatomical segmenter
           "allow_estimated_ac": False},
    "simulation": {
        "fwhm_expected": [9, 11, 13, 15],
        "sbr_levels": [[2.5, 2.7], [4.0, 4.0]],   # [putamen, caudate] true SBR pairs (at the AC plane)
        "fwhm_profile": 12, "putamen_sbr": 2.5, "caudate_sbr": 2.7,
        # The simulated binding fills an ANATOMICAL putamen, not the measurement VOI:
        # the grey-matter core of the VOI (white matter removed only if the VOI holds
        # any), shrunk to at most `cap` mL per side (null = no cap). This size is an
        # assumption; the sweep values are all used for the expected ranges and the
        # decision. The measurement VOI is never changed by this.
        "anatomical_putamen_cap_ml": 5.0,
        "anatomical_putamen_cap_ml_sweep": [4.0, 5.0, 6.0, None],
        # Normal-gradient null: putamen SBR falls linearly front to back so that the
        # true posterior/anterior SBR ratio (split at the AC) equals these values.
        # 0.84 = posterior/anterior putamen in the DaTQUANT normal values for age 65
        # (Grisanti 2023, PMC10338803, Table 1: 2.09/2.49 L, 2.10/2.51 R; a different
        # pipeline and template, so only the order of magnitude transfers).
        "normal_gradient_pa": 0.84,
        "normal_gradient_pa_range": [0.80, 0.84, 0.88],
        # relative activity of non-striatal compartments (occipital grey = 1)
        "background": {"CSF": 0.0, "deep": 1.4, "WM": 1.1, "GMctx": 1.0,
                       "extracran": 0.35, "occGM": 1.0, "cerGM": 0.95},
        # other plausible backgrounds, to show how much the expectation depends on them
        "background_alternatives": {
            "flat": {"CSF": 0.2, "deep": 1.0, "WM": 1.0, "GMctx": 1.0, "extracran": 0.35, "occGM": 1.0, "cerGM": 0.95},
            "low_wm": {"CSF": 0.0, "deep": 1.2, "WM": 0.8, "GMctx": 1.0, "extracran": 0.35, "occGM": 1.0, "cerGM": 0.95},
        },
    },
    "gtm": {"fwhm": [9, 11, 13]},
    # Measurement-VOI sweep (simulated binding held fixed): grow (+) or shrink (-) the
    # putamen VOI by this many mm, the whole putamen or the posterior VOI only.
    # Threshold sweep: re-segment with these offsets added to the putamen thresholds
    # (both the VOI and the anatomical putamen are rebuilt).
    "sweep": {"voi_mm": [-2, -1, 1, 2, 4, 6],
              "threshold": {"lo": [-10, 10], "hi": [-6, 6]},
              "fwhm": [9, 11, 13, 15]},
    # A posterior loss is called only if the measured P/A is below the LOWEST value the
    # normal-gradient null produces (over FWHM, binding level, background, anatomical
    # volume and true P/A 0.80-0.88) by more than this margin.
    "decision": {"margin": 0.05},
    "robustness": {"shift_mm": 2.0, "rot_deg": 2.0, "split_shift_mm": 4, "erode_mm": 1, "gtm_fwhm": 11},
    "trim_mm": [8, 9, 10, 12, 13],
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | None = None, **overrides) -> dict:
    P = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            P = _merge(P, json.load(f))
    return _merge(P, {k: v for k, v in overrides.items() if v is not None})
