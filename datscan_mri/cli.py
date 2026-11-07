"""Command line: `datscan-mri run|convert|inventory|export-viewer`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _ac(s):
    return [float(v) for v in s.split(",")] if s else None


def cmd_inventory(a):
    from .io import inventory
    for r in inventory(a.dir):
        print(f"{r['modality']:3s} {r['n_files']:4d} files  {str(r['rows'])}x{str(r['cols'])}  "
              f"{r['description'][:40]:40s} {r['path']}")


def cmd_convert(a):
    from .io import read_series, write_image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("spect", "ct", "mri"):
        d = getattr(a, k)
        if d:
            im = read_series(d, getattr(a, f"{k}_series"))
            write_image(im, out / f"{k}.nii.gz")
            print(k, im.GetSize(), [round(v, 4) for v in im.GetSpacing()])


def cmd_run(a):
    import SimpleITK as sitk
    from . import params, pipeline, report
    from .export import export_viewer
    from .io import read_series

    t0 = time.time()
    P = params.load(a.params, ac_lps_mm=_ac(a.ac))
    P["qc"]["allow_estimated_ac"] = bool(a.allow_estimated_ac) or P["qc"]["allow_estimated_ac"]
    for k, v in (("min_ventricle_dice", a.min_ventricle_dice), ("min_putamen_voi_ml", a.min_putamen_ml),
                 ("max_voi_outside_fov_frac", a.max_outside_fov)):
        if v is not None:
            P["qc"][k] = v
    Sg = P["segmenter"]
    if a.seg_labels:
        Sg["method"], Sg["labels_file"] = "labels_file", a.seg_labels
        if a.seg_labels_info:
            Sg["labels_file_info"] = json.load(open(a.seg_labels_info))
    elif a.segmenter:
        Sg["method"] = a.segmenter
    for spec in a.compare_labels or []:
        name, _, path = spec.partition("=")
        Sg.setdefault("compare_labels_files", {})[name] = path
    if a.min_segmenter_qc is not None:
        Sg["min_qc_score"] = a.min_segmenter_qc
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("reading inputs...")
    spect = read_series(a.spect, a.spect_series)
    ct = read_series(a.ct, a.ct_series)
    mri = read_series(a.mri, a.mri_series)
    tx = sitk.ReadTransform(a.transform) if a.transform else None
    anat = pipeline.prepare_anatomy(ct, mri, P, transform=tx, workdir=out / "derived" / "segmenter")
    res = pipeline.analyze(anat, spect, P, do_robustness=not a.skip_robustness, do_sweeps=not a.skip_sweeps)
    sg = res.pop("_sg")
    sims = res.pop("_sim")
    res.pop("_fov")
    pipeline.save_derived(anat, sg, out, sims)
    json.dump(P, open(out / "params_used.json", "w"), indent=1)
    doc = report.write_all(anat, res, out)
    if a.viewer_out:
        export_viewer(anat.mg, anat.ct, sg, anat.lab, anat.ic, anat.ac, anat.grid.GetDirection(),
                      a.viewer_out, results=doc, figures_dir=out / "figures", sims=sims)
    _summary(doc)
    print(f"done in {time.time() - t0:.0f} s -> {out}")


def cmd_export(a):
    import nibabel as nib
    from .export import export_viewer

    d = Path(a.run) / "derived"
    ld = lambda n: np.transpose(np.asarray(nib.load(d / n).dataobj), (2, 1, 0))  # noqa: E731
    meta = json.load(open(d / "anatomy.json"))
    A = nib.load(d / "mri_grid.nii.gz").affine
    D = (np.diag([-1.0, -1.0, 1.0]) @ A[:3, :3]).ravel()
    doc = json.load(open(Path(a.run) / "results.json"))
    sims = {k: ld(f"sim_{k}_grid.nii.gz") for k in ("uniform", "normal_gradient")
            if (d / f"sim_{k}_grid.nii.gz").exists()}
    g = export_viewer(ld("mri_grid.nii.gz"), ld("ct_grid.nii.gz"), ld("spect_grid.nii.gz"),
                      ld("labels_grid.nii.gz"), ld("intracranial_grid.nii.gz").astype(bool),
                      tuple(meta["ac_index_zyx"]), D, a.dest, results=doc,
                      figures_dir=Path(a.run) / "figures", sims=sims)
    print(json.dumps({k: g[k] for k in ("full_fov", "shape_xyz")}))


def cmd_flair(a):
    import SimpleITK as sitk
    import nibabel as nib
    from . import params
    from .export import register_flair, viewer_affine
    from .io import read_series, write_clean_nifti

    d = Path(a.run) / "derived"
    ld = lambda n: np.transpose(np.asarray(nib.load(d / n).dataobj), (2, 1, 0))  # noqa: E731
    grid = sitk.ReadImage(str(d / "mri_grid.nii.gz"))
    tx = sitk.ReadTransform(str(d / "mri_to_ct.tfm"))
    meta = json.load(open(d / "anatomy.json"))
    P = params.load(str(Path(a.run) / "params_used.json"))
    arr, info = register_flair(read_series(a.flair, a.flair_series), read_series(a.mri, a.mri_series), tx, grid,
                               ld("ventricles_grid.nii.gz").astype(bool), ld("intracranial_grid.nii.gz").astype(bool), P)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    A = viewer_affine(grid.GetDirection(), tuple(meta["ac_index_zyx"]))
    write_clean_nifti(np.clip(np.rint(arr), -32768, 32767).astype(np.int16), A, dest / "flair.nii.gz")
    json.dump(info, open(dest / "flair_registration_qc.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    if info["ventricle_dice_flair_vs_t1"] < a.min_dice:
        from .pipeline import QCError
        raise QCError(f"FLAIR ventricle Dice {info['ventricle_dice_flair_vs_t1']:.2f} < {a.min_dice}")


def _summary(doc):
    u = doc["uncorrected"]["occ"]
    S = doc["simulation"]
    eu, eg = S["uniform"]["expected_post_ant_range"], S["normal_gradient"]["expected_post_ant_range"]
    f = lambda r: f"{r[0]:.2f}-{r[1]:.2f}"  # noqa: E731
    print("\nPosterior/anterior putamen SBR ratio (occipital reference)")
    print(f"  measured                          R {u['R']['post_ant']:.2f}        L {u['L']['post_ant']:.2f}")
    print(f"  expected, normal gradient (0.80-0.88)  R {f(eg['R'])}   L {f(eg['L'])}   <- primary comparator")
    print(f"  expected, uniform binding (lenient)    R {f(eu['R'])}   L {f(eu['L'])}")
    if "sweep" in doc:
        g = doc["sweep"]["gap_vs_normal_gradient_low_range"]
        print(f"  measured - normal-gradient low, over VOI/segmentation sweep: R {f(g['R'])}  L {f(g['L'])}")
    dc = doc["decision"]
    print(f"  posterior loss called (margin {dc['margin']}): R {dc['posterior_loss_called']['R']}  "
          f"L {dc['posterior_loss_called']['L']}")
    for fw, g in doc["gtm"].items():
        print(f"  GTM {fw:>2s} mm (assumes uniform VOIs)  R {g['occ']['R']['post_ant']:.2f}   "
              f"L {g['occ']['L']['post_ant']:.2f}")
    print("  " + doc["disclaimer"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="datscan-mri", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory", help="list DICOM series under a folder")
    p.add_argument("dir")
    p.set_defaults(f=cmd_inventory)
    p = sub.add_parser("convert", help="DICOM series -> NIfTI (geometry-only headers)")
    for k in ("spect", "ct", "mri"):
        p.add_argument(f"--{k}")
        p.add_argument(f"--{k}-series", help="SeriesInstanceUID if the folder holds several series")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_convert)
    p = sub.add_parser("run", help="full analysis")
    p.add_argument("--spect", required=True, help="attenuation-corrected SPECT series dir (or NIfTI)")
    p.add_argument("--ct", required=True, help="low-dose CT series dir from the same SPECT/CT (or NIfTI)")
    p.add_argument("--mri", required=True, help="3-D T1 (MPRAGE) series dir (or NIfTI)")
    p.add_argument("--out", required=True)
    p.add_argument("--params", help="JSON overriding defaults (see configs/)")
    p.add_argument("--ac", help="anterior commissure, DICOM LPS mm in the CT frame: x,y,z")
    p.add_argument("--transform", help="reuse a saved MRI->CT transform (.tfm) instead of registering")
    for k in ("spect", "ct", "mri"):
        p.add_argument(f"--{k}-series", help=f"SeriesInstanceUID to use if the {k} folder holds several series")
    p.add_argument("--allow-estimated-ac", action="store_true",
                   help="run without --ac using a rough AC estimate (otherwise the run stops)")
    p.add_argument("--min-ventricle-dice", type=float, help="QC gate on CT-vs-MRI ventricle Dice (default 0.6)")
    p.add_argument("--min-putamen-ml", type=float, help="QC gate on each putamen VOI volume (default 4.0 mL)")
    p.add_argument("--max-outside-fov", type=float,
                   help="QC gate: max fraction of VOI voxels outside the SPECT support (default 0.01)")
    p.add_argument("--segmenter", choices=["threshold", "synthseg", "fastsurfer"],
                   help="putamen/caudate segmenter (default from params: threshold; the reference case uses synthseg)")
    p.add_argument("--seg-labels", help="use a precomputed FreeSurfer-LUT label map (NIfTI/MGZ, any space with a "
                                        "correct affine) as the segmentation, e.g. derived/segmenter_fs_labels_grid.nii.gz")
    p.add_argument("--seg-labels-info", help="JSON describing --seg-labels (tool, version, model) for results.json")
    p.add_argument("--compare-labels", action="append", metavar="NAME=PATH",
                   help="another FreeSurfer-LUT label map to compare (Dice) and run as a segmentation variant")
    p.add_argument("--min-segmenter-qc", type=float, help="QC gate on SynthSeg QC scores (default 0.65)")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--skip-sweeps", action="store_true", help="skip the VOI-size and threshold sweeps")
    p.add_argument("--viewer-out", help="also write the de-identified viewer export here")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("export-viewer", help="de-identified cropped volumes + JSON for a web viewer")
    p.add_argument("--run", required=True, help="output folder of a previous `run`")
    p.add_argument("--dest", required=True)
    p.set_defaults(f=cmd_export)
    p = sub.add_parser("add-flair", help="register a FLAIR (same session as the T1) into a run's grid")
    p.add_argument("--run", required=True, help="output folder of a previous `run`")
    p.add_argument("--mri", required=True, help="the T1 used for that run")
    p.add_argument("--flair", required=True)
    p.add_argument("--mri-series")
    p.add_argument("--flair-series")
    p.add_argument("--dest", required=True, help="viewer export folder (flair.nii.gz is written there)")
    p.add_argument("--min-dice", type=float, default=0.5, help="QC gate on FLAIR-vs-T1 ventricle Dice")
    p.set_defaults(f=cmd_flair)
    a = ap.parse_args(argv)
    from .io import SeriesSelectionError
    from .pipeline import QCError
    try:
        a.f(a)
    except QCError as e:
        print(f"QC FAILURE: {e}", file=sys.stderr)
        return 2
    except SeriesSelectionError as e:
        print(f"INPUT ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
