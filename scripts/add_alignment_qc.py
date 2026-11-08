"""Add the SPECT-to-anatomy alignment check (and the combined-shift robustness
grid) to a FINISHED run directory, without rerunning the pipeline.

    .venv/bin/python scripts/add_alignment_qc.py --run out/ --spect <SPECT series dir> [--spect-series UID]

Reads out/derived/ (grid volumes, labels, simulated nulls) and out/results.json,
re-reads the SPECT only to rebuild its field-of-view mask (not saved by a run),
checks that the SPECT resampled now matches derived/spect_grid.nii.gz and that
the zero-shift P/A equals results.json, then writes `alignment_qc` and
`robustness.combined_shift_grid` into out/results.json (the previous file is
kept once as results.pre_alignment_qc.json). Figures are not regenerated.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from datscan_mri import alignment, params, pipeline, register, report, robustness
from datscan_mri.io import read_series


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--spect", required=True)
    ap.add_argument("--spect-series")
    a = ap.parse_args(argv)
    run = Path(a.run)
    d = run / "derived"
    ld = lambda n: np.transpose(np.asarray(nib.load(d / n).dataobj), (2, 1, 0))  # noqa: E731
    P = params.load(str(run / "params_used.json"))     # new keys (alignment_qc, ...) come from the defaults
    doc = json.load(open(run / "results.json"))
    grid = sitk.ReadImage(str(d / "mri_grid.nii.gz"))
    lab = ld("labels_grid.nii.gz").astype(np.uint8)
    sims = {k: ld(f"sim_{k}_grid.nii.gz").astype(np.float32) for k in ("uniform", "normal_gradient")}
    spect = read_series(a.spect, a.spect_series)
    sg = register.to_grid(spect, grid, None, sitk.sitkLinear, 0)
    diff = float(np.abs(sg - ld("spect_grid.nii.gz")).max())
    print(f"SPECT resampled now vs derived/spect_grid: max |diff| {diff:.3g}")
    if diff > 1e-3 * float(np.abs(sg).max()):
        sys.exit("SPECT does not match the run's spect_grid (wrong series or grid); stopping")
    fov = register.spect_fov_mask(spect, grid)
    SL = pipeline._crop(lab, P["analysis"]["crop_margin"])
    lc = lab[SL]
    low = {sd: doc["simulation"]["normal_gradient"]["expected_post_ant_range"][sd][0] for sd in "RL"}
    aq = alignment.check(sg, fov, SL, lc, {k: v[SL] for k, v in sims.items()}, pipeline._measured_main, low,
                         P["alignment_qc"])
    for sd in "RL":
        ref = doc["uncorrected"]["occ"][sd]["post_ant"]
        got = aq["as_registered"]["post_ant"][sd]
        print(f"zero-shift P/A {sd}: {got:.6f} (results.json {ref:.6f})")
        if abs(got - ref) > 1e-9:
            sys.exit("zero-shift P/A differs from results.json: the FOV mask or labels do not reproduce the run")
    R = P["robustness"]
    grid_res = robustness.combined_shift_grid(sg, fov, SL, lc, pipeline._measured_main,
                                              R["combined_shift_mm"], R["combined_shift_step_mm"])
    bak = run / "results.pre_alignment_qc.json"
    if not bak.exists():
        shutil.copy(run / "results.json", bak)
    doc["alignment_qc"] = aq
    doc.setdefault("robustness", {})["combined_shift_grid"] = grid_res
    if aq["shift_exceeds_limit"]:
        doc["warnings"] = [w for w in doc.get("warnings", []) if not w.startswith("SPECT-to-anatomy")] + [aq["warning"]]
        print("!" * 78 + "\nWARNING: " + aq["warning"] + "\n" + "!" * 78)
    json.dump(report._clean(doc), open(run / "results.json", "w"), indent=1)
    for f in aq["fits"]:
        print(f"{f['target']:16s} {f['mask']:27s} shift (x,y,z) mm {tuple(f['best_shift_mm_xyz'])} "
              f"r {f['correlation']:.3f} (0-shift {f['correlation_at_zero_shift']:.3f})  P/A R "
              f"{f['post_ant']['R']:.3f} L {f['post_ant']['L']:.3f}  gap R {f['gap_vs_normal_gradient_low']['R']:+.3f} "
              f"L {f['gap_vs_normal_gradient_low']['L']:+.3f}")
    print("combined-shift grid range:", {k: [round(v, 3) for v in r] for k, r in grid_res["range"].items()})
    print(f"wrote alignment_qc into {run / 'results.json'}")


if __name__ == "__main__":
    main()
