"""Generate the synthetic head phantom and run the full pipeline on it.

    python examples/synthetic_demo.py [uniform|posterior_loss|global_loss] [--fast]

Writes phantom NIfTIs and the pipeline output to examples/out/<scenario>/
(git-ignored). No patient data involved.
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from datscan_mri import cli, phantom

scenario = next((a for a in sys.argv[1:] if not a.startswith("-")), "posterior_loss")
fast = "--fast" in sys.argv
out = Path(__file__).parent / "out" / scenario
inp = out / "inputs"
inp.mkdir(parents=True, exist_ok=True)

print(f"building phantom ({scenario})...")
anat, mri, ct, motion = phantom.build(seed=0)
spect = phantom.make_spect(anat, scenario, np.random.default_rng(1))
for name, im in (("mri", mri), ("ct", ct), ("spect", spect)):
    sitk.WriteImage(im, str(inp / f"{name}.nii.gz"))

ac = ",".join(f"{v:g}" for v in phantom.ac_lps())
args = ["run", "--spect", str(inp / "spect.nii.gz"), "--ct", str(inp / "ct.nii.gz"),
        "--mri", str(inp / "mri.nii.gz"), "--out", str(out / "result"), "--ac", ac,
        "--viewer-out", str(out / "viewer")]
if fast:
    args.append("--skip-robustness")
cli.main(args)
print(f"\nTrue posterior/anterior SBR ratio in this phantom: "
      f"{ {'uniform': 1.0, 'posterior_loss': 0.35, 'global_loss': 1.0}[scenario] }")
print(f"Figures: {out / 'result' / 'figures'}")
