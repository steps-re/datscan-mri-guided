"""Launcher for FreeSurfer 7.4.1 mri_synthseg (unmodified) on Apple silicon.

numpy on macOS arm64 has no np.float128 (the script only uses it in an
isinstance tuple), so alias it to np.longdouble, set FREESURFER_HOME to this
folder (models/ holds the script, label lists and hash-checked weights) and
run the script unchanged."""
import os
import runpy
import sys

import numpy as np

if not hasattr(np, "float128"):
    np.float128 = np.longdouble
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FREESURFER_HOME", HERE)
script = os.path.join(HERE, "models", "mri_synthseg")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
