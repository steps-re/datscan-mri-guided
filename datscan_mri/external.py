"""Optional external anatomical segmenters (SynthSeg, FastSurfer).

The package never imports these tools. They run as subprocesses from their
own environments (the configured `python`), on the registered, N4-corrected
MRI written onto the analysis grid, and their FreeSurfer-LUT label map is
resampled (nearest neighbour) back onto that grid. Only four labels are used:

    FreeSurfer LUT        ->  our code (segment.LABELS)
    50 Right-Caudate      ->  1 caudR
    11 Left-Caudate       ->  2 caudL
    51 Right-Putamen      ->  3 right putamen (split at the AC into 3 / 5)
    12 Left-Putamen       ->  4 left putamen  (split at the AC into 4 / 6)

Nucleus accumbens (26 / 58) and pallidum (13 / 52) are not included.

A precomputed FreeSurfer-LUT label file (any space with a correct affine)
can be used instead (`segmenter.method = "labels_file"`), which is how the
tests run without either tool installed.

Tool locations come from `segmenter.<tool>.home` or the environment
(`DATSCAN_SYNTHSEG_HOME`, `DATSCAN_FASTSURFER_HOME`). Expected layouts:

* SynthSeg: FreeSurfer 7.4.1 `mri_synthseg` (unmodified) with its label
  lists and hash-checked weights in `<home>/models/`, a launcher
  `<home>/run_synthseg.py` and a venv `<home>/.venv` (tensorflow 2.15, surfa).
* FastSurfer: a checkout of Deep-MI/FastSurfer at a release tag with the
  VINN checkpoints in `<home>/checkpoints/` and a venv `<home>/.venv`.
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FS_TO_OURS = {50: 1, 11: 2, 51: 3, 12: 4}
FS_NAMES = {50: "Right-Caudate", 11: "Left-Caudate", 51: "Right-Putamen", 12: "Left-Putamen"}


class SegmenterError(RuntimeError):
    """The external segmenter is missing, failed, or produced unusable output."""


def striatum_from_fs(fs: np.ndarray) -> np.ndarray:
    """FreeSurfer-LUT label map -> uint8 map with 1/2 caudate R/L, 3/4 putamen R/L."""
    out = np.zeros(fs.shape, np.uint8)
    for k, v in FS_TO_OURS.items():
        out[fs == k] = v
    return out


def check_sides(seg: np.ndarray, ac_x: int) -> None:
    """Right structures must lie at smaller x index (patient right in LPS)
    than the AC and left ones at larger: catches a left/right swap from a
    wrong affine anywhere in the external chain."""
    for code, sign, name in ((1, -1, "right caudate"), (2, +1, "left caudate"),
                             (3, -1, "right putamen"), (4, +1, "left putamen")):
        idx = np.argwhere(seg == code)
        if idx.size == 0:
            raise SegmenterError(f"external segmentation has no {name}")
        if np.sign(idx[:, 2].mean() - ac_x) != sign:
            raise SegmenterError(f"{name} lies on the wrong side of the midline: left/right swapped "
                                 "(check the label file's affine)")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resample_to_grid(path, grid):
    """Read a label image (NIfTI/MGZ via SimpleITK or nibabel) and resample it
    onto `grid` (nearest neighbour, physical coordinates). Returns [z, y, x]."""
    import SimpleITK as sitk
    p = str(path)
    if p.endswith(".mgz") or p.endswith(".mgh"):
        import nibabel as nib
        im = nib.load(p)
        with tempfile.TemporaryDirectory() as td:
            tmp = str(Path(td) / "labels.nii.gz")
            nib.save(nib.Nifti1Image(np.asarray(im.dataobj).astype(np.int16), im.affine), tmp)
            src = sitk.ReadImage(tmp)
    else:
        src = sitk.ReadImage(p)
    if src.GetNumberOfComponentsPerPixel() != 1:
        raise SegmenterError(f"{path}: expected a single-channel label map")
    out = sitk.Resample(src, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
    return sitk.GetArrayFromImage(out).astype(np.int16)


def _home(cfg, tool):
    h = os.environ.get(f"DATSCAN_{tool.upper()}_HOME") or cfg.get("home")
    if not h:
        raise SegmenterError(f"{tool} not configured: set segmenter.{tool}.home or DATSCAN_{tool.upper()}_HOME")
    if h.startswith("<") and h.endswith(">"):   # placeholder left in a shared config
        raise SegmenterError(f"{tool} home is the placeholder {h}: set DATSCAN_{tool.upper()}_HOME (or "
                             f"segmenter.{tool}.home) to the install folder")
    h = Path(os.path.expanduser(h))
    if not h.is_dir():
        raise SegmenterError(f"{tool} home {h} does not exist")
    return h


def _run(cmd, log, cwd=None, env=None):
    log("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SegmenterError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed (exit {r.returncode}):\n"
                             + (r.stderr or r.stdout)[-3000:])
    return r.stdout


def _read_csv_row(path):
    rows = list(csv.reader(open(path)))
    return {h.strip(): v.strip() for h, v in zip(rows[0], rows[1])}


def synthseg(mri_path, workdir, cfg, log=print):
    """Run FreeSurfer 7.4.1 mri_synthseg. Returns (label file, info)."""
    home = _home(cfg, "synthseg")
    py = Path(cfg.get("python") or home / ".venv" / "bin" / "python")
    launcher = home / "run_synthseg.py"
    robust = bool(cfg.get("robust", False))
    model = home / "models" / ("synthseg_robust_2.0.h5" if robust else "synthseg_2.0.h5")
    for f in (py, launcher, model, home / "models" / "mri_synthseg"):
        if not f.exists():
            raise SegmenterError(f"SynthSeg: {f} not found")
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    seg, qc, vol = w / "synthseg_labels.nii.gz", w / "synthseg_qc.csv", w / "synthseg_volumes.csv"
    cmd = [py, "-I", launcher, "--i", mri_path, "--o", seg, "--qc", qc, "--vol", vol,
           "--threads", int(cfg.get("threads", 8)), "--cpu"] + (["--robust"] if robust else [])
    env = dict(os.environ, FREESURFER_HOME=str(home))
    _run(cmd, log, env=env)
    q = {k: float(v) for k, v in _read_csv_row(qc).items() if k != "subject"}
    v = {k: float(v) / 1000 for k, v in _read_csv_row(vol).items() if k != "subject"}
    info = {"tool": "SynthSeg", "version": "SynthSeg-robust 2.0" if robust else "SynthSeg 2.0",
            "distribution": cfg.get("distribution", "FreeSurfer 7.4.1 mri_synthseg (freesurfer/freesurfer tag v7.4.1)"),
            "model_file": model.name, "model_sha256": _sha256(model),
            "script_sha256": _sha256(home / "models" / "mri_synthseg"),
            "reference": "Billot B, et al. Robust machine learning segmentation for large-scale analysis of "
                         "heterogeneous clinical brain MRI datasets. PNAS 2023;120:e2216399120",
            "qc_scores": q, "tool_volumes_ml": {k: v[k] for k in v if any(s in k for s in ("putamen", "caudate"))}}
    return seg, info


def fastsurfer(mri_path, workdir, cfg, log=print):
    """Run FastSurferCNN (VINN) segmentation only. Returns (label file, info)."""
    home = _home(cfg, "fastsurfer")
    py = Path(cfg.get("python") or home / ".venv" / "bin" / "python")
    script = home / "FastSurferCNN" / "run_prediction.py"
    ck = {v: home / "checkpoints" / f"aparc_vinn_{v}_v2.0.0.pkl" for v in ("axial", "coronal", "sagittal")}
    for f in (py, script, *ck.values()):
        if not f.exists():
            raise SegmenterError(f"FastSurfer: {f} not found")
    w = Path(workdir) / "fastsurfer"
    (w / "s" / "mri").mkdir(parents=True, exist_ok=True)
    m = w / "s" / "mri"
    seg = m / "aparc.DKTatlas+aseg.deep.mgz"
    cmd = [py, script, "--t1", mri_path, "--sid", "s", "--sd", w, "--asegdkt_segfile", seg,
           "--conformed_name", m / "orig.mgz", "--brainmask_name", m / "mask.mgz",
           "--aseg_name", m / "aseg.auto_noCCseg.mgz", "--seg_log", w / "seg.log",
           "--ckpt_ax", ck["axial"], "--ckpt_cor", ck["coronal"], "--ckpt_sag", ck["sagittal"],
           "--device", cfg.get("device", "cpu"), "--viewagg_device", "cpu", "--threads", int(cfg.get("threads", 8))]
    env = dict(os.environ, PYTHONPATH=str(home))
    _run(cmd, log, cwd=home, env=env)
    ver = "unknown"
    try:
        ver = subprocess.run(["git", "-C", str(home), "describe", "--tags"], capture_output=True,
                             text=True).stdout.strip() or ver
    except OSError:
        pass
    info = {"tool": "FastSurfer", "version": f"FastSurferCNN (VINN) {ver}",
            "model_file": [p.name for p in ck.values()],
            "model_sha256": {p.name: _sha256(p) for p in ck.values()},
            "reference": "Henschel L, et al. FastSurfer - a fast and accurate deep learning based neuroimaging "
                         "pipeline. NeuroImage 2020;219:117012",
            "qc_scores": None}
    return seg, info


TOOLS = {"synthseg": synthseg, "fastsurfer": fastsurfer}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def striatal_dice(seg_a: np.ndarray, seg_b: np.ndarray) -> dict:
    """Dice of putamen (codes 3/4, or 3/5 + 4/6 after the split) and caudate per side."""
    def put(s, side):
        return np.isin(s, (3, 5) if side == "R" else (4, 6))
    return {"putamen_R": dice(put(seg_a, "R"), put(seg_b, "R")), "putamen_L": dice(put(seg_a, "L"), put(seg_b, "L")),
            "caudate_R": dice(seg_a == 1, seg_b == 1), "caudate_L": dice(seg_a == 2, seg_b == 2)}


def segment_striatum(mg, grid, ac, S, workdir, log=print):
    """Run (or load) the configured external segmenter. Returns
    (seg uint8 with codes 1-4 on the grid, FreeSurfer labels on the grid, info)."""
    from .io import sitk_to_ras_affine, write_clean_nifti
    method = S["method"]
    if method == "labels_file":
        path = S.get("labels_file")
        if not path or not Path(path).exists():
            raise SegmenterError(f"segmenter.labels_file {path!r} not found")
        info = dict(S.get("labels_file_info") or {"tool": "precomputed FreeSurfer-LUT label file"})
        info["labels_file_sha256"] = _sha256(path)
        lab_path = path
    elif method in TOOLS:
        w = Path(workdir)
        w.mkdir(parents=True, exist_ok=True)
        mri_path = w / "mri_grid_for_segmenter.nii.gz"
        write_clean_nifti(mg.astype(np.float32), sitk_to_ras_affine(grid), mri_path)
        log(f"external segmenter: {method}...")
        lab_path, info = TOOLS[method](mri_path, w, S.get(method, {}), log)
    else:
        raise SegmenterError(f"unknown segmenter.method {method!r}")
    fs = resample_to_grid(lab_path, grid)
    seg = striatum_from_fs(fs)
    check_sides(seg, ac[2])
    info["method"] = method
    info["label_map"] = {FS_NAMES[k]: f"{k} -> {v}" for k, v in FS_TO_OURS.items()}
    info["volumes_ml_on_grid"] = {n: float((fs == k).sum() / 1000) for k, n in FS_NAMES.items()}
    return seg, fs, info
