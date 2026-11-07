"""External-segmenter integration without the external tools: a FreeSurfer-LUT
label file made from the phantom's TRUE putamen/caudate is loaded through the
same path SynthSeg / FastSurfer output takes (resample to grid, LUT mapping,
side check, volume gate), and the quality gates are exercised."""
import numpy as np
import pytest
import SimpleITK as sitk

from datscan_mri import external, params, phantom, pipeline
from datscan_mri.pipeline import QCError

from conftest import AC_OFFSET, QUIET


def _fs_file(truth, path, swap=False, shrink_put=False):
    fs = np.zeros(truth["put"]["R"].shape, np.int16)
    R, L = (51, 12) if not swap else (12, 51)
    fs[truth["caud"]["R"]] = 50 if not swap else 11
    fs[truth["caud"]["L"]] = 11 if not swap else 50
    for side, lab in (("R", R), ("L", L)):
        m = truth["put"][side]
        if shrink_put:
            idx = np.argwhere(m)[:500]
            m = np.zeros_like(m)
            m[tuple(idx.T)] = True
        fs[m] = lab
    fs[truth["vent"]] = 4                      # a label we must ignore
    im = phantom._vol(fs)
    sitk.WriteImage(sitk.Cast(im, sitk.sitkInt16), str(path))
    return path


def _params(path):
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    P["segmenter"].update(method="labels_file", labels_file=str(path),
                          labels_file_info={"tool": "phantom truth", "version": "test"})
    return P


def test_lut_mapping_and_side_check():
    fs = np.zeros((4, 4, 10), np.int16)
    fs[1, 1, 1], fs[1, 1, 2], fs[1, 1, 8], fs[1, 1, 7], fs[1, 1, 5] = 51, 50, 12, 11, 26
    seg = external.striatum_from_fs(fs)
    assert seg[1, 1, 1] == 3 and seg[1, 1, 2] == 1 and seg[1, 1, 8] == 4 and seg[1, 1, 7] == 2
    assert seg[1, 1, 5] == 0, "accumbens must not enter the striatal VOIs"
    external.check_sides(seg, ac_x=5)
    with pytest.raises(external.SegmenterError, match="wrong side"):
        external.check_sides(seg[:, :, ::-1], ac_x=5)


def test_labels_file_gives_truth_vois(phantom_case, tmp_path):
    t = phantom_case["truth"]
    P = _params(_fs_file(t, tmp_path / "truth_fs.nii.gz"))
    A = pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"],
                                 workdir=tmp_path, **QUIET)
    for side, codes, c in (("R", (3, 5), 3), ("L", (4, 6), 4)):
        assert np.array_equal(A.seg == c, t["put"][side])
        v = np.isin(A.lab, codes)
        assert np.array_equal(v, t["put"][side])
    assert A.seg_info["tool"] == "phantom truth" and "labels_file_sha256" in A.seg_info
    d = A.seg_info["dice_vs"]["threshold"]          # threshold method run as the comparison
    assert 0.6 < d["putamen_R"] < 1 and 0.6 < d["putamen_L"] < 1
    assert "threshold" in A.alt_segs


def test_labels_file_gates(phantom_case, tmp_path):
    t = phantom_case["truth"]
    for kw, msg in ((dict(swap=True), "wrong side"), (dict(shrink_put=True), "plausible")):
        P = _params(_fs_file(t, tmp_path / f"bad_{list(kw)[0]}.nii.gz", **kw))
        with pytest.raises(QCError, match=msg):
            pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"],
                                     **QUIET)
    P = _params(tmp_path / "missing.nii.gz")
    with pytest.raises(QCError, match="not found"):
        pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"], **QUIET)


def test_missing_tool_is_a_loud_failure(phantom_case, monkeypatch, tmp_path):
    monkeypatch.delenv("DATSCAN_SYNTHSEG_HOME", raising=False)
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    P["segmenter"]["method"] = "synthseg"
    P["segmenter"]["synthseg"]["home"] = str(tmp_path / "nowhere")
    with pytest.raises(QCError, match="synthseg"):
        pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"],
                                 workdir=tmp_path, **QUIET)


def test_synthseg_qc_gate(phantom_case, monkeypatch, tmp_path):
    """A low SynthSeg QC score stops the run (tool replaced by a stub)."""
    path = _fs_file(phantom_case["truth"], tmp_path / "truth_fs.nii.gz")

    def fake(mri_path, workdir, cfg, log=print):
        return path, {"tool": "SynthSeg", "version": "stub", "qc_scores": {"putamen+pallidum": 0.41, "csf": 0.9}}
    monkeypatch.setitem(external.TOOLS, "synthseg", fake)
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    P["segmenter"]["method"] = "synthseg"
    with pytest.raises(QCError, match="QC scores below"):
        pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"],
                                 workdir=tmp_path, **QUIET)
