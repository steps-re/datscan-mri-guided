"""Orientation invariance and fail-loud quality gates."""
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from datscan_mri import params, phantom, pipeline, register
from datscan_mri.io import SeriesSelectionError, read_series
from datscan_mri.pipeline import QCError

from conftest import AC_OFFSET, QUIET


def _pa(A, spect, P):
    m = pipeline.quick_metrics(A, spect, P)["occ"]
    return np.array([m["R"]["post_ant"], m["L"]["post_ant"]])


def test_grid_is_lps_identity_for_flipped_ct(phantom_case):
    ct = phantom_case["ct"]
    f = phantom.flip_y(ct)
    assert f.GetDirection()[4] == -1.0
    g0, g1 = register.make_grid(ct), register.make_grid(f)
    assert g1.GetDirection() == register.IDENTITY
    assert g0.GetSize() == g1.GetSize()
    assert np.allclose(g0.GetOrigin(), g1.GetOrigin(), atol=1e-6)


def test_flipped_ct_gives_same_ratios(phantom_case):
    """The same physical CT stored with its y axis reversed must give the same
    analysis (the old code copied the CT direction and read anterior as posterior)."""
    P, A0 = phantom_case["P"], phantom_case["A"]
    ctf = phantom.flip_y(phantom_case["ct"])
    A1 = pipeline.prepare_anatomy(ctf, phantom_case["mri"], P, transform=A0.transform, **QUIET)
    assert A1.ac == A0.ac
    for code in (3, 4, 5, 6):
        assert abs((A1.lab == code).sum() - (A0.lab == code).sum()) <= 0.02 * (A0.lab == code).sum()
    sp = phantom.make_spect(phantom_case["truth"], "normal_gradient", np.random.default_rng(1))
    assert np.allclose(_pa(A0, sp, P), _pa(A1, sp, P), atol=0.01)
    # posterior VOI really is posterior (larger LPS y) on the flipped run
    for a, p in ((3, 5), (4, 6)):
        ya = A1.grid.TransformContinuousIndexToPhysicalPoint(np.argwhere(A1.lab == a).mean(0)[::-1].tolist())[1]
        yp = A1.grid.TransformContinuousIndexToPhysicalPoint(np.argwhere(A1.lab == p).mean(0)[::-1].tolist())[1]
        assert yp > ya


def test_qc_estimated_ac_refused(phantom_case):
    P = params.load(None)
    with pytest.raises(QCError, match="AC"):
        pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"], **QUIET)


def test_qc_collapsed_segmentation_refused():
    """Seed 6 draws a putamen darker than the 'putamen.lo' threshold after
    intensity normalisation; the segmentation collapses and must be refused."""
    anat, mri, ct, motion = phantom.build(seed=6)
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    with pytest.raises(QCError, match="putamen VOI"):
        pipeline.prepare_anatomy(ct, mri, P, transform=motion, **QUIET)


def test_qc_low_ventricle_dice_refused(phantom_case):
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    bad = sitk.Euler3DTransform(phantom_case["motion"])
    par = list(bad.GetParameters())
    par[3] += 10.0                      # 10-mm left-right and 8-mm vertical registration error
    par[5] += 8.0
    bad.SetParameters(par)
    with pytest.raises(QCError, match="Dice"):
        pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=bad, **QUIET)


def test_qc_vois_outside_spect_fov_refused(phantom_case):
    sp = phantom.make_spect(phantom_case["truth"], "uniform", np.random.default_rng(1))
    a = sitk.GetArrayFromImage(sp)
    a[:, a.shape[1] // 2 - 2:, :] = 0   # reconstruction support ends mid-brain (cuts the posterior putamen)
    cut = sitk.GetImageFromArray(a)
    cut.CopyInformation(sp)
    with pytest.raises(QCError, match="outside the SPECT"):
        pipeline.analyze(phantom_case["A"], cut, phantom_case["P"], do_robustness=False, do_sweeps=False, **QUIET)


def test_fov_mask_excludes_zero_padding():
    a = np.zeros((6, 20, 20), np.float32)
    a[1:5, 4:16, 4:16] = 5.0
    a[2, 9, 9] = 0.0                    # an interior zero is NOT padding
    im = sitk.GetImageFromArray(a)
    m = register.spect_fov_mask(im, im)
    assert m[2, 9, 9] and m[1, 10, 10]
    assert not m[0].any() and not m[2, 1, 1]


def _ct_slice(path, uid, k, desc):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality, ds.SeriesDescription = "CT", desc
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.FrameOfReferenceUID = uid, "1.2.3", "1.2.3.9"
    ds.InstanceNumber = k + 1
    ds.Rows = ds.Columns = 4
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, float(k)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = np.zeros((4, 4), np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_series_selection_is_explicit(tmp_path):
    for k in range(3):
        _ct_slice(tmp_path / f"a{k}.dcm", "1.2.3.4.1", k, "big")
    for k in range(2):
        _ct_slice(tmp_path / f"b{k}.dcm", "1.2.3.4.2", k, "small")
    with pytest.raises(SeriesSelectionError, match="1.2.3.4.1"):
        read_series(tmp_path)
    assert read_series(tmp_path, "1.2.3.4.2").GetSize()[2] == 2
    assert pydicom  # imported for the helper
