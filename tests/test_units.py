import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, NuclearMedicineImageStorage, generate_uid

from datscan_mri import profile as prof
from datscan_mri import pvc, simulate
from datscan_mri.io import read_series, write_clean_nifti


def test_gtm_recovers_true_values_noise_free():
    lab = np.zeros((40, 40, 40), np.uint8)
    lab[5:35, 5:35, 5:35] = 9
    lab[15:25, 12:20, 15:22] = 3
    lab[15:25, 20:28, 15:22] = 5
    truth = {9: 1.0, 3: 4.0, 5: 2.0}
    act = np.zeros(lab.shape, np.float32)
    for k, v in truth.items():
        act[lab == k] = v
    img = simulate.blur(act, 10.0)
    valid = np.ones(lab.shape, bool)
    ids = [9, 3, 5]
    obs = prof.voi_means(img, lab, valid, ids)
    T = pvc.gtm(obs, pvc.transfer_matrix(lab, ids, 10.0, valid))
    assert np.allclose(T, [truth[i] for i in ids], rtol=1e-3)
    assert obs[1] < 0.8 * truth[3]  # uncorrected is partial-volume diluted


def _nm_slice(path, z, k, sbs):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "NM"
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.StudyInstanceUID = "1.2.3.4"
    ds.FrameOfReferenceUID = "1.2.3.4.6"
    ds.InstanceNumber = k + 1
    ds.Rows = ds.Columns = 8
    ds.PixelSpacing = [3.9, 3.9]
    ds.SliceThickness = sbs
    ds.SpacingBetweenSlices = sbs
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.full((8, 8), k, np.uint16)).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_nm_series_spacing_from_tag(tmp_path):
    sbs = 3.8953697681427
    for k in range(6):
        _nm_slice(tmp_path / f"s{k}.dcm", k * 3.89527, k, sbs)
    im = read_series(tmp_path)
    assert im.GetSize() == (8, 8, 6)
    assert abs(im.GetSpacing()[2] - sbs) < 1e-9


def test_clean_nifti_has_no_free_text(tmp_path):
    import nibabel as nib
    p = tmp_path / "x.nii.gz"
    write_clean_nifti(np.zeros((4, 5, 6), np.float32), np.eye(4), p)
    h = nib.load(p).header
    assert bytes(h["descrip"]).strip(b"\0") == b""
    assert bytes(h["aux_file"]).strip(b"\0") == b""
    assert nib.load(p).shape == (6, 5, 4)
    assert len(h.extensions) == 0
