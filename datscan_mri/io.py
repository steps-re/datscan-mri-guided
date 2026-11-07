"""DICOM -> image conversion, and de-identified NIfTI writing.

Images are handled as SimpleITK images in DICOM patient coordinates (LPS, mm).
Arrays taken from them are indexed [z, y, x]; x increases toward patient LEFT,
y toward POSTERIOR, z toward SUPERIOR.

Nothing here reads or writes patient identifiers. `inventory` prints only
technical series fields (modality, description, matrix, file count).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def _dicom_files(d: Path) -> list[Path]:
    out = []
    for f in sorted(Path(d).iterdir()):
        if not f.is_file() or f.name.upper() in {"VERSION", "DICOMDIR", "LOCKFILE"}:
            continue
        try:
            pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["Modality"])
        except Exception:
            continue
        out.append(f)
    return out


def inventory(root: str | os.PathLike) -> list[dict]:
    """List DICOM series under `root` (technical fields only)."""
    rows = []
    for dirpath, _dirs, _files in os.walk(root):
        fs = _dicom_files(Path(dirpath))
        if not fs:
            continue
        ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
        rows.append(
            dict(
                path=dirpath,
                n_files=len(fs),
                modality=str(ds.get("Modality", "")),
                description=str(ds.get("SeriesDescription", "")),
                rows=ds.get("Rows"),
                cols=ds.get("Columns"),
                frames=ds.get("NumberOfFrames"),
                pixel_spacing=[float(v) for v in ds.get("PixelSpacing", [])] or None,
                slice_thickness=float(ds.get("SliceThickness", 0) or 0) or None,
            )
        )
    return rows


class SeriesSelectionError(ValueError):
    """A folder holds several DICOM series and none was chosen."""


def read_series(path: str | os.PathLike, series_uid: str | None = None) -> sitk.Image:
    """Read one image volume as float32.

    `path` may be a NIfTI/MetaImage file or a directory holding DICOM series.
    If the directory holds more than one series, `series_uid` must name one;
    otherwise SeriesSelectionError lists them (UID, file count, description).
    For nuclear-medicine (NM) slice stacks the slice spacing is taken from the
    DICOM SpacingBetweenSlices tag when present (GDCM's position-derived value
    can differ in the 4th decimal).
    """
    p = Path(path)
    if p.is_file():
        return sitk.Cast(sitk.ReadImage(str(p)), sitk.sitkFloat32)
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(p))
    if not ids:
        raise FileNotFoundError(f"no DICOM series found in {p}")
    files_by_id = {i: reader.GetGDCMSeriesFileNames(str(p), i) for i in ids}
    if series_uid is not None:
        if series_uid not in files_by_id:
            raise SeriesSelectionError(f"series {series_uid} not found in {p}; found: {sorted(files_by_id)}")
        sid = series_uid
    elif len(files_by_id) == 1:
        sid = ids[0]
    else:
        lines = []
        for i, fs in files_by_id.items():
            d = pydicom.dcmread(fs[0], stop_before_pixels=True, specific_tags=["SeriesDescription", "Modality"])
            lines.append(f"  {i}  {len(fs):4d} files  {d.get('Modality', '')} {d.get('SeriesDescription', '')}")
        raise SeriesSelectionError(f"{p} holds {len(ids)} DICOM series; choose one with --<modality>-series:\n"
                                   + "\n".join(lines))
    files = files_by_id[sid]
    if len(files) == 1:
        im = sitk.ReadImage(files[0])
    else:
        reader.SetFileNames(files)
        im = reader.Execute()
    im = sitk.Cast(im, sitk.sitkFloat32)
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    if str(ds.get("Modality", "")) == "NM" and im.GetDimension() == 3:
        sbs = ds.get("SpacingBetweenSlices")
        if sbs is not None and float(sbs) > 0:
            sp = list(im.GetSpacing())
            sp[2] = float(sbs)
            im.SetSpacing(sp)
    return im


def sitk_to_ras_affine(im: sitk.Image) -> np.ndarray:
    """4x4 RAS affine for an image whose array is transposed to [x, y, z]."""
    D = np.array(im.GetDirection()).reshape(3, 3)
    S = np.diag(im.GetSpacing())
    A = np.eye(4)
    A[:3, :3] = D @ S
    A[:3, 3] = im.GetOrigin()
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS -> RAS
    return flip @ A


def write_clean_nifti(arr_zyx: np.ndarray, affine_ras: np.ndarray, path: str | os.PathLike) -> None:
    """Write a NIfTI with a fresh header: geometry only, no description, no
    intent name, no auxiliary file, no extensions. `arr_zyx` is [z, y, x]."""
    import nibabel as nib

    data = np.ascontiguousarray(np.transpose(arr_zyx, (2, 1, 0)))
    img = nib.Nifti1Image(data, affine_ras)  # brand-new header
    hdr = img.header
    hdr["descrip"] = b""
    hdr["aux_file"] = b""
    hdr["intent_name"] = b""
    hdr["db_name"] = b""
    img.set_qform(affine_ras, code=1)
    img.set_sform(affine_ras, code=1)
    nib.save(img, str(path))


def write_image(im: sitk.Image, path: str | os.PathLike) -> None:
    """Write a SimpleITK image as a clean NIfTI (geometry only)."""
    write_clean_nifti(sitk.GetArrayFromImage(im), sitk_to_ras_affine(im), path)
