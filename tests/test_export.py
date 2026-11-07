import json

import nibabel as nib
import numpy as np

from datscan_mri.export import export_viewer


def test_export_full_fov_clean_headers(phantom_case, tmp_path):
    A = phantom_case["A"]
    sg = np.ones(A.mg.shape, np.float32)
    g = export_viewer(A.mg, A.ct, sg, A.lab, A.ic, A.ac, A.grid.GetDirection(), tmp_path)
    assert g["full_fov"] is True
    shape_xyz = A.mg.shape[::-1]
    for f, dt in (("mri_t1", np.int16), ("ct", np.int16), ("spect", np.float32), ("labels", np.uint8)):
        im = nib.load(tmp_path / f"{f}.nii.gz")
        assert im.shape == shape_xyz
        assert im.get_data_dtype() == dt
        h = im.header
        assert bytes(h["descrip"]).strip(b"\0") == b""
        assert bytes(h["aux_file"]).strip(b"\0") == b""
        assert len(h.extensions) == 0
        # AC maps to world (0, 0, 0)
        ijk = np.linalg.inv(im.affine) @ [0, 0, 0, 1]
        assert np.allclose(ijk[:3], [A.ac[2], A.ac[1], A.ac[0]])
    text = json.dumps(g)
    for frag in ("/Users", "/home", "/private", "/tmp", "C:\\"):
        assert frag not in text  # no local paths in the metadata
