"""Read the bundled, checksum-pinned BrainWeb normal-phantom data.

The paired NIfTI files are included in this package; no network is used.
They retain BrainWeb's original terms and attribution, not this code's MIT
licence. No patient images, fitted tumour parameters or model weights are read.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np

COMMIT = "81ecd42e6f37692a711208caa8d24361904f7e65"
BUNDLED = Path(__file__).resolve().parent / "data/brainweb"
# COMMIT records input provenance only; no external repository is accessed.
FILES = {
    "t1_icbm_normal_1mm_pn0_rf0.nii.gz":
        ("aa40262b2702653188fd51bdec844a393b825650fe1155a84d8606f5dfd99eb3", 8299415),
    "phantom_1.0mm_normal_crisp.nii.gz":
        ("afdff78fc1b06ad7c7d8216cf96be737a1450ef5b6abf27eb732c0bf8b66fac0", 947188),
}


def verify_files(directory):
    directory = Path(directory)
    for name, (digest, size) in FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing BrainWeb file: {path}")
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"BrainWeb checksum/size mismatch: {path}")
    return directory



@dataclass
class Anatomy:
    t1: np.ndarray
    labels: np.ndarray
    affine: np.ndarray
    axes: tuple
    spacing_mm: tuple

    @property
    def domain(self):
        return np.isin(self.labels, [2, 3, 8])


def load_brainweb(directory=None, *, step_mm=4):
    """Canonical RAS+ integer-stride sampling from matched 1-mm T1 and labels.

    This is explicit nearest-sample decimation, not interpolation or registration.
    Thin tissue barriers can disappear on a coarse grid. T1 is display context;
    it is not the input to parameter fitting. No synthetic fallback is used.
    A caller-specified BRAINSMODELS_BRAINWEB_DIR is read without writing to it.
    """
    import nibabel as nib

    if not isinstance(step_mm, int) or step_mm not in (1, 2, 3, 4, 6):
        raise ValueError("Choose an explicit integer spacing: 1, 2, 3, 4 or 6 mm.")
    specified = os.environ.get("BRAINSMODELS_BRAINWEB_DIR")
    directory = verify_files(directory if directory is not None else specified or BUNDLED)
    images = [nib.as_closest_canonical(nib.load(directory / name)) for name in FILES]
    t1, labels = images
    expected_affine = np.diag([1., 1., 1., 1.])
    expected_affine[:3, 3] = [-90., -126., -72.]
    if (t1.shape != (181, 217, 181) or labels.shape != t1.shape
            or not np.allclose(t1.affine, labels.affine, atol=1e-5)
            or not np.allclose(t1.affine, expected_affine, atol=1e-5)):
        raise ValueError("Expected aligned native 1-mm BrainWeb volumes in canonical RAS+.")
    stride = (slice(None, None, step_mm),) * 3
    label_data = np.asarray(labels.dataobj)[stride].copy()
    if not np.isfinite(label_data).all() or not np.equal(label_data, np.rint(label_data)).all():
        raise ValueError("Tissue labels must remain finite integer categories.")
    image = np.asarray(t1.dataobj, dtype=float)[stride].copy()
    image /= max(float(np.percentile(image, 99.8)), 1e-8)
    affine = expected_affine.copy()
    affine[:3, :3] *= step_mm
    axes = tuple(affine[i, 3] + step_mm*np.arange(n) for i, n in enumerate(image.shape))
    return Anatomy(image, label_data.astype(np.uint8), affine, axes, (step_mm,)*3)
