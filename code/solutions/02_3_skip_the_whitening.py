"""Standalone solution; also %load-able from the companion notebook."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    _SRC = Path(__file__).resolve().parents[1] / "src"
except NameError:
    _cwd = Path.cwd().resolve()
    if (_cwd / "src" / "brainsmodels").is_dir():
        _SRC = _cwd / "src"
    elif (_cwd.parent / "src" / "brainsmodels").is_dir():
        _SRC = _cwd.parent / "src"
    else:
        _SRC = _cwd / "code" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from brainsmodels.repro import DEFAULT_SEED, set_seed

SEED = set_seed(DEFAULT_SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
data = PatchDataset.from_pink_noise(n_train=500, n_test=16, n_images=8, seed=SEED)
pix = HierarchicalPredictiveCoding(n_input=data.train_pixel.shape[1], seed=SEED)
zca = HierarchicalPredictiveCoding(n_input=data.train.shape[1], seed=SEED)
pix.train(data.train_pixel, n_epochs=6, n_infer=20, seed=SEED)
zca.train(data.train, n_epochs=6, n_infer=20, seed=SEED)

rf_pix = pix.U1
rf_zca = data.whitening.to_pixel_basis(zca.U1)
energy = rf_zca.reshape(data.patch_size, data.patch_size, -1).var(axis=(0, 1))
order = np.argsort(energy)[::-1][:8]

fig, axes = plt.subplots(2, 8, figsize=(9.5, 2.8))
for col, i in enumerate(order):
    for row, bank, name in ((0, rf_pix, "pixels"), (1, rf_zca, "ZCA→pixels")):
        patch = bank[:, i].reshape(data.patch_size, data.patch_size)
        vmax = np.max(np.abs(patch)) + 1e-8
        axes[row, col].imshow(patch, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=8)
fig.suptitle("Whitening changes the coordinates the Hebbian rule sees")
plt.tight_layout()
plt.show()
print(
    "Without whitening the columns spend their variance on low-frequency "
    "blobs (the 1/f spectrum). ZCA equalizes those coordinates so a local "
    "rule can spend capacity on oriented structure that is already in the "
    "gratings. Whitening is a change of basis, not a retina, and the "
    "dictionary is still not V1."
)
