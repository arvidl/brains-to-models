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
data = PatchDataset.from_pink_noise(n_train=500, n_test=24, n_images=8, seed=SEED)
shallow = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=1, seed=SEED)
deep = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=12, seed=SEED)
shallow.train(data.train, n_epochs=6, n_infer=20, seed=SEED)
deep.train(data.train, n_epochs=6, n_infer=20, seed=SEED + 1)

def recon_mse(model):
    _, _, tr = model.infer(data.test, n_steps=40, record=True)
    return float(np.mean(tr.error**2))

print(f"n_level2=1   held-out MSE {recon_mse(shallow):.4f}")
print(f"n_level2=12  held-out MSE {recon_mse(deep):.4f}")
print('Compare held-out reconstruction errors at the same inference budget. This tests the added latent capacity on these patches; it does not isolate the organization of cortical processing.')
