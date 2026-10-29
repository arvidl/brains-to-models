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
x = data.test[0]
fresh = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)
trained = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)
trained.train(data.train, n_epochs=6, n_infer=20, seed=SEED)

def curve(model, steps=40):
    _, mse = model.infer_history(x, n_steps=steps)
    return np.asarray(mse)

c0, c1 = curve(fresh), curve(trained)
print(f"untrained  MSE t=1 {c0[0]:.3f}  t=40 {c0[-1]:.3f}")
print(f"trained    MSE t=1 {c1[0]:.3f}  t=40 {c1[-1]:.3f}")
print(f"drop untrained {c0[0] - c0[-1]:.3f}   drop trained {c1[0] - c1[-1]:.3f}")

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(c0, label="untrained weights")
ax.plot(c1, label="after 6 epochs")
ax.set_xlabel("inference step")
ax.set_ylabel("prediction-error MSE")
ax.legend(frameon=False)
ax.set_title("Same infer loop; only the descending weights changed")
plt.tight_layout()
plt.show()
print('Inference changes the current representation with weights fixed. Compare the measured residuals for trained and initial weights to separate iterative fitting from the effect of learning.')
