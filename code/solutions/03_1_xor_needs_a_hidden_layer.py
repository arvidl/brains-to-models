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

from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_xor
x, y = make_xor(n=200, seed=SEED)
spk = SpikingPerceptron(seed=SEED)
mlp = SmallMLP(n_hidden=16, seed=SEED)
thin = SmallMLP(n_hidden=1, seed=SEED)
spk.train(x, y, n_epochs=30, seed=SEED)
mlp.train(x, y, n_epochs=30)
thin.train(x, y, n_epochs=30)
print(f"LIF perceptron (2→2, local delta)     acc {np.mean(spk.predict(x, seed=SEED) == y):.2f}")
print(f"MLP 2→16→2, SGD                       acc {np.mean(mlp.predict(x) == y):.2f}")
print(f"MLP 2→1→2, SGD                        acc {np.mean(thin.predict(x) == y):.2f}")
print(
    "\nThe thin MLP is still SGD, and it still cannot draw the XOR "
    "partition: one hidden unit is a single half-plane. The LIF net is "
    "a different failure: two output cells and a local count rule, no "
    "hidden layer to assign credit through. Same points, two reasons to "
    "miss. Neither miss is a synapse in cortex."
)
