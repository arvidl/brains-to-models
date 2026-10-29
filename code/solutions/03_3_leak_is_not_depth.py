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

from brainsmodels.bio_vs_artificial import SpikingPerceptron, make_linear
x, y = make_linear(n=200, seed=SEED)
for leak in (0.05, 0.20, 0.80):
    net = SpikingPerceptron(leak=leak, seed=SEED)
    net.train(x, y, n_epochs=25, seed=SEED)
    acc = float(np.mean(net.predict(x, seed=SEED) == y))
    print(f"leak {leak:.2f}  train acc {acc:.2f}")
print('Leak changes membrane-state decay within a trial. Report its measured effect here; changing leak leaves the architecture and update rule fixed.')
