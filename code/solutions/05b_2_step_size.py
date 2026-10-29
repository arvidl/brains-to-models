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

from brainsmodels.landscape import DEFAULT_START, descend, energy, nearest_attractor
for lr in (0.02, 0.08, 0.25):
    path = descend(DEFAULT_START, lr=lr, n_steps=200)
    end = path[-1]
    well = nearest_attractor(tuple(end), radius=0.6)
    print(
        f"lr {lr:.2f}  steps {len(path):3d}  U={float(energy(*end)):+.3f}  "
        f"well {well}"
    )
print(
    "\nA small step crawls; the default step is the figure; a large step "
    "can overshoot a well or leave the plotted square (the path is "
    "clipped). The surface is a teaching slice of activity space, not "
    "SGD on a transformer loss. Notebook 05 is the controlled "
    "demonstration; this one is the map."
)
