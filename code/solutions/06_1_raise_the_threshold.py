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

import brainsmodels.workspace as ws
old = ws.THRESHOLD
try:
    for thr in (0.40, 0.55, 0.70):
        ws.THRESHOLD = thr
        trial = ws.run_trial("mid", 0.35, 0.0)
        print(
            f"θ={thr:.2f}  ignited={trial.ignited}  "
            f"t={trial.ignition_t}  occupants={trial.occupants}"
        )
finally:
    ws.THRESHOLD = old
print('Compare workspace occupancy for identical input at different thresholds. Ignition is defined by the model rule; it is not an independent consciousness measurement.')
