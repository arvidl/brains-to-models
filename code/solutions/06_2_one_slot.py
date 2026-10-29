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
old = ws.CAPACITY
try:
    two = ws.run_trial("cap2", ws.STRONG_PEAK, ws.RUNNER_PEAK)
    ws.CAPACITY = 1
    one = ws.run_trial("cap1", ws.STRONG_PEAK, ws.RUNNER_PEAK)
finally:
    ws.CAPACITY = old
print("capacity 2 occupants:", two.occupants)
print("capacity 1 occupants:", one.occupants)
print('A one-slot buffer excludes some active modules from broadcast. This demonstrates capacity-limited access under the chosen selection rule.')
