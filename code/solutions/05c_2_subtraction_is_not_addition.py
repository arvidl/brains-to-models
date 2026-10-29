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

from brainsmodels.analog_waves import combine_waves, control_waves, positions
x = positions()
waves = control_waves(x)
pairs = (
    ("first", "match"),
    ("first", "recall"),
    ("second", "match"),
    ("second", "recall"),
)
added = np.stack([combine_waves(waves[a], waves[b]) for a, b in pairs])
subbed = np.stack([np.abs(waves[a] - waves[b]) for a, b in pairs])

def mean_offdiag(stack):
    diffs = [np.mean(np.abs(stack[i] - stack[j]))
             for i in range(4) for j in range(i + 1, 4)]
    return float(np.mean(diffs))

print(f"mean |gate_i − gate_j|  addition     {mean_offdiag(added):.3f}")
print(f"mean |gate_i − gate_j|  |difference| {mean_offdiag(subbed):.3f}")
print('Compare the gates produced by the mean and absolute difference. These are different pointwise combination rules; neither has been fitted to physiological measurements.')
