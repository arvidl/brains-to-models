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
intact = ws.run_trial("intact", ws.STRONG_PEAK, ws.RUNNER_PEAK)
lesion = ws.run_trial("lesion count", 0.0, ws.STRONG_PEAK)
print("intact :", intact.occupants, "ignition_t", intact.ignition_t)
print("lesion :", lesion.occupants, "ignition_t", lesion.ignition_t)
print(
    "\nWhoever crosses the bar is broadcast. Remove the original winner's "
    "drive and the runner occupies the slots. The workspace did not "
    "interpret the pulse; it admitted the largest local activities. "
    "That is access as a bottleneck, not a mind, and not a clinical "
    "neglect syndrome."
)
