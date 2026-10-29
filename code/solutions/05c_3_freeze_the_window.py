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

from brainsmodels.analog_waves import gaussian, positions, traveling_open
x = positions()
drive = gaussian(x, 0.50, 0.32)
times = np.linspace(0.0, 1.0, 24)
moving = np.stack([drive * traveling_open(x, t) for t in times])
frozen = np.stack([drive * traveling_open(x, 0.0) for _ in times])
peak_move = x[np.argmax(moving, axis=1)]
peak_frz = x[np.argmax(frozen, axis=1)]
print(f"moving  peak x: {peak_move[0]:.2f} → {peak_move[-1]:.2f}")
print(f"frozen  peak x: {peak_frz[0]:.2f} → {peak_frz[-1]:.2f}")
print(
    "A frozen window is a static stencil: drive is still gated, but "
    "activity no longer travels. Relocation was the traveling-wave "
    "claim. Both versions are designed functions, not LFPs, and not "
    "phenomenal experience."
)
