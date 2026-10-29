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

from brainsmodels import values as V
print(f"library summary  switch {V.HUMAN_SWITCH_ACT:.2f}  foot {V.HUMAN_FOOT_ACT:.2f}")
V.HUMAN_SWITCH_ACT = 0.70
V.HUMAN_FOOT_ACT = 0.30
print(f"edited labels    switch {V.HUMAN_SWITCH_ACT:.2f}  foot {V.HUMAN_FOOT_ACT:.2f}")
print(f"gap (switch − foot) {V.HUMAN_SWITCH_ACT - V.HUMAN_FOOT_ACT:.2f}")
print(
    "\nOnly panel A's reference bars would move. Panels B–D are trained "
    "on a designed corpus; they do not read HUMAN_*. If 0.90/0.10 were "
    "mistaken for Greene's sample, the error would be in the caption, "
    "not in the next-token mechanism. Restore the constants before "
    "`make figures` or the paper figure's panel A will be wrong."
)
V.HUMAN_SWITCH_ACT = 0.90
V.HUMAN_FOOT_ACT = 0.10
