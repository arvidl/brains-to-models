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

from brainsmodels.landscape import ATTRACTORS, descend, energy, nearest_attractor
starts = ((-2.50, -3.50), (-2.18, -3.50), (0.80, 4.80))
print(f"{'start':<16}{'end':<16}{'U(end)':>8}  well")
for s in starts:
    path = descend(s)
    end = tuple(float(v) for v in path[-1])
    well = nearest_attractor(end, radius=0.6)
    print(f"{s!s:<16}{end!s:<16}{float(energy(*end)):>8.3f}  {well}")

print(
    "\nA small shift in the start can change the basin. On this surface "
    "that is a ridge in U. In the paper's reading, a slight rewording "
    "of a prompt is a perturbation near a saddle — a teaching analogy, "
    "not a measurement of a frontier model, and not a Hopfield net "
    "(that is notebook 05)."
)
