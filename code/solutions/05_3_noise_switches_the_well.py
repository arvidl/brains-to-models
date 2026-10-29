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

from brainsmodels.dynamics import necker_walk
def n_switches(xs, win=40):
    ma = np.convolve(xs, np.ones(win) / win, mode="valid")
    return int(np.sum(np.diff(np.sign(ma)) != 0))

fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.0), sharex=True)
for ax, sig in zip(axes, (0.05, 0.55)):
    xs = necker_walk(sigma=sig, x0=0.0, seed=SEED)
    ax.plot(xs, lw=0.8)
    ax.axhline(1, color="0.7", ls="--", lw=1)
    ax.axhline(-1, color="0.7", ls="--", lw=1)
    ax.set_ylabel("x")
    ax.set_title(f"σ = {sig}   switches ≈ {n_switches(xs)}", fontsize=9)
axes[-1].set_xlabel("step")
fig.suptitle("The same double well; only the noise scale changed")
plt.tight_layout()
plt.show()
print(
    "Small noise: the walk falls into one well and stays. Larger noise: "
    "it crosses the saddle. That is the teaching picture of bistable "
    "percepts as two attractors plus a perturbation. There is no cube, "
    "no rivalry experiment, and no claim that next-token sampling is "
    "seeing."
)
