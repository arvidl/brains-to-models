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

from brainsmodels.landscape import DEFAULT_START, descend, energy
path = descend(DEFAULT_START)
u = np.array([float(energy(x, y)) for x, y in path])
ups = int(np.sum(np.diff(u) > 1e-9))
print(f"steps {len(path)}  ΔU {u[-1] - u[0]:+.3f}  uphill steps {ups}")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(u)
ax.set_xlabel("descent step")
ax.set_ylabel("U(x, y)")
ax.set_title("Teaching descent on a plotted square")
plt.tight_layout()
plt.show()
print('A finite Euler step can increase energy when it overshoots. Check the step size and clipping before interpreting a nonmonotone path.')
