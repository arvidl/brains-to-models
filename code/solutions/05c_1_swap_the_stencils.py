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

from brainsmodels.analog_waves import (
    expressed,
    run_experiment,
    selectivity_table,
)
result = run_experiment(seed=SEED)
before = selectivity_table(result)
# Exchange stencils; drives stay put.
result.stencil_first, result.stencil_second = (
    result.stencil_second,
    result.stencil_first,
)
after = selectivity_table(result)
print("unit 2 before / after swap")
for name in before:
    print(f"  {name:10s}  {before[name][1]:.3f} → {after[name][1]:.3f}")
print('The fixed control pattern changes expressed activity while the object drives and weights remain fixed. This is a designed gating calculation.')
