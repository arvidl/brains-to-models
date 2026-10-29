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

from brainsmodels.four_views import QUADRANTS, format_quadrant
for cell in QUADRANTS:
    flag = cell.companion or "(none)"
    print(f"{cell.title:<22}  companion {flag}")
print()
empty = [c for c in QUADRANTS if not c.companion]
print(format_quadrant(empty[0]))
print('The missing companion marks a gap in this collection. A useful addition could compare generated arguments with a proof checker or independently verified answers. The training objective alone does not determine whether reasoning succeeds.')
