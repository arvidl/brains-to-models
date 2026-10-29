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

from brainsmodels.four_views import format_quadrant, lookup
for query in ("mechanisms", "validity", "imitation", "goals"):
    hits = lookup(query)
    print(f"query {query!r}  →  {len(hits)} cell(s)")
    for cell in hits:
        print(" ", cell.title, "—", cell.verdict)
    print()
print('The catalog lists four evaluation criteria. For a clinical note-completer, specify a behavioral or mechanistic comparison, a validity check, an imitation test and an explicit decision objective. No scores are measured here.')
