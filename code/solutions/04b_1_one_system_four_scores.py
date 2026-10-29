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
for query in ("errors", "proof", "imitation", "goals"):
    hits = lookup(query)
    print(f"query {query!r}  →  {len(hits)} cell(s)")
    for cell in hits:
        print(" ", cell.title, "—", cell.verdict)
    print()
print(
    "A note-completer can copy a documented bias from discharge text "
    "(thinking humanly), emit a fluent but unproved plan (thinking "
    "rationally), pass a short Turing-style screen (acting humanly), and "
    "still have its objective added after pretraining (acting rationally). "
    "One architecture; four scores. Notebook 04 is the upper-left toy. "
    "The other three cells are verdicts, not extra nets."
)
