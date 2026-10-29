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
for q in ("human", "not a word in the catalogue", "04", "02b"):
    hits = lookup(q)
    print(f"{q!r:40}  {len(hits)} hit(s):", ", ".join(c.title for c in hits) or "—")
print()
print(
    "The word 'human' sits in two cells by construction (the left column). "
    "A nonsense string sits in none. This notebook is an index, like "
    "notebook 00. It does not train, and a lookup hit is not evidence "
    "that a foundation model 'is' that cell. Notebook 04 is where a "
    "designed corpus is put on the table."
)
