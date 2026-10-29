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

from brainsmodels.dynamics import attention_retrieve, make_patterns
keys = make_patterns(n=32, seed=SEED)
query = 0.55 * keys[0] + 0.45 * keys[1]
query = query / float(np.linalg.norm(query))
print(f"{'beta':>6}  w(ξ0)   w(ξ1)")
ws = []
for beta in (0.5, 1.0, 4.0, 16.0):
    _, w = attention_retrieve(query, keys, beta)
    ws.append(w)
    print(f"{beta:6.1f}  {w[0]:.3f}   {w[1]:.3f}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot((0.5, 1, 4, 16), [w[0] for w in ws], "o-", label="ξ0")
ax.plot((0.5, 1, 4, 16), [w[1] for w in ws], "s-", label="ξ1")
ax.set_xlabel("β")
ax.set_ylabel("softmax weight")
ax.legend(frameon=False)
ax.set_title("Soft β shares; sharp β retrieves")
plt.tight_layout()
plt.show()
print(
    "Ramsauer et al.: one attention step with keys = values = stored "
    "patterns. Large β is a hard lookup, not a deeper network. Notebook "
    "01's maps are many such steps, trained, with a causal mask. This "
    "exercise is the *update*, not a language model and not a percept."
)
