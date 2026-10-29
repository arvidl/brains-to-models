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

from brainsmodels.dynamics import corrupt, hebb_weights, hopfield_async, make_patterns, overlaps
patterns = make_patterns(n=32, seed=SEED)
W = hebb_weights(patterns)
fracs = np.linspace(0.0, 0.6, 13)
ok = []
for frac in fracs:
    cue = corrupt(patterns[0], frac=float(frac), seed=SEED + 3)
    recalled, _, _ = hopfield_async(cue, W, n_sweeps=8, seed=SEED)
    ov = float(overlaps(recalled, patterns)[0])
    ok.append(ov)
    print(f"frac {frac:.2f}  overlap with ξ0 {ov:+.2f}")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(fracs, ok, "o-")
ax.axhline(0.8, color="0.6", ls="--", lw=1)
ax.set_xlabel("flip fraction")
ax.set_ylabel("final overlap with ξ0")
ax.set_title("A Hopfield basin has a radius; past it, recall fails")
plt.tight_layout()
plt.show()
print('Report retrieval success across corruption levels and repeat with several seeds. The result concerns this stored-pattern system; a perceptual interpretation would need separate observations.')
