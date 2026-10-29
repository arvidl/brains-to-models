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

from brainsmodels.biases import make_crawl, train_on_crawl
rates = (1.0, 0.75, 0.5, 0.25)
model_p = []
for p in rates:
    crawl = make_crawl(n_sentences=240, seed=SEED, gain_p_sure=p, loss_p_sure=1.0 - p)
    run = train_on_crawl(crawl, seed=SEED, n_steps=350)
    model_p.append(run.gain_sure)
    print(f"corpus P(sure|gain) {p:.2f}  model {run.gain_sure:.2f}")

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.plot([0, 1], [0, 1], color="0.75", lw=1)
ax.plot(rates, model_p, "o-", color="#2c5f6e")
ax.set_xlabel("corpus P(sure | gain frame)")
ax.set_ylabel("model P(sure | gain probe)")
ax.set_title("Next-token mass tracks the designed frequency")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()
print('These probes reuse training-template prefixes. Compare completion probability with the designed corpus frequency and report deviations; this is a fitted association, not a human behavioral measurement.')
