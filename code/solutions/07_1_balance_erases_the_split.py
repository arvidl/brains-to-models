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

from brainsmodels.values import make_crawl, train_on_crawl
specs = {
    "biased (usual)": dict(switch_p_act=1.0, foot_p_act=0.0),
    "balanced": dict(switch_p_act=0.5, foot_p_act=0.5),
    "reversed": dict(switch_p_act=0.0, foot_p_act=1.0),
}
print(f"{'corpus':<18}{'P(act|switch)':>16}{'P(act|foot)':>14}")
for name, kw in specs.items():
    run = train_on_crawl(make_crawl(n_sentences=280, seed=SEED, **kw), seed=SEED, n_steps=400)
    print(f"{name:<18}{run.switch_act:>16.2f}{run.foot_act:>14.2f}")
print('Compare the measured probabilities across all three corpora. Changes show sensitivity to designed text associations, with finite training and seed effects; they are not measurements of human moral judgment.')
