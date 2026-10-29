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

from brainsmodels.values import KILL_PROBE, SAVE_PROBE, make_crawl, train_on_crawl
crawl = make_crawl(
    n_sentences=280,
    seed=SEED,
    switch_p_act=0.5,
    foot_p_act=0.5,
    save_p_act=1.0,
    kill_p_act=0.0,
)
run = train_on_crawl(crawl, seed=SEED, n_steps=400)
print(f"P(act | 'to save five…')  {run.save_act:.2f}")
print(f"P(act | 'to kill one…')   {run.kill_act:.2f}")
print(
    "\nThe two probes are paraphrases of one five-versus-one arithmetic. "
    "The corpus tied `save` to `act` and `kill` to `refrain`. The model "
    "follows the words it was given, not the outcome those words name. "
    "That is notebook 04's framing result on a trolley. It is not a "
    "value, and not dumbfounding in a person."
)
