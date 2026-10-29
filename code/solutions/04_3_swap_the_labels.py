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
classic = make_crawl(n_sentences=300, seed=SEED, gain_p_sure=1.0, loss_p_sure=0.0)
flipped = make_crawl(n_sentences=300, seed=SEED + 1, gain_p_sure=0.0, loss_p_sure=1.0)
a = train_on_crawl(classic, seed=SEED, n_steps=350)
b = train_on_crawl(flipped, seed=SEED + 1, n_steps=350)
print(f"{'corpus':<12}{'P(sure|gain)':>14}{'P(sure|loss)':>14}")
print(f"{'classic':<12}{a.gain_sure:>14.2f}{a.loss_sure:>14.2f}")
print(f"{'reversed':<12}{b.gain_sure:>14.2f}{b.loss_sure:>14.2f}")
print(
    "\nThe architecture did not prefer the textbook answers. It preferred "
    "the completions it was given. Calling the reverse run 'debiasing' "
    "would be a category error: nothing in a person was changed. A "
    "preference-tuned LLM can be pushed the same way. That is still not "
    "a value, and not Tversky and Kahneman's subjects."
)
