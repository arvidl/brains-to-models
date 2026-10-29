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

from brainsmodels.biases import make_crawl, next_token_probs, sure_rate, train_on_crawl
crawl = make_crawl(n_sentences=300, seed=SEED, gain_p_sure=1.0, loss_p_sure=0.0)
run = train_on_crawl(crawl, seed=SEED, n_steps=350)
novel = "if program one is adopted two hundred people will be saved . people choose the"
probs = next_token_probs(run.model, run.tokenizer, novel)
print("trained probe  P(sure|gain)", f"{run.gain_sure:.2f}")
ps = float(probs[run.tokenizer.stoi["sure"]])
pr = float(probs[run.tokenizer.stoi["risky"]])
top = sorted(run.tokenizer.stoi, key=lambda w: -float(probs[run.tokenizer.stoi[w]]))[:6]
print(f"novel probe    P(sure) {ps:.3f}  P(risky) {pr:.3f}")
print("novel probe top words:", ", ".join(f"{w} {probs[run.tokenizer.stoi[w]]:.3f}" for w in top))
print(
    "\nDropping `most` moves the mass onto other training words "
    f"({top[0]!r}), not onto `sure`. The sure/risky ratio among a tiny "
    "remainder is not the bias surviving. The framing pattern was a "
    "regularity of the designed sentences, not a property of the "
    "architecture, and not a person facing lives saved or lost."
)
