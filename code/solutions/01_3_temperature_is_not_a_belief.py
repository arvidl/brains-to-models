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

from brainsmodels.tiny_transformer import train_demo
model, tokenizer, _, _ = train_demo(seed=SEED, n_steps=400)
prompt = "the cat sat on the "
temps = (0.0, 0.4, 1.2)
objects = ("mat", "hat", "log", "box")

for T in temps:
    texts = [
        model.generate(prompt, tokenizer, n_tokens=12, temperature=T, seed=SEED + k)
        for k in range(20)
    ]
    tails = [t[len(prompt) :].split()[0] if t[len(prompt) :].split() else "" for t in texts]
    hits = {w: sum(w in tail for tail in tails) for w in objects}
    uniq = len(set(tails))
    print(f"T={T:<4}  unique first-words {uniq:2d}  object hits {hits}")
    print("   e.g.", texts[0][len(prompt) : 28])

print('Temperature rescales the next-character distribution. Compare repetition and variation in the samples; sampling sharpness alone does not measure factual accuracy or understanding.')
