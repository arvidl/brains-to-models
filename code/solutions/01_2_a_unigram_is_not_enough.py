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

from brainsmodels.tiny_transformer import CharTokenizer, make_corpus, train_demo
model, tokenizer, text, _ = train_demo(seed=SEED, n_steps=200)
ids = tokenizer.encode(text)
n = len(ids)
cut = int(0.85 * n)
train_ids, test_ids = ids[:cut], ids[cut:]

unigram = np.ones(tokenizer.vocab_size)
for i in train_ids:
    unigram[int(i)] += 1.0
unigram = unigram / unigram.sum()

bigram = np.ones((tokenizer.vocab_size, tokenizer.vocab_size))
for a, b in zip(train_ids[:-1], train_ids[1:]):
    bigram[int(a), int(b)] += 1.0
bigram = bigram / bigram.sum(axis=1, keepdims=True)

def nll_unigram(seq):
    return float(-np.mean(np.log(unigram[seq])))

def nll_bigram(seq):
    return float(-np.mean(np.log(bigram[seq[:-1], seq[1:]])))

def nll_model(seq, block=32):
    losses = []
    for start in range(0, len(seq) - block - 1, block):
        ctx = seq[start : start + block]
        tgt = seq[start + 1 : start + 1 + block]
        logits, _ = model.forward(ctx)
        p = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        losses.append(-np.log(np.clip(p[np.arange(block), tgt], 1e-9, 1.0)).mean())
    return float(np.mean(losses))

print(f"chance (ln V)   {np.log(tokenizer.vocab_size):.3f}")
print(f"unigram  test   {nll_unigram(test_ids):.3f}")
print(f"bigram   test   {nll_bigram(test_ids):.3f}")
print(f"1-layer  test   {nll_model(test_ids):.3f}")
print(
    "\nThe unigram knows that space and e are common. The bigram knows that "
    "'t' often follows 'a' in 'cat' / 'sat' / 'mat'. The transformer still "
    "wins: the shared sentence frame is longer than one previous character. "
    "Attention is a look-back over a window, not a better unigram. It is still "
    "not a world model."
)
