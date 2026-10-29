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

from brainsmodels.tiny_transformer import _softmax, train_demo
model, tokenizer, text, _ = train_demo(seed=SEED, n_steps=200)
prompt = "the cat sat on the "
ids = tokenizer.encode(prompt)
logits, causal = model.forward(ids, record_attn=True)
q, k = model._cache["q"], model._cache["k"]
scale = np.sqrt(model._d_head)
scores = np.einsum("bhtd,bhsd->bhts", q, k) / scale
leaky = _softmax(scores, axis=-1)

t = len(prompt)
# Query at the space after "cat" (index of that space).
q_at = prompt.index(" ", 4)
future = slice(q_at + 1, t)
leaked = float(leaky[0, :, q_at, future].sum() / model.n_heads)
blocked = float(causal[0, :, q_at, future].sum() / model.n_heads)
print(f"query at index {q_at} ({prompt[q_at]!r})")
print(f"mean mass on later characters, causal mask on : {blocked:.3f}")
print(f"mean mass on later characters, mask dropped   : {leaked:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
labels = [("␣" if c == " " else c) for c in prompt]
for ax, mat, title in (
    (axes[0], causal[0, 0, :t, :t], "head 1, causal"),
    (axes[1], leaky[0, 0, :t, :t], "head 1, no mask"),
):
    ax.imshow(mat, cmap="magma", vmin=0, vmax=max(0.2, float(mat.max())))
    ax.set_xticks(range(t), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(t), labels, fontsize=7)
    ax.set_title(title, fontsize=9)
fig.suptitle("Without the mask, a query can look at characters that do not yet exist")
plt.tight_layout()
plt.show()
print(
    "Generation writes one character at a time. A look-ahead weight is a weight "
    "on a blank. Training with the mask dropped would let the loss cheat on the "
    "training window (the future tokens are already in the batch) and would not "
    "transfer to left-to-right sampling. The mask is what makes next-token "
    "prediction a well-posed online objective, not a better picture of a cat."
)
