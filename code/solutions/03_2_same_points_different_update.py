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

from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_linear
x, y = make_linear(n=200, seed=SEED)
spk = SpikingPerceptron(seed=SEED)
mlp = SmallMLP(n_hidden=16, seed=SEED)
spk.train(x, y, n_epochs=25, seed=SEED)
mlp.train(x, y, n_epochs=25)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, pts, c, lab in (
    (axes[0], x[y == 0], "#4a7c8c", "class 0"),
    (axes[0], x[y == 1], "#c47a3a", "class 1"),
):
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=c, label=lab, alpha=0.7)
for k, col in enumerate(("#2c5f6e", "#9a4a2a")):
    w = spk.w[:, k]
    axes[0].arrow(0, 0, w[0], w[1], color=col, width=0.02, head_width=0.12)
    axes[0].set_title("LIF: two columns of W (2×2)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False)
axes[0].set_aspect("equal")

im = axes[1].imshow(mlp.w1.T, cmap="coolwarm", aspect="auto")
axes[1].set_title("MLP: W1 (2 → 16), updated by SGD", fontsize=9)
axes[1].set_xlabel("input dim")
axes[1].set_ylabel("hidden unit")
fig.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.show()
print(
    "LIF: each synapse saw a pre-count and a post-count. MLP: every hidden "
    "weight saw a scalar that had already been multiplied by the downstream "
    "error. Same point cloud; the elementary increment is not the same "
    "object. That is the §4 claim, not a verdict on which rule is 'more biological'."
)
