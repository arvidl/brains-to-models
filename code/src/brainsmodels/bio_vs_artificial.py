"""Local Hebbian / LIF versus SGD on the same 2-D tasks.

A leaky integrate-and-fire perceptron updates synapses from pre- and
post-synaptic spike counts (a local delta rule). A small MLP assigns
error through a hidden layer by SGD. Both see the same points: a
linearly separable pair of Gaussians, and a four-blob XOR.

Companion: ``code/notebooks/03_bio_vs_artificial.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

_BIO = "#2c5f6e"
_MLP = "#9a4a2a"
_C0 = "#4a7c8c"
_C1 = "#c47a3a"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_linear(
    n: int = 200,
    seed: int = DEFAULT_SEED,
    std: float = 0.28,
) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.15, 0.0], std, size=(n // 2, 2))
    x1 = rng.normal([1.15, 0.0], std, size=(n - n // 2, 2))
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def make_xor(
    n: int = 200,
    seed: int = DEFAULT_SEED,
    std: float = 0.22,
) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    labels = np.array([0, 1, 1, 0])
    per = n // 4
    xs, ys = [], []
    for i, c in enumerate(centres):
        k = per if i < 3 else n - 3 * per
        xs.append(rng.normal(c, std, size=(k, 2)))
        ys.append(np.full(k, labels[i], dtype=int))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Spiking perceptron (local delta on spike counts)
# ---------------------------------------------------------------------------


@dataclass
class SpikingPerceptron:
    """Two Poisson inputs, two LIF outputs, local delta on spike counts."""

    n_steps: int = 28
    leak: float = 0.20
    threshold: float = 0.70
    lr: float = 0.18
    seed: int = DEFAULT_SEED
    w: FloatArray = field(init=False)
    bias: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.25, size=(2, 2))
        self.bias = np.zeros(2)

    def _rates(self, x: FloatArray) -> FloatArray:
        return np.clip(0.20 + 0.16 * np.asarray(x, dtype=float), 0.05, 0.55)

    def encode(self, x: FloatArray, rng: np.random.Generator) -> FloatArray:
        rate = self._rates(np.asarray(x, dtype=float))
        return (rng.random((self.n_steps, 2)) < rate).astype(float)

    def _run(self, spikes: FloatArray) -> tuple[FloatArray, FloatArray]:
        v = np.zeros(2)
        pre = spikes.mean(axis=0)
        post = np.zeros(2)
        for t in range(self.n_steps):
            v = (1.0 - self.leak) * v + spikes[t] @ self.w + self.bias
            fired = v >= self.threshold
            post += fired.astype(float)
            v[fired] = 0.0
        return post / self.n_steps, pre

    def predict_one(self, x: FloatArray, rng: np.random.Generator) -> int:
        spikes = self.encode(x, rng)
        rate, _ = self._run(spikes)
        return int(np.argmax(rate))

    def predict_rate(self, x: FloatArray) -> IntArray:
        """Deterministic readout from mean input rates (for learning curves)."""
        pre = self._rates(x)
        return np.argmax(pre @ self.w + self.bias, axis=1).astype(int)

    def predict(self, x: FloatArray, seed: int | None = None, n_votes: int = 3) -> IntArray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        votes = np.zeros((len(x), 2))
        for _ in range(n_votes):
            for i, row in enumerate(x):
                spikes = self.encode(row, rng)
                rate, _ = self._run(spikes)
                votes[i] += rate
        return np.argmax(votes, axis=1).astype(int)

    def train(
        self,
        x: FloatArray,
        y: IntArray,
        n_epochs: int = 40,
        seed: int | None = None,
    ) -> list[float]:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        acc: list[float] = []
        eye = np.eye(2)
        for _ in range(n_epochs):
            order = rng.permutation(len(x))
            for i in order:
                spikes = self.encode(x[i], rng)
                rate, pre = self._run(spikes)
                err = eye[int(y[i])] - rate
                self.w += self.lr * np.outer(pre, err)
                self.bias += self.lr * err
            acc.append(float(np.mean(self.predict(x, seed=int(rng.integers(0, 1_000_000)), n_votes=5) == y)))
        return acc


# ---------------------------------------------------------------------------
# MLP + SGD
# ---------------------------------------------------------------------------


@dataclass
class SmallMLP:
    n_hidden: int = 16
    lr: float = 0.15
    seed: int = DEFAULT_SEED
    w1: FloatArray = field(init=False)
    b1: FloatArray = field(init=False)
    w2: FloatArray = field(init=False)
    b2: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 0.4, size=(2, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = rng.normal(0.0, 0.4, size=(self.n_hidden, 2))
        self.b2 = np.zeros(2)

    def _forward(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        return h, logits, p

    def predict(self, x: FloatArray) -> IntArray:
        return np.argmax(self._forward(x)[2], axis=1).astype(int)

    def train(
        self,
        x: FloatArray,
        y: IntArray,
        n_epochs: int = 40,
        batch: int = 16,
        seed: int | None = None,
    ) -> list[float]:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        acc: list[float] = []
        n = len(x)
        eye = np.eye(2)
        for _ in range(n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                xb, yb = x[idx], y[idx]
                h, _, p = self._forward(xb)
                dlogit = (p - eye[yb]) / len(xb)
                dw2 = h.T @ dlogit
                db2 = dlogit.sum(axis=0)
                dh = dlogit @ self.w2.T
                dh *= (xb @ self.w1 + self.b1) > 0
                dw1 = xb.T @ dh
                db1 = dh.sum(axis=0)
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
            acc.append(float(np.mean(self.predict(x) == y)))
        return acc


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    name: str
    x: FloatArray
    y: IntArray
    bio: SpikingPerceptron
    mlp: SmallMLP
    acc_bio: list[float]
    acc_mlp: list[float]


def run_task(
    name: str,
    maker,
    seed: int = DEFAULT_SEED,
    n: int = 200,
    n_epochs: int = 40,
) -> TaskResult:
    x, y = maker(n=n, seed=seed)
    bio = SpikingPerceptron(seed=seed)
    mlp = SmallMLP(seed=seed)
    acc_bio = bio.train(x, y, n_epochs=n_epochs, seed=seed)
    acc_mlp = mlp.train(x, y, n_epochs=n_epochs, seed=seed)
    return TaskResult(name, x, y, bio, mlp, acc_bio, acc_mlp)


def run_experiment(seed: int = DEFAULT_SEED, n_epochs: int = 40) -> dict[str, TaskResult]:
    set_seed(seed)
    return {
        "linear": run_task("linear", make_linear, seed=seed, n_epochs=n_epochs),
        "xor": run_task("xor", make_xor, seed=seed + 1, n_epochs=n_epochs),
    }


def _mesh_predict(
    predict_fn,
    lim: float = 2.2,
    n: int = 80,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    g = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(g, g)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = predict_fn(grid).reshape(xx.shape)
    return xx, yy, zz


def render_paper_figure(results: dict[str, TaskResult] | None = None, seed: int = DEFAULT_SEED) -> plt.Figure:
    if results is None:
        results = run_experiment(seed=seed)
    lin, xor = results["linear"], results["xor"]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(6.3, 6.2), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.06)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1.0])
    axes_top = top.subplots(2, 3)
    ax_acc_l, ax_acc_x = bottom.subplots(1, 2)

    def _scatter(ax: plt.Axes, task: TaskResult) -> None:
        ax.scatter(
            task.x[task.y == 0, 0],
            task.x[task.y == 0, 1],
            s=8,
            c=_C0,
            linewidths=0,
            alpha=0.85,
        )
        ax.scatter(
            task.x[task.y == 1, 0],
            task.x[task.y == 1, 1],
            s=8,
            c=_C1,
            linewidths=0,
            alpha=0.85,
        )
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _region(ax: plt.Axes, task: TaskResult, which: str) -> None:
        if which == "bio":
            xx, yy, zz = _mesh_predict(lambda g: task.bio.predict(g, seed=seed))
            cmap = "Blues"
        else:
            xx, yy, zz = _mesh_predict(task.mlp.predict)
            cmap = "Oranges"
        ax.contourf(xx, yy, zz, levels=[-0.1, 0.5, 1.1], colors=["#d9e6ea", "#f3e2d2"], alpha=0.95)
        ax.contour(xx, yy, zz, levels=[0.5], colors=[_BIO if which == "bio" else _MLP], linewidths=1.0)
        _ = cmap
        _scatter(ax, task)

    # Row 0: linear. Row 1: XOR.
    for row, task, letter in ((0, lin, "A"), (1, xor, "B")):
        _scatter(axes_top[row, 0], task)
        axes_top[row, 0].set_title(
            f"{letter}   {task.name}  ·  data",
            loc="left",
            fontsize=8,
            fontweight="bold",
        )
        _region(axes_top[row, 1], task, "bio")
        axes_top[row, 1].set_title("LIF + local delta", loc="left", fontsize=8)
        _region(axes_top[row, 2], task, "mlp")
        axes_top[row, 2].set_title("MLP + SGD", loc="left", fontsize=8)
        if row == 1:
            for ax in axes_top[row]:
                ax.set_xlabel("$x_1$", fontsize=7)
        axes_top[row, 0].set_ylabel("$x_2$", fontsize=7)

    def _acc(ax: plt.Axes, task: TaskResult, title: str) -> None:
        ep = np.arange(1, len(task.acc_bio) + 1)
        ax.plot(ep, task.acc_bio, color=_BIO, lw=1.6, label="LIF + local delta")
        ax.plot(ep, task.acc_mlp, color=_MLP, lw=1.6, label="MLP + SGD")
        ax.axhline(0.5, color="0.75", lw=0.6, ls=":")
        ax.set_ylim(0.35, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train accuracy")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _acc(ax_acc_l, lin, "C   Linear: both succeed")
    _acc(ax_acc_x, xor, "D   XOR: only SGD assigns credit")
    return fig


def write_figure(
    results: dict[str, TaskResult] | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(results, seed=seed)
    pdf = out / "bio_vs_artificial.pdf"
    png = out / "bio_vs_artificial.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    results = run_experiment(seed=seed)
    for name, task in results.items():
        print(f"{name:7s}  LIF {task.acc_bio[-1]:.2f}   MLP {task.acc_mlp[-1]:.2f}")
    return write_figure(results, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
