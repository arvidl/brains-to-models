"""Hopfield basins, Necker switching, and attention as one retrieval step.

A classical Hopfield net stores two ±1 patterns as minima of
E = −½ sᵀ W s. Recall is descent. A double-well Langevin walk is the
Necker cube as two interpretations and enough noise to switch. One
modern-Hopfield / attention step retrieves a stored key by softmax
similarity (Ramsauer et al., 2020). No animation: the paper figure is
static traces.

Companion: ``code/notebooks/05_multistable_dynamics.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

_TEAL = "#2c5f6e"
_RUST = "#9a4a2a"
_INK = "0.2"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Classical Hopfield
# ---------------------------------------------------------------------------


def make_patterns(n: int = 32, seed: int = DEFAULT_SEED) -> FloatArray:
    """Two somewhat orthogonal ±1 patterns of length ``n``."""
    rng = np.random.default_rng(seed)
    p1 = rng.choice(np.array([-1.0, 1.0]), size=n)
    p2 = rng.choice(np.array([-1.0, 1.0]), size=n)
    # Flip a few bits of p2 if the patterns landed too aligned.
    if abs(float(p1 @ p2) / n) > 0.25:
        flip = rng.choice(n, size=n // 3, replace=False)
        p2 = p2.copy()
        p2[flip] *= -1.0
    return np.stack([p1, p2])


def hebb_weights(patterns: FloatArray) -> FloatArray:
    n = patterns.shape[1]
    w = patterns.T @ patterns / n
    np.fill_diagonal(w, 0.0)
    return w


def energy(state: FloatArray, weights: FloatArray) -> float:
    return float(-0.5 * state @ weights @ state)


def overlaps(state: FloatArray, patterns: FloatArray) -> FloatArray:
    n = patterns.shape[1]
    return patterns @ state / n


def hopfield_async(
    state: FloatArray,
    weights: FloatArray,
    n_sweeps: int = 8,
    seed: int = DEFAULT_SEED,
) -> tuple[FloatArray, list[FloatArray], list[float]]:
    """Asynchronous sign updates. Returns final state, overlap path, energies."""
    rng = np.random.default_rng(seed)
    s = state.copy()
    path = [s.copy()]
    energies = [energy(s, weights)]
    n = s.size
    for _ in range(n_sweeps * n):
        i = int(rng.integers(0, n))
        h = float(weights[i] @ s)
        s[i] = 1.0 if h >= 0.0 else -1.0
        path.append(s.copy())
        energies.append(energy(s, weights))
    return s, path, energies


def corrupt(pattern: FloatArray, frac: float = 0.35, seed: int = DEFAULT_SEED) -> FloatArray:
    rng = np.random.default_rng(seed)
    s = pattern.copy()
    n_flip = max(1, int(round(frac * s.size)))
    idx = rng.choice(s.size, size=n_flip, replace=False)
    s[idx] *= -1.0
    return s


def plane_state(u: float, v: float, patterns: FloatArray) -> FloatArray:
    """A continuous state in the span of the two stored patterns."""
    s = u * patterns[0] + v * patterns[1]
    nrm = float(np.linalg.norm(s))
    if nrm < 1e-9:
        return np.zeros_like(s)
    return np.tanh(s / nrm * 3.0)


def plane_coords(state: FloatArray, patterns: FloatArray) -> FloatArray:
    """Least-squares coefficients of ``state`` in the span of the patterns."""
    uv, *_ = np.linalg.lstsq(patterns.T, state, rcond=None)
    return uv


def plane_energy(
    patterns: FloatArray,
    weights: FloatArray,
    n_grid: int = 81,
    lim: float = 1.35,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    us = np.linspace(-lim, lim, n_grid)
    vs = np.linspace(-lim, lim, n_grid)
    uu, vv = np.meshgrid(us, vs)
    zz = np.empty_like(uu)
    for i in range(n_grid):
        for j in range(n_grid):
            zz[i, j] = energy(plane_state(uu[i, j], vv[i, j], patterns), weights)
    return uu, vv, zz


# ---------------------------------------------------------------------------
# Necker: double-well Langevin
# ---------------------------------------------------------------------------


def necker_energy(x: FloatArray | float) -> FloatArray | float:
    """E(x) = ¼ (x² − 1)². Wells at ±1; saddle at 0."""
    return 0.25 * (np.asarray(x) ** 2 - 1.0) ** 2


def necker_force(x: float) -> float:
    """−E'(x) = x − x³."""
    return x - x**3


def necker_walk(
    n_steps: int = 1600,
    dt: float = 0.03,
    sigma: float = 0.55,
    x0: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = float(x0)
    xs = np.empty(n_steps)
    scale = sigma * np.sqrt(dt)
    for t in range(n_steps):
        x = x + dt * necker_force(x) + scale * float(rng.normal())
        xs[t] = x
    return xs


# ---------------------------------------------------------------------------
# Modern Hopfield / attention (one step)
# ---------------------------------------------------------------------------


def attention_retrieve(
    query: FloatArray,
    keys: FloatArray,
    beta: float,
) -> tuple[FloatArray, FloatArray]:
    """ξ ← X softmax(β Xᵀ ξ). Keys are also values."""
    logits = beta * (keys @ query)
    logits = logits - float(np.max(logits))
    w = np.exp(logits)
    w = w / w.sum()
    retrieved = w @ keys
    return retrieved, w


# ---------------------------------------------------------------------------
# Experiment bundle
# ---------------------------------------------------------------------------


@dataclass
class DynamicsResult:
    patterns: FloatArray
    weights: FloatArray
    cue: FloatArray
    recalled: FloatArray
    path: list[FloatArray]
    energies: list[float]
    uu: FloatArray
    vv: FloatArray
    zz: FloatArray
    necker: FloatArray
    query: FloatArray
    weights_soft: dict[float, FloatArray]


def run_experiment(seed: int = DEFAULT_SEED) -> DynamicsResult:
    patterns = make_patterns(n=32, seed=seed)
    weights = hebb_weights(patterns)
    cue = corrupt(patterns[0], frac=0.35, seed=seed + 3)
    recalled, path, energies = hopfield_async(cue, weights, n_sweeps=6, seed=seed)
    uu, vv, zz = plane_energy(patterns, weights)
    necker = necker_walk(seed=seed)
    query = 0.55 * patterns[0] + 0.45 * patterns[1]
    query = query / float(np.linalg.norm(query))
    soft = {beta: attention_retrieve(query, patterns, beta)[1] for beta in (1.0, 8.0)}
    return DynamicsResult(
        patterns=patterns,
        weights=weights,
        cue=cue,
        recalled=recalled,
        path=path,
        energies=energies,
        uu=uu,
        vv=vv,
        zz=zz,
        necker=necker,
        query=query,
        weights_soft=soft,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_paper_figure(
    result: DynamicsResult | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if result is None:
        result = run_experiment(seed=seed)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A — energy on the plane of the two memories
    zz = result.zz
    vmax = float(np.percentile(zz, 92))
    vmin = float(zz.min())
    ax_a.contourf(
        result.uu,
        result.vv,
        zz,
        levels=18,
        cmap="YlGnBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    uv = np.array([plane_coords(s, result.patterns) for s in result.path[::4]])
    mem = np.stack([plane_coords(p, result.patterns) for p in result.patterns])
    ax_a.plot(uv[:, 0], uv[:, 1], color=_RUST, lw=1.2, zorder=3)
    ax_a.scatter(uv[0, 0], uv[0, 1], s=18, c=_RUST, zorder=4, label="cue")
    ax_a.scatter(uv[-1, 0], uv[-1, 1], s=22, c=_TEAL, zorder=4, label="recalled")
    ax_a.scatter(mem[:, 0], mem[:, 1], s=28, c="white", edgecolors=_INK, lw=0.6, zorder=5)
    ax_a.text(mem[0, 0] + 0.06, mem[0, 1] + 0.06, r"$\xi_1$", fontsize=8, color="0.15")
    ax_a.text(mem[1, 0] + 0.06, mem[1, 1] + 0.06, r"$\xi_2$", fontsize=8, color="0.15")
    ax_a.set_xlabel(r"coefficient of $\xi_1$")
    ax_a.set_ylabel(r"coefficient of $\xi_2$")
    ax_a.set_title("A   Hopfield energy, two memories", loc="left", fontsize=8, fontweight="bold")
    ax_a.set_aspect("equal")
    ax_a.legend(frameon=False, fontsize=6.5, loc="lower left")

    # B — energy falls
    ax_b.plot(result.energies, color=_TEAL, lw=1.3)
    ax_b.set_xlabel("Async update")
    ax_b.set_ylabel("Energy $E$")
    ax_b.set_title("B   Recall is descent", loc="left", fontsize=8, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # C — Necker two-well + switching
    xs = np.linspace(-1.8, 1.8, 200)
    ax_c.plot(xs, necker_energy(xs), color="0.35", lw=1.2)
    walk = result.necker
    t = np.arange(len(walk))
    # Plot the walk as a time-coloured scatter along the well, offset above E
    e_walk = necker_energy(walk)
    ax_c.scatter(walk, e_walk + 0.04, c=t, cmap="cividis", s=4, linewidths=0, zorder=3)
    ax_c.axvline(-1.0, color=_TEAL, lw=0.6, ls=":")
    ax_c.axvline(1.0, color=_RUST, lw=0.6, ls=":")
    ax_c.set_xlabel("Interpretation  $x$")
    ax_c.set_ylabel("$E(x)$")
    ax_c.set_title("C   Necker: two wells, noise switches", loc="left", fontsize=8, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # D — attention / modern Hopfield
    betas = sorted(result.weights_soft)
    x = np.arange(2)
    width = 0.36
    w_lo = result.weights_soft[betas[0]]
    w_hi = result.weights_soft[betas[1]]
    ax_d.bar(x - width / 2, w_lo, width, color=_TEAL, label=rf"$\beta={betas[0]:.0f}$")
    ax_d.bar(x + width / 2, w_hi, width, color=_RUST, label=rf"$\beta={betas[1]:.0f}$")
    ax_d.set_xticks(x, [r"$\xi_1$", r"$\xi_2$"])
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_ylabel("Attention weight")
    ax_d.set_title("D   One Hopfield / attention step", loc="left", fontsize=8, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=6.5)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    return fig


def write_figure(
    result: DynamicsResult | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(result, seed=seed)
    pdf = out / "multistable_dynamics.pdf"
    png = out / "multistable_dynamics.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    result = run_experiment(seed=seed)
    ov0 = overlaps(result.cue, result.patterns)
    ov1 = overlaps(result.recalled, result.patterns)
    print(f"cue      overlaps {ov0[0]:+.2f}  {ov0[1]:+.2f}  E={result.energies[0]:.3f}")
    print(f"recalled overlaps {ov1[0]:+.2f}  {ov1[1]:+.2f}  E={result.energies[-1]:.3f}")
    print(f"attention β=1  {result.weights_soft[1.0]}")
    print(f"attention β=8  {result.weights_soft[8.0]}")
    return write_figure(result, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
