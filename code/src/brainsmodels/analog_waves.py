"""Spatial stencils, analog superposition, and a traveling gate.

A designed 1-D sheet of multifunctional units. An alpha/beta-like
suppression pattern (a stencil) gates feedforward drive without a
weight update. Superposing two such patterns yields four combined
gates. A traveling stencil interpolates a sensory gate into a memory
gate. Teaching model of Miller, Brincat, and Roy (2026), not an LFP
and not a cortex.

Companion: ``code/notebooks/05c_analog_brain_waves.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]

_TEAL = "#2c5f6e"
_RUST = "#9a4a2a"
_GOLD = "#c4a35a"
_INK = "0.2"
_PALE = "#d9e4ea"

N = 48
N_TIME = 24


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


def positions(n: int = N) -> FloatArray:
    return np.linspace(0.0, 1.0, n)


def gaussian(x: FloatArray, mu: float, sigma: float) -> FloatArray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def clip01(v: FloatArray) -> FloatArray:
    return np.clip(v, 0.0, 1.0)


def object_drives(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Feedforward / gamma-like drive for two objects."""
    obj_a = gaussian(x, 0.30, 0.16)
    obj_b = gaussian(x, 0.70, 0.16)
    return obj_a, obj_b


def context_stencils(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Alpha/beta-like suppression (1 = closed, 0 = open).

    ``first`` closes the right/object-B side; ``second`` closes the
    mid-sheet overlap. The far left stays open in both contexts.
    """
    first = 0.92 * gaussian(x, 0.72, 0.13)
    second = 0.92 * gaussian(x, 0.48, 0.12)
    return clip01(first), clip01(second)


def pick_units(x: FloatArray, obj_a: FloatArray, obj_b: FloatArray,
               stencil_first: FloatArray, stencil_second: FloatArray) -> tuple[int, int]:
    """Unit 1: context-stable object-A preference. Unit 2: context-dependent mix."""
    a1 = expressed(obj_a, stencil_first)
    a2 = expressed(obj_a, stencil_second)
    b1 = expressed(obj_b, stencil_first)
    b2 = expressed(obj_b, stencil_second)
    a_mean = 0.5 * (a1 + a2)
    b_mean = 0.5 * (b1 + b2)
    a_stability = 1.0 - np.abs(a1 - a2)
    pure_score = a_mean - b_mean + 0.35 * a_stability
    mixed_score = np.abs(a1 - a2) + np.abs(b1 - b2) + 0.25 * (a1 + b2)
    unit_pure = int(np.argmax(pure_score))
    mixed_score[unit_pure] = -np.inf
    unit_mixed = int(np.argmax(mixed_score))
    return unit_pure, unit_mixed


def expressed(drive: FloatArray, stencil: FloatArray) -> FloatArray:
    """Gated activity: drive survives only where the stencil is open."""
    return drive * (1.0 - stencil)


def control_waves(x: FloatArray) -> dict[str, FloatArray]:
    """Two pairs of analog control waves (order × report mode)."""
    order_first = clip01(0.50 + 0.45 * np.sin(2.0 * np.pi * x))
    order_second = clip01(0.50 - 0.45 * np.sin(2.0 * np.pi * x))
    mode_match = clip01(0.50 + 0.45 * np.sin(4.0 * np.pi * x))
    mode_recall = clip01(0.50 - 0.45 * np.sin(4.0 * np.pi * x))
    return {
        "first": order_first,
        "second": order_second,
        "match": mode_match,
        "recall": mode_recall,
    }


def combine_waves(a: FloatArray, b: FloatArray) -> FloatArray:
    """Analog combination: the mean of two control waves."""
    return 0.5 * (a + b)


def traveling_open(x: FloatArray, t: float) -> FloatArray:
    """Open window (1 = expressed) that travels from a sensory to a memory locus."""
    mu = 0.28 + 0.44 * t
    return clip01(gaussian(x, mu, 0.11))


def traveling_movie(x: FloatArray, n_time: int = N_TIME) -> tuple[FloatArray, FloatArray]:
    """Space–time expressed activity: a broad drive gated by a traveling window."""
    drive = gaussian(x, 0.50, 0.32)
    times = np.linspace(0.0, 1.0, n_time)
    frames = np.stack([drive * traveling_open(x, t) for t in times])
    return times, frames


@dataclass
class AnalogWavesResult:
    x: FloatArray
    obj_a: FloatArray
    obj_b: FloatArray
    stencil_first: FloatArray
    stencil_second: FloatArray
    waves: dict[str, FloatArray]
    times: FloatArray
    travel: FloatArray
    unit_pure: int
    unit_mixed: int


def run_experiment(seed: int = DEFAULT_SEED) -> AnalogWavesResult:
    set_seed(seed)
    x = positions()
    obj_a, obj_b = object_drives(x)
    stencil_first, stencil_second = context_stencils(x)
    waves = control_waves(x)
    times, travel = traveling_movie(x)
    unit_pure, unit_mixed = pick_units(x, obj_a, obj_b, stencil_first, stencil_second)
    return AnalogWavesResult(
        x=x,
        obj_a=obj_a,
        obj_b=obj_b,
        stencil_first=stencil_first,
        stencil_second=stencil_second,
        waves=waves,
        times=times,
        travel=travel,
        unit_pure=unit_pure,
        unit_mixed=unit_mixed,
    )


def selectivity_table(result: AnalogWavesResult) -> dict[str, FloatArray]:
    """Expressed activity at the two marked units, four conditions."""
    conds = {
        "A-first": expressed(result.obj_a, result.stencil_first),
        "A-second": expressed(result.obj_a, result.stencil_second),
        "B-first": expressed(result.obj_b, result.stencil_first),
        "B-second": expressed(result.obj_b, result.stencil_second),
    }
    return {
        name: np.array([act[result.unit_pure], act[result.unit_mixed]])
        for name, act in conds.items()
    }


def render_paper_figure(
    result: AnalogWavesResult | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if result is None:
        result = run_experiment(seed=seed)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(6.8, 5.6), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.08)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    x = result.x
    idx1, idx2 = result.unit_pure, result.unit_mixed
    x1, x2 = float(x[idx1]), float(x[idx2])

    # A — 2 objects × 2 contexts as stacked 1-D activity
    rows = [
        ("A, first", expressed(result.obj_a, result.stencil_first), result.stencil_first, _TEAL),
        ("A, second", expressed(result.obj_a, result.stencil_second), result.stencil_second, _TEAL),
        ("B, first", expressed(result.obj_b, result.stencil_first), result.stencil_first, _RUST),
        ("B, second", expressed(result.obj_b, result.stencil_second), result.stencil_second, _RUST),
    ]
    offset = 1.15
    for i, (label, act, stencil, color) in enumerate(rows):
        y0 = (3 - i) * offset
        ax_a.fill_between(x, y0, y0 + stencil, color=_PALE, linewidth=0, zorder=1)
        ax_a.plot(x, y0 + act, color=color, lw=1.3, zorder=2)
        ax_a.text(-0.02, y0 + 0.35, label, ha="right", va="center", fontsize=6.5, color="0.25")
    ax_a.axvline(x1, color=_GOLD, lw=0.8, ls="--", zorder=3)
    ax_a.axvline(x2, color=_INK, lw=0.8, ls=":", zorder=3)
    ax_a.text(x1, 4 * offset - 0.15, "1", ha="center", va="bottom", fontsize=7, color=_GOLD)
    ax_a.text(x2, 4 * offset - 0.15, "2", ha="center", va="bottom", fontsize=7, color=_INK)
    ax_a.set_xlim(-0.22, 1.02)
    ax_a.set_ylim(-0.08, 4 * offset)
    ax_a.set_xlabel("Sheet position")
    ax_a.set_yticks([])
    ax_a.set_title("A   Stencil gates mixed selectivity", loc="left", fontsize=8, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)

    # B — analog combination of two control waves
    waves = result.waves
    combos = [
        ("first + match", combine_waves(waves["first"], waves["match"]), _TEAL),
        ("first + recall", combine_waves(waves["first"], waves["recall"]), "#5a8a7a"),
        ("second + match", combine_waves(waves["second"], waves["match"]), _RUST),
        ("second + recall", combine_waves(waves["second"], waves["recall"]), _GOLD),
    ]
    for label, gate, color in combos:
        ax_b.plot(x, gate, color=color, lw=1.3, label=label)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_xlabel("Sheet position")
    ax_b.set_ylabel("Combined gate")
    ax_b.set_title("B   Superposition of two contexts", loc="left", fontsize=8, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=6.0, loc="lower left", ncol=1)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # C — traveling stencil as space–time expressed activity
    im = ax_c.imshow(
        result.travel,
        aspect="auto",
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=float(result.travel.max()),
    )
    ax_c.set_xlabel("Sheet position")
    ax_c.set_ylabel("Time  (sensory $\\rightarrow$ memory)")
    ax_c.set_title("C   Traveling stencil relocates activity", loc="left", fontsize=8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.035, pad=0.02)
    cbar.set_label("Expressed drive", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    return fig


def write_figure(
    result: AnalogWavesResult | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(result, seed=seed)
    pdf = out / "analog_brain_waves.pdf"
    png = out / "analog_brain_waves.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    result = run_experiment(seed=seed)
    table = selectivity_table(result)
    print(f"unit 1 (pure-ish, idx {result.unit_pure})  unit 2 (mixed, idx {result.unit_mixed})")
    for name, vals in table.items():
        print(f"  {name:10s}  u1={vals[0]:.3f}  u2={vals[1]:.3f}")
    return write_figure(result, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
