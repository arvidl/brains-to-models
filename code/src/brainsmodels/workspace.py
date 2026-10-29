"""Toy global workspace: limited-capacity broadcast and ignition.

Eight specialised modules run in parallel. A workspace of two slots
opens only when some module crosses a threshold; the winners are
broadcast back. Below threshold the same computation stays local.
That is Dehaene / Baars as a schematic, not a cortex and not Claude.

Companion: ``code/notebooks/06_global_workspace.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

_TEAL = "#2c5f6e"
_RUST = "#9a4a2a"
_GOLD = "#c4a35a"
_INK = "0.18"
_PALE = "#d9e4ea"
_AUTO = "#c5d4dc"

N_MODULES = 8
CAPACITY = 2
THRESHOLD = 0.55
DECAY = 0.22
BROADCAST_GAIN = 0.28
N_STEPS = 40
PULSE_START = 8
PULSE_WIDTH = 4
BASELINE_DRIVE = 0.02

MODULE_NAMES: tuple[str, ...] = (
    "vision",
    "lexicon",
    "syntax",
    "count",
    "plan",
    "affect",
    "motor",
    "eval",
)

# Pulse targets: count is the intended winner; plan is the runner-up.
WINNER = 3
RUNNER = 4
WEAK_PEAK = 0.20
STRONG_PEAK = 0.55
RUNNER_PEAK = 0.45


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


def pulse(t: int, peak: float) -> float:
    """Raised-cosine pulse on ``[PULSE_START, PULSE_START + PULSE_WIDTH]``."""
    if t < PULSE_START or t > PULSE_START + PULSE_WIDTH:
        return 0.0
    phase = (t - PULSE_START) / PULSE_WIDTH
    return float(peak * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase)))


def drive_vector(t: int, peak: float, runner: float = 0.0) -> FloatArray:
    d = np.full(N_MODULES, BASELINE_DRIVE, dtype=float)
    d[WINNER] += pulse(t, peak)
    d[RUNNER] += pulse(t, runner)
    return d


def occupy(activity: FloatArray) -> IntArray:
    """Workspace slots: at most ``CAPACITY`` modules already above threshold."""
    above = np.flatnonzero(activity >= THRESHOLD)
    mask = np.zeros(N_MODULES, dtype=int)
    if len(above) == 0:
        return mask
    ranked = above[np.argsort(-activity[above])]
    mask[ranked[:CAPACITY]] = 1
    return mask


def step(activity: FloatArray, drive: FloatArray) -> tuple[FloatArray, IntArray]:
    mask = occupy(activity)
    nxt = (1.0 - DECAY) * activity + drive + BROADCAST_GAIN * mask
    nxt = np.clip(nxt, 0.0, 1.0)
    return nxt, occupy(nxt)


@dataclass(frozen=True)
class Trial:
    """One drive condition, run to ``N_STEPS``."""

    name: str
    peak: float
    runner: float
    activity: FloatArray
    occupancy: IntArray
    ignition_t: int | None

    @property
    def ignited(self) -> bool:
        return self.ignition_t is not None

    @property
    def occupants(self) -> tuple[str, ...]:
        last = self.occupancy[-1]
        return tuple(MODULE_NAMES[i] for i in range(N_MODULES) if last[i])


def run_trial(name: str, peak: float, runner: float = 0.0) -> Trial:
    s = np.full(N_MODULES, 0.05, dtype=float)
    act = np.zeros((N_STEPS, N_MODULES), dtype=float)
    occ = np.zeros((N_STEPS, N_MODULES), dtype=int)
    ignition_t: int | None = None
    for t in range(N_STEPS):
        s, mask = step(s, drive_vector(t, peak, runner))
        act[t] = s
        occ[t] = mask
        if ignition_t is None and mask.any():
            ignition_t = t
    return Trial(name, peak, runner, act, occ, ignition_t)


@dataclass(frozen=True)
class WorkspaceResult:
    weak: Trial
    strong: Trial
    drive_grid: FloatArray
    occupancy_curve: FloatArray


def run_experiment(seed: int = DEFAULT_SEED) -> WorkspaceResult:
    set_seed(seed)
    weak = run_trial("weak", WEAK_PEAK, 0.0)
    strong = run_trial("strong", STRONG_PEAK, RUNNER_PEAK)
    drives = np.linspace(0.10, 0.90, 17)
    curve = np.zeros_like(drives)
    for i, peak in enumerate(drives):
        trial = run_trial("sweep", float(peak), (RUNNER_PEAK / STRONG_PEAK) * float(peak))
        curve[i] = float(trial.occupancy[-1].sum())
    return WorkspaceResult(weak, strong, drives, curve)


def facts(result: WorkspaceResult) -> dict[str, object]:
    """Printed numbers for the notebook. None of these is a measurement of Claude."""
    return {
        "n_modules": N_MODULES,
        "capacity": CAPACITY,
        "threshold": THRESHOLD,
        "weak_ignited": result.weak.ignited,
        "weak_occupants": result.weak.occupants,
        "weak_peak_activity": float(result.weak.activity.max()),
        "strong_ignited": result.strong.ignited,
        "strong_ignition_t": result.strong.ignition_t,
        "strong_occupants": result.strong.occupants,
        "strong_peak_activity": float(result.strong.activity.max()),
        "jspace_variance_note": (
            "Gurnee et al. 2026: J-space component is a small slice "
            "(concept-vector median ~6–7%; activation variance never >10%). "
            "Not measured here."
        ),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_schematic(ax: plt.Axes, trial: Trial) -> None:
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    last = trial.occupancy[-1]
    angles = np.linspace(0.5 * np.pi, 0.5 * np.pi + 2 * np.pi, N_MODULES, endpoint=False)
    radius = 1.02
    ws = FancyBboxPatch(
        (-0.38, -0.28),
        0.76,
        0.56,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#eef4f6",
        edgecolor=_TEAL,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(ws)
    ax.text(0.0, 0.14, "workspace", ha="center", va="center", fontsize=7, color=_TEAL)
    ax.text(0.0, -0.02, f"{CAPACITY} slots", ha="center", va="center", fontsize=6.5, color="0.35")
    filled = [MODULE_NAMES[i] for i in range(N_MODULES) if last[i]]
    if filled:
        ax.text(
            0.0,
            -0.16,
            " · ".join(filled),
            ha="center",
            va="center",
            fontsize=6.5,
            color=_RUST,
            fontweight="bold",
        )
    else:
        ax.text(0.0, -0.16, "(empty)", ha="center", va="center", fontsize=6.5, color="0.5")

    for i, ang in enumerate(angles):
        x, y = radius * np.cos(ang), radius * np.sin(ang)
        in_ws = bool(last[i])
        face = {WINNER: _TEAL, RUNNER: _GOLD}.get(i, _PALE) if in_ws else _PALE
        edge = {WINNER: _TEAL, RUNNER: _GOLD}.get(i, "0.45") if in_ws else "0.45"
        circ = Circle((x, y), 0.20, facecolor=face, edgecolor=edge, lw=0.9, zorder=3)
        ax.add_patch(circ)
        ink = "white" if in_ws and i == WINNER else _INK
        ax.text(x, y, MODULE_NAMES[i], ha="center", va="center", fontsize=5.4, color=ink)
        if in_ws:
            ax.add_patch(
                FancyArrowPatch(
                    (0.72 * x, 0.72 * y),
                    (0.42 * x, 0.42 * y),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=_TEAL,
                    lw=0.9,
                    zorder=1,
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    (0.38 * x, 0.38 * y),
                    (0.68 * x, 0.68 * y),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    color=_GOLD,
                    lw=0.7,
                    linestyle="--",
                    zorder=1,
                )
            )
    ax.set_title("A   Modules compete; two can broadcast", loc="left", fontsize=8, fontweight="bold")


def _draw_traces(ax: plt.Axes, strong: Trial, weak: Trial) -> None:
    t = np.arange(N_STEPS)
    last = strong.occupancy[-1]
    winner_color = {WINNER: _TEAL, RUNNER: _GOLD}
    for i in range(N_MODULES):
        in_ws = bool(last[i])
        ax.plot(
            t,
            strong.activity[:, i],
            color=winner_color.get(i, "0.72"),
            lw=1.6 if in_ws else 0.8,
            zorder=3 if in_ws else 1,
            label=MODULE_NAMES[i] if in_ws else None,
        )
    ax.plot(
        t,
        weak.activity[:, WINNER],
        color=_RUST,
        lw=1.1,
        ls="--",
        zorder=2,
        label="count, weak drive",
    )
    if strong.ignition_t is not None:
        ax.axvline(strong.ignition_t, color=_RUST, lw=0.8, ls=":", zorder=2)
        ax.text(
            strong.ignition_t + 0.4,
            0.93,
            "ignition",
            fontsize=6.5,
            color=_RUST,
        )
    ax.axhline(THRESHOLD, color="0.4", lw=0.6, ls="--")
    ax.text(N_STEPS - 0.4, THRESHOLD + 0.03, r"$\theta$", fontsize=7, color="0.35", ha="right")
    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Module activity")
    ax.set_title("B   Strong drive ignites; weak stays local", loc="left", fontsize=8, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    _style(ax)


def _draw_curve(ax: plt.Axes, result: WorkspaceResult) -> None:
    ax.plot(result.drive_grid, result.occupancy_curve, color=_TEAL, lw=1.5, marker="o", ms=3.5)
    ax.axvline(WEAK_PEAK, color="0.55", lw=0.7, ls=":")
    ax.axvline(STRONG_PEAK, color=_RUST, lw=0.7, ls=":")
    ax.text(WEAK_PEAK, 2.12, "weak", fontsize=6.5, color="0.4", ha="center")
    ax.text(STRONG_PEAK, 2.12, "strong", fontsize=6.5, color=_RUST, ha="center")
    ax.set_ylim(-0.15, 2.35)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("Peak drive on the winning module")
    ax.set_ylabel("Workspace slots filled")
    ax.set_title("C   Ignition is a threshold, not a slope", loc="left", fontsize=8, fontweight="bold")
    _style(ax)


def _draw_jspace(ax: plt.Axes) -> None:
    """Teaching split: a small verbalizable slice in a large automatic field.

    The 6–7% note is Gurnee et al. (2026), not a number this toy measured.
    """
    ax.set_xlim(-0.6, 16.6)
    ax.set_ylim(-1.4, 16.2)
    ax.set_aspect("equal")
    ax.axis("off")
    n = 15
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (i, j),
                    0.92,
                    0.92,
                    facecolor=_AUTO,
                    edgecolor="white",
                    lw=0.4,
                )
            )
    # Privileged 3×3 in the middle.
    for i in range(6, 9):
        for j in range(6, 9):
            ax.add_patch(
                Rectangle(
                    (i, j),
                    0.92,
                    0.92,
                    facecolor=_TEAL,
                    edgecolor="white",
                    lw=0.4,
                    zorder=3,
                )
            )
    ax.text(
        7.5,
        7.5,
        "J",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold",
        zorder=4,
    )
    ax.text(7.5, -0.55, "automatic processing", ha="center", va="top", fontsize=6.5, color="0.35")
    ax.annotate(
        "verbalizable\nslice",
        xy=(8.5, 8.5),
        xytext=(12.4, 13.2),
        fontsize=6.5,
        color=_TEAL,
        arrowprops={"arrowstyle": "-|>", "color": _TEAL, "lw": 0.8},
    )
    ax.set_title(
        "D   Schematic: a small reportable slice",
        loc="left",
        fontsize=8,
        fontweight="bold",
    )


def render_paper_figure(
    result: WorkspaceResult | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if result is None:
        result = run_experiment(seed=seed)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    _draw_schematic(ax_a, result.strong)
    _draw_traces(ax_b, result.strong, result.weak)
    _draw_curve(ax_c, result)
    _draw_jspace(ax_d)
    return fig


def write_figure(
    result: WorkspaceResult | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(result, seed=seed)
    pdf = out / "global_workspace.pdf"
    png = out / "global_workspace.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    result = run_experiment(seed=seed)
    info = facts(result)
    print(
        f"weak   ignited={info['weak_ignited']}  "
        f"peak={info['weak_peak_activity']:.3f}  occupants={info['weak_occupants']}"
    )
    print(
        f"strong ignited={info['strong_ignited']}  t={info['strong_ignition_t']}  "
        f"peak={info['strong_peak_activity']:.3f}  occupants={info['strong_occupants']}"
    )
    print(info["jspace_variance_note"])
    return write_figure(result, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
