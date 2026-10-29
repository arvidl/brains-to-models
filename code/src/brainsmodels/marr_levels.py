"""Marr's three levels, drawn for a brain and for a machine.

The paper's opening method, as a schematic: an information-processing
system must be described at three loosely coupled levels
(Marr, *Vision*, 1982, pp. 24–25). The same table applies to a brain
and to a machine. Agreement on one row does not fill the rows below.
Theme (ii) starts from hardware and is not a fourth Marr level.

No training, no parameters — a labelled table and a drawing.

Companion: ``code/notebooks/00_intro_timeline.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from brainsmodels.repro import DEFAULT_SEED, set_seed


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Catalogue (same sentences as the figure boxes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One Marr level, with the paper's name and one example per substrate."""

    index: int
    name: str
    paper_name: str
    question: str
    brain: str
    machine: str
    coupling: str


LEVELS: tuple[Level, ...] = (
    Level(
        index=1,
        name="Computational theory",
        paper_name="function",
        question="What is computed, and why\nthat mapping is appropriate",
        brain="Example task: infer an object's location.\nSpecify the behavioral goal.",
        machine="Example task: predict the next token.\nSpecify the training objective.",
        coupling="A shared objective is possible. It does not fix the rows below.",
    ),
    Level(
        index=2,
        name="Representation and algorithm",
        paper_name="dynamics",
        question="Which internal variables,\nand how they are transformed",
        brain="Candidate codes and update mechanisms.\nTest them against neural measurements.",
        machine="Tokens, embeddings and attention.\nTraining updates differ from inference.",
        coupling="The same representation can be updated by a different algorithm.",
    ),
    Level(
        index=3,
        name="Hardware implementation",
        paper_name="architecture",
        question="How representation and algorithm\nare realized physically",
        brain="Neurons, synapses, spikes.\nEnergy, and a body.",
        machine="Processors, memory and arithmetic.\nMeasure the physical implementation.",
        coupling="Different physical substrates can realize the same upper rows.",
    ),
)

THEME_II = (
    "Bodily regulation, experience and values require additional questions "
    "and evidence. This context band is not a fourth Marr level or a "
    "test that settles machine consciousness."
)


def format_levels() -> str:
    """Plain-text table of the same content the figure draws."""
    lines = [
        "Marr's three levels (Vision, 1982, pp. 24–25).",
        "Each row is a description of the same system — brain or machine.",
        "The levels are only loosely coupled.",
        "",
    ]
    for lv in LEVELS:
        lines.append(f"{lv.index}  {lv.name}   [this paper: {lv.paper_name}]")
        lines.append(f"   {lv.question.replace(chr(10), ' ')}.")
        lines.append(f"   brain    {lv.brain.replace(chr(10), ' ')}")
        lines.append(f"   machine  {lv.machine.replace(chr(10), ' ')}")
        lines.append(f"   {lv.coupling}")
        lines.append("")
    lines.append(THEME_II)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_BRAIN = "#2c5f6e"
_MACHINE = "#9a4a2a"
_INK = "0.18"
_MUTED = "0.35"
_RULE = "0.70"
_LEVEL_FACE = "#f4f1ea"
_BRAIN_FACE = "#e8f0f2"
_MACHINE_FACE = "#f8efe9"
_THEME_FACE = "#f3ead8"


FIGURE_NAMES: tuple[str, ...] = (
    "1  Computational\n    theory",
    "2  Representation\n    and algorithm",
    "3  Hardware\n    implementation",
)


def _round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    lw: float = 0.8,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            clip_on=False,
        )
    )


def _dashed_down(ax: plt.Axes, x: float, y0: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={
            "arrowstyle": "->",
            "color": _RULE,
            "lw": 0.7,
            "ls": (0, (2.2, 1.6)),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=1,
    )


def render_paper_figure() -> plt.Figure:
    """Three Marr rows, two substrates, a band that is not a fourth level."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, ax = plt.subplots(figsize=(6.3, 5.70))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.97, bottom=0.03)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.02,
        0.975,
        "A   Three levels of an information-processing system  (Marr, 1982)",
        fontsize=8,
        fontweight="bold",
        va="top",
        color=_INK,
    )
    ax.text(
        0.02,
        0.938,
        "The same table describes a brain or a machine. "
        "Dashed arrows: agreement on one row does not fill the rows below.",
        fontsize=6.5,
        va="top",
        color=_MUTED,
    )

    x_lvl, w_lvl = 0.02, 0.28
    x_br, w_col = 0.32, 0.325
    x_mc = 0.655
    ax.text(x_br + w_col / 2, 0.905, "Brain", ha="center", va="center", fontsize=8, fontweight="bold", color=_BRAIN)
    ax.text(
        x_br + w_col / 2,
        0.882,
        "biological substrate",
        ha="center",
        va="center",
        fontsize=6.2,
        style="italic",
        color=_BRAIN,
    )
    ax.text(x_mc + w_col / 2, 0.905, "Machine", ha="center", va="center", fontsize=8, fontweight="bold", color=_MACHINE)
    ax.text(
        x_mc + w_col / 2,
        0.882,
        "artificial substrate",
        ha="center",
        va="center",
        fontsize=6.2,
        style="italic",
        color=_MACHINE,
    )

    # Bottoms of the three Marr rows (computational → hardware).
    row_y = (0.675, 0.430, 0.185)
    row_h = 0.168

    for i, lv in enumerate(LEVELS):
        y = row_y[i]
        _round_box(ax, x_lvl, y, w_lvl, row_h, facecolor=_LEVEL_FACE, edgecolor="0.55")
        _round_box(ax, x_br, y, w_col, row_h, facecolor=_BRAIN_FACE, edgecolor=_BRAIN)
        _round_box(ax, x_mc, y, w_col, row_h, facecolor=_MACHINE_FACE, edgecolor=_MACHINE)

        ax.text(
            x_lvl + 0.012,
            y + row_h - 0.016,
            FIGURE_NAMES[i],
            ha="left",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            color=_INK,
            linespacing=1.2,
        )
        ax.text(
            x_lvl + 0.012,
            y + 0.062,
            f"this paper: {lv.paper_name}",
            ha="left",
            va="center",
            fontsize=6.1,
            style="italic",
            color=_MUTED,
        )
        ax.text(
            x_lvl + 0.012,
            y + 0.014,
            lv.question,
            ha="left",
            va="bottom",
            fontsize=5.8,
            color=_INK,
            linespacing=1.2,
        )
        ax.text(
            x_br + w_col / 2,
            y + row_h / 2,
            lv.brain,
            ha="center",
            va="center",
            fontsize=6.4,
            color=_BRAIN,
            linespacing=1.28,
        )
        ax.text(
            x_mc + w_col / 2,
            y + row_h / 2,
            lv.machine,
            ha="center",
            va="center",
            fontsize=6.4,
            color=_MACHINE,
            linespacing=1.28,
        )

        if i < len(LEVELS) - 1:
            y_from = y - 0.006
            y_to = row_y[i + 1] + row_h + 0.018
            _dashed_down(ax, x_br + w_col / 2, y_from, y_to)
            _dashed_down(ax, x_mc + w_col / 2, y_from, y_to)
            ax.text(
                0.318,
                (y_from + y_to) / 2,
                "loosely\ncoupled",
                ha="right",
                va="center",
                fontsize=5.6,
                color=_MUTED,
                style="italic",
                linespacing=1.15,
            )

    y_th, h_th = 0.018, 0.128
    _round_box(ax, 0.02, y_th, 0.96, h_th, facecolor=_THEME_FACE, edgecolor="#b0894a", lw=0.7)
    ax.text(
        0.035,
        y_th + h_th - 0.014,
        "Not a Marr level",
        ha="left",
        va="top",
        fontsize=6.4,
        fontweight="bold",
        color="#6b4f1d",
    )
    ax.text(
        0.035,
        y_th + 0.014,
        "Bodily regulation, experience and values raise additional questions.\n"
        "This context band is not a fourth Marr level or a test that\n"
        "settles machine consciousness.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#6b4f1d",
        linespacing=1.25,
    )
    return fig


def write_figure(out_dir: Path | str | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure()
    pdf = out / "marr_levels.pdf"
    png = out / "marr_levels.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    return write_figure(out_dir)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
