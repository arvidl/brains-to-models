"""Russell & Norvig's four definitions of AI, as a map.

A 2x2 of thought vs behaviour against human performance vs rationality.
The catalog lists evaluation criteria without assigning model scores.
Notebook 04 supplies a designed training-association exercise for discussion.

Companion: ``code/notebooks/04b_four_views_of_AI.ipynb``.
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


@dataclass(frozen=True)
class Quadrant:
    """One cell of the Russell–Norvig grid."""

    key: str
    title: str
    aima: str
    thought: bool
    human: bool
    verdict: str
    companion: str
    note: str
    color: str


QUADRANTS: tuple[Quadrant, ...] = (
    Quadrant(
        key="think-humanly",
        title="Thinking humanly",
        aima="Cognitive science",
        thought=True,
        human=True,
        verdict="compare behavior and proposed mechanisms",
        companion="04_quadrants_biases.ipynb",
        note=(
            "Score the process against how people actually think: "
            "heuristics, biases, content effects. Where a language model "
            "reproduces a published human error, test corpus exposure, "
            "generalization and alternative mechanisms. Notebook 04 shows "
            "a designed training association, not the cause of human errors."
        ),
        color="#cfe2f3",
    ),
    Quadrant(
        key="think-rationally",
        title="Thinking rationally",
        aima="Laws of thought",
        thought=True,
        human=False,
        verdict="test validity and consistency",
        companion="",
        note=(
            "Score the process against validity, not against a person. "
            "A generated reasoning trace can be checked for valid steps, "
            "consistency and agreement with independently verified results. "
            "The training objective alone does not settle reasoning ability."
        ),
        color="#d9ead3",
    ),
    Quadrant(
        key="act-humanly",
        title="Acting humanly",
        aima="Turing test",
        thought=False,
        human=True,
        verdict="specify the imitation test",
        companion="01_tiny_transformer.ipynb",
        note=(
            "Score the public interface of thought: the strings a person "
            "might produce. Results depend on the task, interaction and "
            "judges. Successful imitation does not by itself identify "
            "the underlying mechanism or establish subjective experience."
        ),
        color="#fff2cc",
    ),
    Quadrant(
        key="act-rationally",
        title="Acting rationally",
        aima="Rational agent",
        thought=False,
        human=False,
        verdict="state the objective and information available",
        companion="02b_active_inference_vs_rl.ipynb",
        note=(
            "Choose the action that maximises expected utility given "
            "goals and available information. Examine the full agent, "
            "including its model, tools, feedback and action-selection rule. "
            "Notebook 02b distinguishes model knowledge from the objective."
        ),
        color="#f4cccc",
    ),
)


def lookup(query: str) -> list[Quadrant]:
    q = query.lower().strip()
    hits = []
    for cell in QUADRANTS:
        blob = " ".join(
            [cell.key, cell.title, cell.aima, cell.verdict, cell.note, cell.companion]
        ).lower()
        if q in blob:
            hits.append(cell)
    return hits


def format_quadrant(cell: Quadrant) -> str:
    axes = (
        f"{'thought' if cell.thought else 'behaviour'} × "
        f"{'human' if cell.human else 'rational'}"
    )
    lines = [
        f"{cell.title}  ({cell.aima})",
        f"  axes     {axes}",
        f"  criterion {cell.verdict}",
    ]
    if cell.companion:
        lines.append(f"  see      code/notebooks/{cell.companion}")
    lines.append(f"  {cell.note}")
    return "\n".join(lines)


def render_paper_figure() -> plt.Figure:
    """2x2 map; pastel cells after the keynote notebook."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, ax = plt.subplots(figsize=(6.3, 3.85), layout="constrained")
    ax.set_xlim(-0.16, 1.04)
    ax.set_ylim(-0.06, 1.14)
    ax.axis("off")

    ax.text(
        0.0,
        1.08,
        "A   Four definitions of AI  (Russell & Norvig)",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="0.2",
    )

    layout = {
        "think-humanly": (0.25, 0.74),
        "think-rationally": (0.75, 0.74),
        "act-humanly": (0.25, 0.28),
        "act-rationally": (0.75, 0.28),
    }
    by_key = {c.key: c for c in QUADRANTS}
    for key, (x, y) in layout.items():
        cell = by_key[key]
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.23, y - 0.20),
                0.46,
                0.40,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=cell.color,
                edgecolor="0.55",
                linewidth=1.0,
            )
        )
        ax.text(x, y + 0.10, cell.title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(
            x,
            y - 0.01,
            f"({cell.aima})",
            ha="center",
            va="center",
            fontsize=7.5,
            style="italic",
            color="0.25",
        )
        ax.text(x, y - 0.11, cell.verdict, ha="center", va="center", fontsize=7, color="0.2")

    ax.text(0.25, 1.02, "Human performance", ha="center", va="center", fontsize=8, style="italic")
    ax.text(0.75, 1.02, "Rationality", ha="center", va="center", fontsize=8, style="italic")
    ax.text(
        -0.08,
        0.74,
        "Thought processes\nand reasoning",
        ha="center",
        va="center",
        fontsize=7.5,
        style="italic",
        rotation=90,
    )
    ax.text(
        -0.08,
        0.28,
        "Behaviour",
        ha="center",
        va="center",
        fontsize=7.5,
        style="italic",
        rotation=90,
    )
    return fig


def write_figure(out_dir: Path | str | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure()
    pdf = out / "four_views.pdf"
    png = out / "four_views.png"
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
