"""A two-lane timeline of brain metaphors and AI milestones.

The paper's opening claim, as a figure: for most of the history of the
sciences of mind the newest machine explained the brain. Foundation
models reverse the question. This module is only a catalogue and a
drawing — no training, no parameters.

Companion: ``code/notebooks/00_intro_timeline.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from brainsmodels.repro import DEFAULT_SEED, set_seed


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One dated item on the metaphor lane, the AI lane, or both."""

    year: int
    label: str
    track: str  # "metaphor" | "ai" | "both"
    title: str
    gloss: str
    note: str
    in_figure: bool = True
    figure_text: str = ""
    nudge: float = 0.0  # horizontal offset of the figure label, in years


EVENTS: tuple[Event, ...] = (
    Event(
        year=1664,
        label="c. 1664",
        track="metaphor",
        title="hydraulics",
        gloss="Descartes: animal spirits in pipes",
        note=(
            "Descartes' Treatise on Man (written in the 1630s, published 1664) "
            "pictures nerves as tubes and the pineal gland as a valve. The "
            "brain is the newest hydraulic machine. The metaphor is "
            "unidirectional: machines explain bodies; bodies do not yet "
            "suggest better machines."
        ),
        in_figure=False,
    ),
    Event(
        year=1748,
        label="1748",
        track="metaphor",
        title="clockwork",
        gloss="La Mettrie, L'Homme Machine",
        note=(
            "Julien Offray de La Mettrie argues that man is a machine — "
            "clockwork, not a soul steering a body. The claim is "
            "philosophical and medical at once. It is still one-way: the "
            "watch explains the animal."
        ),
        in_figure=False,
    ),
    Event(
        year=1870,
        label="c. 1870",
        track="metaphor",
        title="telegraph",
        gloss="Helmholtz: nerve as a telegraph wire",
        note=(
            "Helmholtz measured the speed of the nerve impulse (1850) and "
            "the late nineteenth century talked about the nervous system as "
            "a telegraph, later a telephone exchange. Signals, delays, "
            "switchboards: the brain inherits the infrastructure of the age."
        ),
        in_figure=False,
    ),
    Event(
        year=1943,
        label="1943",
        track="both",
        title="logic gates",
        gloss="McCulloch & Pitts: neurons as propositions",
        note=(
            "A neuron that fires or does not is treated as a proposition "
            "that is true or false. McCulloch and Pitts write a calculus "
            "in which networks compute logical functions. This is the "
            "shared ancestor: a brain metaphor that is also a machine, and "
            "a machine that is also a theory of mind. It sits on both lanes."
        ),
        figure_text="logic gates\nMcCulloch–Pitts",
        nudge=-2.0,
    ),
    Event(
        year=1948,
        label="1948",
        track="metaphor",
        title="feedback",
        gloss="Wiener: cybernetics, control by error",
        note=(
            "Norbert Wiener's Cybernetics makes feedback — sensing an error "
            "and acting to cancel it — the common language of animals and "
            "servomechanisms. Purpose is no longer a ghost in the machine; "
            "it is a loop. Predictive processing will later radicalise the "
            "same idea."
        ),
        figure_text="feedback\nWiener",
        nudge=3.0,
    ),
    Event(
        year=1950,
        label="1950",
        track="ai",
        title="imitation game",
        gloss="Turing: can a machine pass as a person?",
        note=(
            "Turing replaces 'can machines think?' with a behavioural test. "
            "The question is already bidirectional: not only 'is the brain "
            "a computer?' but 'could a computer be taken for us?'. The "
            "paper's later sections return to why passing the test is not "
            "the same as having a world."
        ),
        figure_text="imitation game\nTuring",
    ),
    Event(
        year=1956,
        label="1956",
        track="ai",
        title="Dartmouth",
        gloss="The summer that named 'artificial intelligence'",
        note=(
            "McCarthy, Minsky, Rochester, and Shannon convene at Dartmouth. "
            "The field gets a name and a bet: that every aspect of learning "
            "or intelligence can in principle be described so precisely "
            "that a machine can simulate it. The winter will come later."
        ),
        in_figure=False,
    ),
    Event(
        year=1958,
        label="1958",
        track="metaphor",
        title="the computer",
        gloss="von Neumann, The Computer and the Brain",
        note=(
            "von Neumann's unfinished lectures compare the digital computer "
            "to the nervous system and stress the differences (precision, "
            "architecture, reliability). The metaphor is now contemporaneous "
            "with the artefact: the same year Rosenblatt publishes the "
            "perceptron. A dotted line on the figure marks that coincidence."
        ),
        figure_text="the computer\nvon Neumann",
        nudge=2.0,
    ),
    Event(
        year=1958,
        label="1958",
        track="ai",
        title="perceptron",
        gloss="Rosenblatt: a learning machine from a simplified neuron",
        note=(
            "Frank Rosenblatt's perceptron is a trainable classifier inspired "
            "by a simplified neuron. It is the first famous case of an artefact "
            "built *from* a brain metaphor rather than only used *as* one. "
            "Minsky and Papert (1969) will later map its limits."
        ),
        figure_text="perceptron\nRosenblatt",
        nudge=2.0,
    ),
    Event(
        year=1969,
        label="1969",
        track="ai",
        title="perceptrons",
        gloss="Minsky & Papert: what a single layer cannot do",
        note=(
            "Perceptrons shows limits of the simple learning machine "
            "(XOR is the classroom example). The result is often blamed "
            "for a winter; more carefully, it is a theorem about a "
            "shallow architecture. Multi-layer nets and backprop will "
            "reopen the door."
        ),
        in_figure=False,
    ),
    Event(
        year=1982,
        label="1982",
        track="both",
        title="Hopfield",
        gloss="Memories as attractors of an energy",
        note=(
            "Hopfield writes a network whose stable states are memories, "
            "and whose dynamics descend an energy. Neuroscience and AI "
            "share a physical picture: content-addressable memory as a "
            "dynamical system. Section 5 of the paper returns to this."
        ),
        in_figure=False,
    ),
    Event(
        year=1986,
        label="1986",
        track="metaphor",
        title="PDP",
        gloss="Rumelhart & McClelland: the mind as a parallel network",
        note=(
            "The Parallel Distributed Processing volumes offer a network "
            "as a theory of cognition, not only as an engineering trick. "
            "'The brain is a neural net' becomes the default metaphor of "
            "a generation. The same year, on the other lane, back-propagation "
            "makes that net trainable."
        ),
        figure_text="PDP\nparallel mind",
        nudge=-1.5,
    ),
    Event(
        year=1986,
        label="1986",
        track="ai",
        title="backprop",
        gloss="Rumelhart, Hinton, Williams: error driven through layers",
        note=(
            "Back-propagation of error trains a multi-layer net by the "
            "chain rule. Combined with the PDP metaphor it reopens the "
            "door that Perceptrons (1969) had seemed to close. The "
            "algorithm is not a claim about how cortex learns; it is a "
            "claim about how a layered map can be fit. Notebook 01 uses "
            "the same score at character scale."
        ),
        figure_text="backprop\nRumelhart et al.",
        nudge=-1.5,
    ),
    Event(
        year=1999,
        label="1999",
        track="metaphor",
        title="predictive coding",
        gloss="Rao & Ballard: cortex as a generative model",
        note=(
            "Descending weights predict the layer below; the mismatch "
            "(prediction error) drives inference and learning. The brain "
            "is no longer a computer that waits for input. It is a model "
            "that is surprised. Notebook 02 is a toy of this paper."
        ),
        figure_text="predictive coding\nRao & Ballard",
        nudge=-2.0,
    ),
    Event(
        year=2010,
        label="2010",
        track="metaphor",
        title="free energy",
        gloss="Friston: persist by minimising surprise",
        note=(
            "The free-energy principle is the most ambitious form of the "
            "prediction metaphor: a system that lasts must keep its "
            "sensory states inside a viable bound. Perception, action, "
            "and (later) goals are one imperative. Section 3 of the paper."
        ),
        figure_text="free energy\nFriston",
        nudge=2.0,
    ),
    Event(
        year=2012,
        label="2012",
        track="ai",
        title="AlexNet",
        gloss="Deep convnets win ImageNet; scale starts to work",
        note=(
            "Krizhevsky, Sutskever, and Hinton train a deep convolutional "
            "net on GPUs and win ImageNet by a wide margin. The old "
            "connectionist bet — depth plus data plus gradient descent — "
            "suddenly pays. Neuroscience will spend the next decade asking "
            "how far the same nets go as models of ventral stream."
        ),
        figure_text="AlexNet",
    ),
    Event(
        year=2017,
        label="2017",
        track="ai",
        title="transformer",
        gloss="Vaswani et al.: attention is all you need",
        note=(
            "A stack of causal (or bidirectional) attention layers, trained "
            "to predict the next token or to denoise a sequence. No "
            "recurrence required. Notebook 01 is this object at toy scale: "
            "one layer, four heads, next-character loss. The architecture "
            "of almost every foundation model since."
        ),
        figure_text="transformer",
    ),
    Event(
        year=2020,
        label="2020",
        track="ai",
        title="GPT-3",
        gloss="In-context learning at scale",
        note=(
            "Brown et al. show that a large transformer trained on next-token "
            "prediction can do new tasks from a prompt, without weight "
            "updates. Competence looks like instruction-following; the "
            "objective has not changed. Scale makes the same score "
            "stranger."
        ),
        in_figure=False,
    ),
    Event(
        year=2021,
        label="2021",
        track="both",
        title="foundation models",
        gloss="Bommasani et al. name the class; brains and models compared",
        note=(
            "A Stanford report names 'foundation models': large nets trained "
            "by self-supervision and adapted widely. The same years, "
            "integrative modelling papers (e.g. Schrimpf, Goldstein) ask "
            "how well next-token systems predict human language responses. "
            "The comparison in this paper becomes a research programme, "
            "not only a metaphor."
        ),
        in_figure=False,
    ),
    Event(
        year=2022,
        label="2022",
        track="ai",
        title="instruction",
        gloss="InstructGPT / ChatGPT: prediction dressed as a colleague",
        note=(
            "Instruction tuning and preference optimisation (Ouyang et al., "
            "then ChatGPT) turn a next-token model into something people "
            "address in the second person. Apparent agency is a layer on "
            "the same objective. The paper's question is now public: if "
            "this is not a mind, what is?"
        ),
        figure_text="ChatGPT",
    ),
)


PREFACE = tuple(ev for ev in EVENTS if ev.year < 1940)
FIGURE_EVENTS = tuple(ev for ev in EVENTS if ev.in_figure)


def events_on(*tracks: str) -> tuple[Event, ...]:
    wanted = set(tracks)
    return tuple(ev for ev in EVENTS if ev.track in wanted or ev.track == "both")


def find_events(query: str) -> tuple[Event, ...]:
    """Case-insensitive substring match on title, gloss, label, or note."""
    q = query.strip().lower()
    if not q:
        return ()
    hits = [
        ev
        for ev in EVENTS
        if q in ev.title.lower()
        or q in ev.gloss.lower()
        or q in ev.label.lower()
        or q in ev.note.lower()
        or q in ev.track
    ]
    return tuple(hits)


def in_year_range(lo: int, hi: int, events: tuple[Event, ...] | None = None) -> tuple[Event, ...]:
    src = EVENTS if events is None else events
    return tuple(ev for ev in src if lo <= ev.year <= hi)


def format_event(ev: Event) -> str:
    lanes = {"metaphor": "brain as…", "ai": "AI", "both": "both lanes"}
    return (
        f"{ev.label}  {ev.title}  [{lanes[ev.track]}]\n"
        f"    {ev.gloss}\n\n"
        f"{ev.note}"
    )


def format_catalogue(events: tuple[Event, ...] | None = None) -> str:
    src = EVENTS if events is None else events
    lines = [f"{len(src)} events\n"]
    for ev in src:
        mark = {"metaphor": "B", "ai": "A", "both": "↔"}[ev.track]
        lines.append(f"  {ev.label:>7}  {mark}  {ev.title:<22}  {ev.gloss}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_METAPHOR = "#2c5f6e"
_AI = "#9a4a2a"
_INK = "0.2"
_RULE = "0.75"


# Label placement: (dx in years, |dy| from the lane, horizontal alignment).
# Alternating |dy| keeps neighbours from sitting on the same baseline.
_LABEL = {
    (1943, "metaphor"): (-9.0, 0.10, "right"),
    (1943, "ai"): (-9.0, 0.10, "right"),
    (1948, "metaphor"): (3.0, 0.21, "left"),
    (1950, "ai"): (-1.0, 0.21, "right"),
    (1958, "metaphor"): (0.0, 0.10, "center"),
    (1958, "ai"): (0.0, 0.10, "center"),
    (1986, "metaphor"): (0.0, 0.21, "center"),
    (1986, "ai"): (0.0, 0.21, "center"),
    (1999, "metaphor"): (-2.0, 0.10, "center"),
    (2010, "metaphor"): (4.0, 0.21, "left"),
}

_CHIP_WHO = ("Descartes", "La Mettrie", "Helmholtz")


def _stem_label(
    ax: plt.Axes,
    x_dot: float,
    y_dot: float,
    text: str,
    color: str,
    dx: float,
    dy: float,
    ha: str,
) -> None:
    x_t, y_t = x_dot + dx, y_dot + dy
    ax.plot([x_dot, x_t], [y_dot, y_t], color=color, lw=0.55, zorder=2)
    ax.text(
        x_t,
        y_t,
        text,
        ha=ha,
        va="bottom" if dy > 0 else "top",
        fontsize=6.4,
        color=color,
        linespacing=1.15,
        clip_on=False,
    )


def render_paper_figure() -> plt.Figure:
    """Two lanes, 1943–2022, with three older metaphors as chips."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, ax = plt.subplots(figsize=(6.3, 5.8), layout="constrained")

    y_b, y_a = 0.64, 0.44
    ax.set_xlim(1926.0, 2038.0)
    ax.set_ylim(0.00, 1.14)
    ax.axis("off")

    ax.text(
        0.0,
        1.07,
        "A   Each era's machine became the brain. Then the machines started answering back.",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color=_INK,
    )

    # Chips live in axes coordinates so they never sit on the year axis
    # or collide with the italic tag.
    chip_y, chip_h = 0.905, 0.085
    for i, (ev, who) in enumerate(zip(PREFACE, _CHIP_WHO, strict=True)):
        left = 0.01 + i * 0.25
        box = FancyBboxPatch(
            (left, chip_y),
            0.235,
            chip_h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.6,
            edgecolor=_METAPHOR,
            facecolor="#e8f0f2",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(box)
        ax.text(
            left + 0.1175,
            chip_y + chip_h / 2,
            f"{ev.title}  ·  {ev.label}\n{who}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.3,
            color=_METAPHOR,
            linespacing=1.2,
        )
    ax.text(
        0.80,
        chip_y + chip_h / 2,
        "unidirectional\nmetaphors",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.2,
        color="0.4",
        style="italic",
    )

    for y, color, name in (
        (y_b, _METAPHOR, "Brain as …"),
        (y_a, _AI, "AI does …"),
    ):
        ax.plot([1942, 2025], [y, y], color=_RULE, lw=0.9, zorder=1)
        ax.text(
            1940.5,
            y,
            name,
            ha="right",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

    years_b = {ev.year for ev in FIGURE_EVENTS if ev.track in ("metaphor", "both")}
    years_a = {ev.year for ev in FIGURE_EVENTS if ev.track in ("ai", "both")}
    for year in sorted(years_b & years_a):
        ax.plot([year, year], [y_a, y_b], color="0.82", lw=0.7, ls=":", zorder=0)

    drawn: set[tuple[int, str]] = set()
    for ev in FIGURE_EVENTS:
        for track, y_lane, sign in (
            ("metaphor", y_b, +1.0),
            ("ai", y_a, -1.0),
        ):
            if ev.track not in (track, "both"):
                continue
            key = (ev.year, track)
            if key in drawn:
                continue
            drawn.add(key)
            color = _METAPHOR if track == "metaphor" else _AI
            ax.scatter([ev.year], [y_lane], s=28, color=color, zorder=3, linewidths=0)
            if track == "ai" and ev.year >= 2012:
                continue
            dx, dy, ha = _LABEL.get(key, (ev.nudge, 0.12, "center"))
            _stem_label(ax, ev.year, y_lane, ev.figure_text, color, dx, sign * dy, ha)

    recent_ai = sorted(
        (ev for ev in FIGURE_EVENTS if ev.track in ("ai", "both") and ev.year >= 2012),
        key=lambda e: e.year,
    )
    x_box, w_box = 2026.2, 11.2
    y_box, h_box = y_a - 0.235, 0.195
    ax.add_patch(
        Rectangle(
            (x_box, y_box),
            w_box,
            h_box,
            linewidth=0.6,
            edgecolor=_AI,
            facecolor="#f8efe9",
            zorder=3,
            clip_on=False,
        )
    )
    # Newest at the top so leader lines fan without crossing.
    for i, ev in enumerate(reversed(recent_ai)):
        y_t = y_box + h_box - 0.038 - i * 0.055
        ax.annotate(
            f"{ev.year}   {ev.figure_text}",
            xy=(ev.year, y_a),
            xytext=(x_box + 0.6, y_t),
            ha="left",
            va="center",
            fontsize=6.5,
            color=_AI,
            zorder=4,
            arrowprops={"arrowstyle": "-", "color": _AI, "lw": 0.55, "shrinkA": 3, "shrinkB": 1},
            clip_on=False,
        )

    # Year axis sits below the lowest AI labels, not through them.
    y_axis = 0.072
    ax.plot([1941, 2025], [y_axis, y_axis], color="0.7", lw=0.6)
    for year in (1945, 1960, 1975, 1990, 2005, 2020):
        ax.plot([year, year], [y_axis, y_axis + 0.012], color="0.5", lw=0.6)
        ax.text(year, y_axis - 0.014, str(year), ha="center", va="top", fontsize=6.5, color="0.35")

    ax.text(
        0.5,
        0.012,
        "Dotted lines mark years that sit on both lanes. "
        "The chips are older: the newest machine explained the brain, and stopped there.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    return fig


def write_figure(out_dir: Path | str | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure()
    pdf = out / "intro_timeline.pdf"
    png = out / "intro_timeline.png"
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
