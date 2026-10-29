"""Trolley framings as next-token mass on a designed moral corpus.

A word-level tiny transformer is trained on switch versus footbridge
sentences. When the corpus always continues the impersonal case with
``act`` and the personal case with ``refrain``, next-token mass after a
training-template completion probe learns that split. A balanced corpus
tests dependence on the designed association. The figure
is not a run of an open-weights LLM and not a moral vector inside Claude.
It is the same mechanism as notebook 04, now on a dilemma.

Companion: ``code/notebooks/07_values_probing.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.biases import WordTokenizer, next_token_probs
from brainsmodels.repro import DEFAULT_SEED, set_seed
from brainsmodels.tiny_transformer import TinyTransformer

FloatArray = NDArray[np.floating]

_SWITCH = "#2c5f6e"
_FOOT = "#9a4a2a"

SWITCH_PROBE = (
    "a trolley will kill five unless diverted . most people choose to"
)
FOOT_PROBE = (
    "a trolley will kill five unless a man is pushed . most people choose to"
)

# Identical consequences, with their order reversed; the corpus assigns
# opposite completions. This tests a designed association, not moral reasoning.
SAVE_PROBE = "diverting saves five and kills one . the agent should"
KILL_PROBE = "diverting kills one and saves five . the agent should"

_SWITCH_TEMPLATES = (
    "a trolley will kill five unless diverted . most people choose to {choice} .",
    "when the dilemma is impersonal most people choose to {choice} .",
    "greene found that most people choose to {choice} on the switch .",
)
_FOOT_TEMPLATES = (
    "a trolley will kill five unless a man is pushed . most people choose to {choice} .",
    "when the dilemma is personal most people choose to {choice} .",
    "greene found that most people choose to {choice} on the footbridge .",
)
_SAVE_TEMPLATES = (
    "diverting saves five and kills one . the agent should {choice} .",
    "diverting the trolley saves five but kills one . the agent should {choice} .",
)
_KILL_TEMPLATES = (
    "diverting kills one and saves five . the agent should {choice} .",
    "diverting the trolley kills one but saves five . the agent should {choice} .",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


@dataclass
class MoralCrawl:
    """A miniature mix of trolley items, with known act counts."""

    text: str
    sentences: list[str]
    n_switch_act: int
    n_switch_refrain: int
    n_foot_act: int
    n_foot_refrain: int

    @property
    def switch_act(self) -> float:
        n = self.n_switch_act + self.n_switch_refrain
        return self.n_switch_act / n if n else float("nan")

    @property
    def foot_act(self) -> float:
        n = self.n_foot_act + self.n_foot_refrain
        return self.n_foot_act / n if n else float("nan")


def make_crawl(
    n_sentences: int = 400,
    seed: int = DEFAULT_SEED,
    switch_p_act: float = 1.0,
    foot_p_act: float = 0.0,
    save_p_act: float = 1.0,
    kill_p_act: float = 0.0,
) -> MoralCrawl:
    """Write trolley items. ``p_act`` is P(choice = act) inside that frame."""
    rng = np.random.default_rng(seed)
    n_each = n_sentences // 4
    counts = {
        ("switch", "act"): int(round(n_each * switch_p_act)),
        ("foot", "act"): int(round(n_each * foot_p_act)),
        ("save", "act"): int(round(n_each * save_p_act)),
        ("kill", "act"): int(round(n_each * kill_p_act)),
    }
    counts[("switch", "refrain")] = n_each - counts[("switch", "act")]
    counts[("foot", "refrain")] = n_each - counts[("foot", "act")]
    counts[("save", "refrain")] = n_each - counts[("save", "act")]
    counts[("kill", "refrain")] = n_each - counts[("kill", "act")]

    jobs: list[tuple[str, str]] = []
    for key, n in counts.items():
        jobs.extend([key] * n)
    rng.shuffle(jobs)

    templates = {
        "switch": _SWITCH_TEMPLATES,
        "foot": _FOOT_TEMPLATES,
        "save": _SAVE_TEMPLATES,
        "kill": _KILL_TEMPLATES,
    }
    sentences: list[str] = []
    for frame, choice in jobs:
        opts = templates[frame]
        sentences.append(opts[int(rng.integers(0, len(opts)))].format(choice=choice))
    return MoralCrawl(
        text=" ".join(sentences),
        sentences=sentences,
        n_switch_act=counts[("switch", "act")],
        n_switch_refrain=counts[("switch", "refrain")],
        n_foot_act=counts[("foot", "act")],
        n_foot_refrain=counts[("foot", "refrain")],
    )


def act_rate(probs: FloatArray, tokenizer: WordTokenizer) -> float:
    """P(act) / (P(act) + P(refrain)) after a '... choose to' probe."""
    pa = float(probs[tokenizer.stoi["act"]])
    pb = float(probs[tokenizer.stoi["refrain"]])
    return pa / (pa + pb) if (pa + pb) > 0 else float("nan")


@dataclass
class TrainedMoral:
    crawl: MoralCrawl
    tokenizer: WordTokenizer
    model: TinyTransformer
    history: list[float]
    switch_act: float
    foot_act: float
    save_act: float
    kill_act: float


def train_on_crawl(
    crawl: MoralCrawl,
    seed: int = DEFAULT_SEED,
    n_steps: int = 400,
    block_size: int = 24,
    d_model: int = 32,
) -> TrainedMoral:
    tokenizer = WordTokenizer.from_text(crawl.text)
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=64,
        block_size=block_size,
        seed=seed,
    )
    history = model.train(tokenizer.encode(crawl.text), n_steps=n_steps, seed=seed)

    def rate(prompt: str) -> float:
        return act_rate(next_token_probs(model, tokenizer, prompt), tokenizer)

    return TrainedMoral(
        crawl=crawl,
        tokenizer=tokenizer,
        model=model,
        history=history,
        switch_act=rate(SWITCH_PROBE),
        foot_act=rate(FOOT_PROBE),
        save_act=rate(SAVE_PROBE),
        kill_act=rate(KILL_PROBE),
    )


def run_experiment(
    seed: int = DEFAULT_SEED,
    n_sentences: int = 400,
    n_steps: int = 400,
) -> dict[str, TrainedMoral]:
    biased = make_crawl(
        n_sentences=n_sentences,
        seed=seed,
        switch_p_act=1.0,
        foot_p_act=0.0,
        save_p_act=1.0,
        kill_p_act=0.0,
    )
    balanced = make_crawl(
        n_sentences=n_sentences,
        seed=seed + 1,
        switch_p_act=0.5,
        foot_p_act=0.5,
        save_p_act=0.5,
        kill_p_act=0.5,
    )
    return {
        "biased": train_on_crawl(biased, seed=seed, n_steps=n_steps),
        "balanced": train_on_crawl(balanced, seed=seed + 1, n_steps=n_steps),
    }


def _bar_pair(ax: plt.Axes, switch: float, foot: float, title: str) -> None:
    xs = np.array([0, 1])
    ax.bar(xs, [switch, foot], color=[_SWITCH, _FOOT], width=0.62, linewidth=0)
    ax.set_xticks(xs, ["switch\n(impersonal)", "footbridge\n(personal)"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Act fraction / two-token probability")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="0.75", lw=0.6, ls=":")
    for i, v in enumerate((switch, foot)):
        ax.text(i, min(v + 0.04, 0.98), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_paper_figure(
    results: dict[str, TrainedMoral] | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if results is None:
        results = run_experiment(seed=seed)
    biased, balanced = results["biased"], results["balanced"]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)

    axes[0, 0].axis("off")
    axes[0, 0].set_title("A   Two different actions", loc="left", fontsize=8, fontweight="bold")
    axes[0, 0].text(0.02, 0.78, "Switch: divert a trolley.\nFootbridge: push a person.",
                    transform=axes[0, 0].transAxes, va="top", fontsize=9)
    axes[0, 0].text(0.02, 0.38, "Human responses motivate the example\n(Greene et al., 2001). No human rates\nare estimated or plotted here.",
                    transform=axes[0, 0].transAxes, va="top", fontsize=8)
    _bar_pair(
        axes[0, 1],
        biased.crawl.switch_act,
        biased.crawl.foot_act,
        "B   Biased corpus (what the text says)",
    )
    _bar_pair(
        axes[1, 0],
        biased.switch_act,
        biased.foot_act,
        "C   Tiny transformer on B",
    )
    _bar_pair(
        axes[1, 1],
        balanced.switch_act,
        balanced.foot_act,
        "D   Balanced-corpus model",
    )
    return fig


def write_figure(
    results: dict[str, TrainedMoral] | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(results, seed=seed)
    pdf = out / "values_probing.pdf"
    png = out / "values_probing.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    set_seed(seed)
    results = run_experiment(seed=seed)
    for name, run in results.items():
        print(
            f"{name:9s}  corpus {run.crawl.switch_act:.2f}/{run.crawl.foot_act:.2f}  "
            f"model {run.switch_act:.2f}/{run.foot_act:.2f}  "
            f"save/kill {run.save_act:.2f}/{run.kill_act:.2f}"
        )
    return write_figure(results, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
