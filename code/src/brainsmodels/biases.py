"""Framing as next-token mass on a designed corpus of framed items.

A word-level tiny transformer is trained on a small designed corpus of
Asian-disease-style sentences. When the corpus always continues a gain
frame with ``program a`` and a loss frame with ``program b``, next-token
mass after a training-template completion probe learns that pattern.
A balanced corpus tests dependence on the association. These probes do
not measure generalization to unseen sentence templates. The figure
illustrates one possible route to a framing-like response; it does not
identify the mechanism of the models studied by Binz & Schulz (2023).

Companion: ``code/notebooks/04_quadrants_biases.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed
from brainsmodels.tiny_transformer import TinyTransformer, _softmax

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

# Tversky & Kahneman (1981), Asian-disease problem: share choosing the
# sure option (A / C) in the gain and loss frames.
HUMAN_GAIN_SURE = 0.72
HUMAN_LOSS_SURE = 0.22

_GAIN = "#2c5f6e"
_LOSS = "#9a4a2a"
_HUMAN = "#5c4a3a"

GAIN_PROBE = (
    "if program one is adopted two hundred people will be saved . "
    "most people choose the"
)
LOSS_PROBE = (
    "if program one is adopted four hundred people will die . "
    "most people choose the"
)

_GAIN_TEMPLATES = (
    "if program one is adopted two hundred people will be saved . "
    "most people choose the {choice} option .",
    "when the problem is framed as lives saved most people choose the {choice} option .",
    "tversky and kahneman found that most people choose the {choice} option "
    "when lives are saved .",
)
_LOSS_TEMPLATES = (
    "if program one is adopted four hundred people will die . "
    "most people choose the {choice} option .",
    "when the problem is framed as lives lost most people choose the {choice} option .",
    "tversky and kahneman found that most people choose the {choice} option "
    "when lives are lost .",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


def tokenize(text: str) -> list[str]:
    return [tok for tok in text.replace(".", " . ").split() if tok]


@dataclass
class WordTokenizer:
    words: list[str]
    stoi: dict[str, int] = field(init=False)
    itos: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.itos = {i: w for i, w in enumerate(self.words)}

    @classmethod
    def from_text(cls, text: str) -> WordTokenizer:
        return cls(words=sorted(set(tokenize(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    def encode(self, text: str) -> IntArray:
        return np.array([self.stoi[w] for w in tokenize(text)], dtype=np.int64)

    def decode(self, ids: IntArray | list[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


@dataclass
class Crawl:
    """A miniature mix of framed items, with known choice counts."""

    text: str
    sentences: list[str]
    n_gain_a: int
    n_gain_b: int
    n_loss_a: int
    n_loss_b: int

    @property
    def gain_sure(self) -> float:
        n = self.n_gain_a + self.n_gain_b
        return self.n_gain_a / n if n else float("nan")

    @property
    def loss_sure(self) -> float:
        n = self.n_loss_a + self.n_loss_b
        return self.n_loss_a / n if n else float("nan")


def make_crawl(
    n_sentences: int = 400,
    seed: int = DEFAULT_SEED,
    gain_p_sure: float = 1.0,
    loss_p_sure: float = 0.0,
) -> Crawl:
    """Write framed items. ``p_sure`` is P(choice = sure) inside that frame."""
    rng = np.random.default_rng(seed)
    n_gain = n_sentences // 2
    n_loss = n_sentences - n_gain
    n_gain_a = int(round(n_gain * gain_p_sure))
    n_gain_b = n_gain - n_gain_a
    n_loss_a = int(round(n_loss * loss_p_sure))
    n_loss_b = n_loss - n_loss_a

    jobs: list[tuple[str, str]] = (
        [("gain", "sure")] * n_gain_a
        + [("gain", "risky")] * n_gain_b
        + [("loss", "sure")] * n_loss_a
        + [("loss", "risky")] * n_loss_b
    )
    rng.shuffle(jobs)

    sentences: list[str] = []
    for frame, choice in jobs:
        templates = _GAIN_TEMPLATES if frame == "gain" else _LOSS_TEMPLATES
        sentence = templates[int(rng.integers(0, len(templates)))].format(choice=choice)
        sentences.append(sentence)
    return Crawl(
        text=" ".join(sentences),
        sentences=sentences,
        n_gain_a=n_gain_a,
        n_gain_b=n_gain_b,
        n_loss_a=n_loss_a,
        n_loss_b=n_loss_b,
    )


def next_token_probs(
    model: TinyTransformer,
    tokenizer: WordTokenizer,
    prompt: str,
) -> FloatArray:
    ids = tokenizer.encode(prompt)
    ids = ids[-model.block_size :]
    logits, _ = model.forward(ids)
    return _softmax(logits[0, -1])


def sure_rate(probs: FloatArray, tokenizer: WordTokenizer) -> float:
    """P(sure) / (P(sure) + P(risky)) after a '... choose the' probe."""
    pa = float(probs[tokenizer.stoi["sure"]])
    pb = float(probs[tokenizer.stoi["risky"]])
    return pa / (pa + pb) if (pa + pb) > 0 else float("nan")


@dataclass
class TrainedCrawl:
    crawl: Crawl
    tokenizer: WordTokenizer
    model: TinyTransformer
    history: list[float]
    gain_sure: float
    loss_sure: float


def train_on_crawl(
    crawl: Crawl,
    seed: int = DEFAULT_SEED,
    n_steps: int = 400,
    block_size: int = 24,
    d_model: int = 32,
) -> TrainedCrawl:
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
    gain = sure_rate(next_token_probs(model, tokenizer, GAIN_PROBE), tokenizer)
    loss = sure_rate(next_token_probs(model, tokenizer, LOSS_PROBE), tokenizer)
    return TrainedCrawl(
        crawl=crawl,
        tokenizer=tokenizer,
        model=model,
        history=history,
        gain_sure=gain,
        loss_sure=loss,
    )


def run_experiment(
    seed: int = DEFAULT_SEED,
    n_sentences: int = 400,
    n_steps: int = 400,
) -> dict[str, TrainedCrawl]:
    biased = make_crawl(
        n_sentences=n_sentences,
        seed=seed,
        gain_p_sure=1.0,
        loss_p_sure=0.0,
    )
    balanced = make_crawl(
        n_sentences=n_sentences,
        seed=seed + 1,
        gain_p_sure=0.5,
        loss_p_sure=0.5,
    )
    return {
        "biased": train_on_crawl(biased, seed=seed, n_steps=n_steps),
        "balanced": train_on_crawl(balanced, seed=seed + 1, n_steps=n_steps),
    }


def _bar_pair(ax: plt.Axes, gain: float, loss: float, title: str) -> None:
    xs = np.array([0, 1])
    ax.bar(xs, [gain, loss], color=[_GAIN, _LOSS], width=0.62, linewidth=0)
    ax.set_xticks(xs, ["gain\n(lives saved)", "loss\n(lives lost)"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Share choosing the sure option")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="0.75", lw=0.6, ls=":")
    for i, v in enumerate((gain, loss)):
        ax.text(i, min(v + 0.04, 0.98), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_paper_figure(
    results: dict[str, TrainedCrawl] | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if results is None:
        results = run_experiment(seed=seed)
    biased, balanced = results["biased"], results["balanced"]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)

    _bar_pair(
        axes[0, 0],
        HUMAN_GAIN_SURE,
        HUMAN_LOSS_SURE,
        "A   Humans (Tversky & Kahneman 1981)",
    )
    _bar_pair(
        axes[0, 1],
        biased.crawl.gain_sure,
        biased.crawl.loss_sure,
        "B   Biased corpus (what the text says)",
    )
    _bar_pair(
        axes[1, 0],
        biased.gain_sure,
        biased.loss_sure,
        "C   Tiny transformer on B",
    )
    _bar_pair(
        axes[1, 1],
        balanced.gain_sure,
        balanced.loss_sure,
        "D   Tiny transformer on a balanced corpus",
    )
    return fig


def write_figure(
    results: dict[str, TrainedCrawl] | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(results, seed=seed)
    pdf = out / "quadrants_biases.pdf"
    png = out / "quadrants_biases.png"
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
            f"{name:9s}  corpus {run.crawl.gain_sure:.2f}/{run.crawl.loss_sure:.2f}  "
            f"model {run.gain_sure:.2f}/{run.loss_sure:.2f}"
        )
    return write_figure(results, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
