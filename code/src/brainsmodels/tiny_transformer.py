"""One-layer character-level transformer with causal self-attention.

The smallest model that still *is* a transformer: token + position embeddings,
multi-head causal attention, a tiny FFN, next-character prediction. Trained on
a tiny mix of web-like frames so attention maps stay readable.

Reference: Vaswani et al., NeurIPS (2017).
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Corpus and tokenizer
# ---------------------------------------------------------------------------

_ANIMALS = ("cat", "dog", "fox")
_OBJECTS = ("mat", "hat", "log", "box")
_PLACES = ("paris", "oslo", "tokyo")

# A small designed corpus: the same nouns show up in the genres a web
# corpus actually mixes — story, headline, wiki, FAQ — so one layer
# can still learn, and the attention maps stay readable.
_FRAMES = (
    "the {a} sat on the {o}.",
    "once upon a time the {a} sat on the {o}.",
    "breaking from {p}: the {a} sat on the {o}.",
    "according to wikipedia the {a} sat on the {o}.",
    "q: where did the {a} sit? a: on the {o}.",
)


def make_corpus(n_sentences: int = 400, seed: int = DEFAULT_SEED) -> str:
    """Web-like frames that still share 'the cat sat on the mat'."""
    rng = np.random.default_rng(seed)
    parts: list[str] = []
    for _ in range(n_sentences):
        a = _ANIMALS[int(rng.integers(0, len(_ANIMALS)))]
        o = _OBJECTS[int(rng.integers(0, len(_OBJECTS)))]
        p = _PLACES[int(rng.integers(0, len(_PLACES)))]
        frame = _FRAMES[int(rng.integers(0, len(_FRAMES)))]
        parts.append(frame.format(a=a, o=o, p=p) + " ")
    return "".join(parts)


@dataclass
class CharTokenizer:
    chars: list[str]
    stoi: dict[str, int] = field(init=False)
    itos: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(chars=sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> IntArray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids: IntArray | list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Tiny causal transformer (NumPy)
# ---------------------------------------------------------------------------


def _softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: FloatArray, eps: float = 1e-5) -> tuple[FloatArray, FloatArray, FloatArray]:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + eps)
    return xhat, mean, var


@dataclass
class TinyTransformer:
    vocab_size: int
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    block_size: int = 32
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        rng = np.random.default_rng(self.seed)
        d, h = self.d_model, self.n_heads
        scale = 0.02
        self.E = rng.normal(0.0, scale, (self.vocab_size, d))
        self.P = rng.normal(0.0, scale, (self.block_size, d))
        self.Wq = rng.normal(0.0, scale, (d, d))
        self.Wk = rng.normal(0.0, scale, (d, d))
        self.Wv = rng.normal(0.0, scale, (d, d))
        self.Wo = rng.normal(0.0, scale, (d, d))
        self.W1 = rng.normal(0.0, scale, (d, self.d_ff))
        self.b1 = np.zeros(self.d_ff)
        self.W2 = rng.normal(0.0, scale, (self.d_ff, d))
        self.b2 = np.zeros(d)
        self.g1 = np.ones(d)
        self.b_ln1 = np.zeros(d)
        self.g2 = np.ones(d)
        self.b_ln2 = np.zeros(d)
        self.Wout = rng.normal(0.0, scale, (d, self.vocab_size))
        self.bout = np.zeros(self.vocab_size)
        self._d_head = d // h
        self._cache: dict[str, object] = {}

    def parameters(self) -> list[FloatArray]:
        return [
            self.E, self.P, self.Wq, self.Wk, self.Wv, self.Wo,
            self.W1, self.b1, self.W2, self.b2,
            self.g1, self.b_ln1, self.g2, self.b_ln2,
            self.Wout, self.bout,
        ]

    def _split_heads(self, x: FloatArray) -> FloatArray:
        b, t, _ = x.shape
        x = x.reshape(b, t, self.n_heads, self._d_head)
        return np.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: FloatArray) -> FloatArray:
        x = np.transpose(x, (0, 2, 1, 3))
        b, t, h, dh = x.shape
        return x.reshape(b, t, h * dh)

    def forward(
        self,
        idx: IntArray,
        record_attn: bool = False,
    ) -> tuple[FloatArray, FloatArray | None]:
        if idx.ndim == 1:
            idx = idx[None, :]
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")

        tok = self.E[idx]
        pos = self.P[:t]
        x0 = tok + pos

        xhat1, _, _ = _layernorm(x0)
        ln1 = xhat1 * self.g1 + self.b_ln1
        q = self._split_heads(ln1 @ self.Wq)
        k = self._split_heads(ln1 @ self.Wk)
        v = self._split_heads(ln1 @ self.Wv)
        scale = np.sqrt(self._d_head)
        scores = np.einsum("bhtd,bhsd->bhts", q, k) / scale
        causal = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores = np.where(causal, -1e9, scores)
        attn = _softmax(scores, axis=-1)
        y = np.einsum("bhts,bhsd->bhtd", attn, v)
        attn_out = self._merge_heads(y) @ self.Wo
        x1 = x0 + attn_out

        xhat2, _, _ = _layernorm(x1)
        ln2 = xhat2 * self.g2 + self.b_ln2
        ff = np.maximum(ln2 @ self.W1 + self.b1, 0.0)
        x2 = x1 + ff @ self.W2 + self.b2
        logits = x2 @ self.Wout + self.bout

        self._cache = {
            "idx": idx,
            "x0": x0,
            "xhat1": xhat1,
            "ln1": ln1,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "y": y,
            "attn_out": attn_out,
            "x1": x1,
            "xhat2": xhat2,
            "ln2": ln2,
            "ff": ff,
            "x2": x2,
            "logits": logits,
        }
        return logits, attn if record_attn else None

    def loss_and_backward(self, idx: IntArray, targets: IntArray) -> float:
        logits, _ = self.forward(idx)
        if targets.ndim == 1:
            targets = targets[None, :]
        b, t, v = logits.shape
        flat = logits.reshape(-1, v)
        probs = _softmax(flat, axis=-1)
        y = targets.reshape(-1)
        nll = -np.log(np.clip(probs[np.arange(b * t), y], 1e-9, 1.0))
        loss = float(nll.mean())

        dlogits = probs
        dlogits[np.arange(b * t), y] -= 1.0
        dlogits = dlogits.reshape(b, t, v) / (b * t)
        self._backward(dlogits)
        return loss

    def _backward(self, dlogits: FloatArray) -> None:
        c = self._cache
        idx: IntArray = c["idx"]  # type: ignore[assignment]
        x0: FloatArray = c["x0"]  # type: ignore[assignment]
        xhat1: FloatArray = c["xhat1"]  # type: ignore[assignment]
        ln1: FloatArray = c["ln1"]  # type: ignore[assignment]
        q: FloatArray = c["q"]  # type: ignore[assignment]
        k: FloatArray = c["k"]  # type: ignore[assignment]
        v: FloatArray = c["v"]  # type: ignore[assignment]
        attn: FloatArray = c["attn"]  # type: ignore[assignment]
        y: FloatArray = c["y"]  # type: ignore[assignment]
        x1: FloatArray = c["x1"]  # type: ignore[assignment]
        xhat2: FloatArray = c["xhat2"]  # type: ignore[assignment]
        ln2: FloatArray = c["ln2"]  # type: ignore[assignment]
        ff: FloatArray = c["ff"]  # type: ignore[assignment]
        x2: FloatArray = c["x2"]  # type: ignore[assignment]
        b, t, _ = x0.shape
        scale = np.sqrt(self._d_head)

        self.dWout = np.einsum("btd,btv->dv", x2, dlogits)
        self.dbout = dlogits.sum(axis=(0, 1))
        dx2 = dlogits @ self.Wout.T

        dff_out = dx2
        self.dW2 = np.einsum("bth,btd->hd", ff, dff_out)
        self.db2 = dff_out.sum(axis=(0, 1))
        dff = (dff_out @ self.W2.T) * (ff > 0)
        self.dW1 = np.einsum("btd,bth->dh", ln2, dff)
        self.db1 = dff.sum(axis=(0, 1))
        dln2 = dff @ self.W1.T
        self.dg2 = (xhat2 * dln2).sum(axis=(0, 1))
        self.db_ln2 = dln2.sum(axis=(0, 1))
        dxhat2 = dln2 * self.g2
        dx1_from_ln = _layernorm_backward(x1, xhat2, dxhat2)
        dx1 = dx2 + dx1_from_ln

        dattn_out = dx1
        merged = self._merge_heads(y)
        self.dWo = np.einsum("btd,bte->de", merged, dattn_out)
        dmerged = dattn_out @ self.Wo.T
        dy = self._split_heads(dmerged)

        dv = np.einsum("bhts,bhtd->bhsd", attn, dy)
        dattn = np.einsum("bhtd,bhsd->bhts", dy, v)
        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dq = np.einsum("bhts,bhsd->bhtd", dscores, k) / scale
        dk = np.einsum("bhts,bhtd->bhsd", dscores, q) / scale

        dq_flat = self._merge_heads(dq)
        dk_flat = self._merge_heads(dk)
        dv_flat = self._merge_heads(dv)
        self.dWq = np.einsum("btd,bte->de", ln1, dq_flat)
        self.dWk = np.einsum("btd,bte->de", ln1, dk_flat)
        self.dWv = np.einsum("btd,bte->de", ln1, dv_flat)
        dln1 = dq_flat @ self.Wq.T + dk_flat @ self.Wk.T + dv_flat @ self.Wv.T

        self.dg1 = (xhat1 * dln1).sum(axis=(0, 1))
        self.db_ln1 = dln1.sum(axis=(0, 1))
        dxhat1 = dln1 * self.g1
        dx0_from_ln = _layernorm_backward(x0, xhat1, dxhat1)
        dx0 = dx1 + dx0_from_ln

        self.dP = dx0.sum(axis=0)
        if t < self.block_size:
            pad = np.zeros((self.block_size - t, self.d_model))
            self.dP = np.concatenate([self.dP, pad], axis=0)
        self.dE = np.zeros_like(self.E)
        np.add.at(self.dE, idx, dx0)

    def _grads(self) -> list[FloatArray]:
        return [
            self.dE, self.dP, self.dWq, self.dWk, self.dWv, self.dWo,
            self.dW1, self.db1, self.dW2, self.db2,
            self.dg1, self.db_ln1, self.dg2, self.db_ln2,
            self.dWout, self.dbout,
        ]

    def train(
        self,
        tokens: IntArray,
        n_steps: int = 500,
        batch_size: int = 16,
        lr: float = 3e-3,
        seed: int = DEFAULT_SEED,
        log_every: int = 50,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        n = len(tokens) - self.block_size - 1
        if n <= 0:
            raise ValueError("corpus shorter than block_size")
        params = self.parameters()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.99, 1e-8
        history: list[float] = []
        for step in range(1, n_steps + 1):
            starts = rng.integers(0, n, size=batch_size)
            idx = np.stack([tokens[s : s + self.block_size] for s in starts])
            tgt = np.stack([tokens[s + 1 : s + 1 + self.block_size] for s in starts])
            loss = self.loss_and_backward(idx, tgt)
            grads = self._grads()
            total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
            if total > 1.0:
                grads = [g * (1.0 / total) for g in grads]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * g
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
                mhat = m[i] / (1.0 - beta1**step)
                vhat = v[i] / (1.0 - beta2**step)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
            history.append(loss)
            if step == 1 or step % log_every == 0 or step == n_steps:
                print(f"step {step:04d}/{n_steps}  loss {loss:.4f}")
        return history

    def next_char_probs(self, prompt: str, tokenizer: CharTokenizer) -> FloatArray:
        ids = tokenizer.encode(prompt[-self.block_size :])
        logits, _ = self.forward(ids)
        return _softmax(logits[0, -1])

    def attention(self, prompt: str, tokenizer: CharTokenizer) -> FloatArray:
        ids = tokenizer.encode(prompt[-self.block_size :])
        _, attn = self.forward(ids, record_attn=True)
        assert attn is not None
        return attn[0]  # (H, T, T)

    def generate(
        self,
        prompt: str,
        tokenizer: CharTokenizer,
        n_tokens: int = 40,
        temperature: float = 0.8,
        seed: int = DEFAULT_SEED,
    ) -> str:
        rng = np.random.default_rng(seed)
        text = prompt
        for _ in range(n_tokens):
            probs = self.next_char_probs(text, tokenizer)
            if temperature <= 0:
                nxt = int(np.argmax(probs))
            else:
                logits = np.log(np.clip(probs, 1e-9, 1.0)) / temperature
                probs = _softmax(logits)
                nxt = int(rng.choice(self.vocab_size, p=probs))
            text += tokenizer.decode([nxt])
        return text


def _layernorm_backward(
    x: FloatArray,
    xhat: FloatArray,
    dxhat: FloatArray,
    eps: float = 1e-5,
) -> FloatArray:
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    dxhat_mean = dxhat.mean(axis=-1, keepdims=True)
    dxhat_xhat_mean = (dxhat * xhat).mean(axis=-1, keepdims=True)
    return (dxhat - dxhat_mean - xhat * dxhat_xhat_mean) * inv


# ---------------------------------------------------------------------------
# Paper figure
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "the cat sat on the "


def train_demo(
    seed: int = DEFAULT_SEED,
    n_steps: int = 500,
) -> tuple[TinyTransformer, CharTokenizer, str, list[float]]:
    text = make_corpus(seed=seed)
    tokenizer = CharTokenizer.from_text(text)
    model = TinyTransformer(vocab_size=tokenizer.vocab_size, seed=seed)
    history = model.train(tokenizer.encode(text), n_steps=n_steps, seed=seed)
    return model, tokenizer, text, history


def render_paper_figure(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    history: list[float],
    prompt: str = DEFAULT_PROMPT,
) -> plt.Figure:
    attn = model.attention(prompt, tokenizer)
    probs = model.next_char_probs(prompt, tokenizer)
    chars = list(prompt)
    t = len(chars)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(6.3, 7.6), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.06)
    sub_a, bottom = fig.subfigures(2, 1, height_ratios=[2.35, 1.0])
    sub_b, sub_c = bottom.subfigures(1, 2, width_ratios=[1.15, 1.0])

    sub_a.suptitle(
        f"A   Causal attention on “{prompt.rstrip()}”",
        fontsize=9,
        fontweight="bold",
    )
    axes = sub_a.subplots(2, 2)
    labels = [("␣" if c == " " else c) for c in chars]
    for h, ax in enumerate(axes.ravel()):
        im = ax.imshow(
            attn[h, :t, :t],
            cmap="magma",
            vmin=0.0,
            vmax=max(0.15, float(attn[h, :t, :t].max())),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"head {h + 1}", fontsize=9)
        ax.set_xticks(range(t), labels, fontsize=7, rotation=90)
        ax.set_yticks(range(t), labels, fontsize=7)
        if h in (0, 2):
            ax.set_ylabel("query (who looks)", fontsize=7)
        if h in (2, 3):
            ax.set_xlabel("key (who is looked at)", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="attention weight")
    sub_a.text(
        0.5,
        0.0,
        "Row = query, column = key. Black above the diagonal is the causal mask (no future tokens).",
        transform=sub_a.transFigure,
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.25",
    )

    ax_loss = sub_b.subplots(1, 1)
    ax_loss.plot(np.arange(1, len(history) + 1), history, color="0.25", lw=1.2)
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Next-character NLL")
    ax_loss.set_title("B   Next-token loss", loc="left", fontsize=9, fontweight="bold")
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    top = np.argsort(probs)[::-1][:8]
    labels = [("␣" if tokenizer.itos[i] == " " else tokenizer.itos[i]) for i in top]
    ax_p = sub_c.subplots(1, 1)
    ax_p.barh(range(len(top))[::-1], probs[top], color="0.35")
    ax_p.set_yticks(range(len(top))[::-1], labels)
    ax_p.set_xlabel("p(next char)")
    ax_p.set_title("C   After the prompt", loc="left", fontsize=9, fontweight="bold")
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)
    return fig


def write_figure(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    history: list[float],
    out_dir: Path | str | None = None,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(model, tokenizer, history)
    pdf = out / "tiny_transformer_attention.pdf"
    png = out / "tiny_transformer_attention.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    n_steps: int = 500,
) -> Path:
    set_seed(seed)
    model, tokenizer, _, history = train_demo(seed=seed, n_steps=n_steps)
    return write_figure(model, tokenizer, history, out_dir)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
