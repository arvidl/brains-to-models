"""Two-level Rao–Ballard predictive coding on image patches.

A smallest-interesting implementation: higher-level units predict lower-level
activity; prediction errors drive inference; a Hebbian learning rule updates
the descending weights. Patches come from synthetic 1/f images so the demo
runs offline on a laptop.

Reference: Rao & Ballard, Nature Neuroscience (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


# ---------------------------------------------------------------------------
# Synthetic "natural" images and patches
# ---------------------------------------------------------------------------


def _unit_variance(image: FloatArray) -> FloatArray:
    image = image - image.mean()
    std = image.std()
    if std > 0:
        image = image / std
    return image


def pink_noise_image(size: int, rng: np.random.Generator) -> FloatArray:
    """Unit-variance image with a 1/f amplitude spectrum."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = np.inf
    phase = rng.uniform(0.0, 2.0 * np.pi, (size, size))
    spectrum = (1.0 / freq) * np.exp(1j * phase)
    spectrum[0, 0] = 0.0
    return _unit_variance(np.fft.ifft2(spectrum).real)


def grating_image(size: int, rng: np.random.Generator) -> FloatArray:
    """Windowed sinusoidal grating — oriented structure for the dictionary to find."""
    angle = float(rng.uniform(0.0, np.pi))
    cycles = float(rng.uniform(2.5, 7.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    ys, xs = np.indices((size, size))
    xs = (xs - size / 2.0) / size
    ys = (ys - size / 2.0) / size
    projected = xs * np.cos(angle) + ys * np.sin(angle)
    grating = np.sin(2.0 * np.pi * cycles * projected + phase)
    envelope = np.exp(-(xs**2 + ys**2) / 0.18)
    return _unit_variance(grating * envelope)


def contrast_normalize(patches: FloatArray, eps: float = 1e-6) -> FloatArray:
    patches = patches - patches.mean(axis=1, keepdims=True)
    std = patches.std(axis=1, keepdims=True)
    return patches / (std + eps)


@dataclass
class Whitening:
    """ZCA whitening in patch space (Olshausen–Field / Rao–Ballard style)."""

    mean: FloatArray
    W: FloatArray
    W_inv: FloatArray

    def apply(self, patches: FloatArray) -> FloatArray:
        return (patches - self.mean) @ self.W

    def invert(self, whitened: FloatArray) -> FloatArray:
        return whitened @ self.W_inv + self.mean

    def to_pixel_basis(self, weights: FloatArray) -> FloatArray:
        """Map columns of `weights` (whitened space) back to pixel space."""
        return self.W_inv.T @ weights


def fit_zca(patches: FloatArray, eps: float = 0.1) -> Whitening:
    mean = patches.mean(axis=0)
    centered = patches - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    d_inv = 1.0 / np.sqrt(evals + eps)
    d = np.sqrt(evals + eps)
    W = (evecs * d_inv) @ evecs.T
    W_inv = (evecs * d) @ evecs.T
    return Whitening(mean=mean, W=W, W_inv=W_inv)


@dataclass
class PatchDataset:
    patch_size: int
    train: FloatArray
    test: FloatArray
    whitening: Whitening
    train_pixel: FloatArray
    test_pixel: FloatArray

    @classmethod
    def from_pink_noise(
        cls,
        patch_size: int = 8,
        n_images: int = 20,
        image_size: int = 64,
        n_train: int = 2400,
        n_test: int = 32,
        zca_eps: float = 0.1,
        seed: int = DEFAULT_SEED,
    ) -> PatchDataset:
        """Build a whitened patch set from 1/f images plus a few gratings."""
        rng = np.random.default_rng(seed)
        n_gratings = max(4, n_images // 3)
        images = [pink_noise_image(image_size, rng) for _ in range(n_images)]
        images.extend(grating_image(image_size, rng) for _ in range(n_gratings))
        raw = _extract_patches(images, patch_size, n_train + n_test, rng)
        raw = contrast_normalize(raw)
        train_pixel, test_pixel = raw[:n_train], raw[n_train:]
        whitening = fit_zca(train_pixel, eps=zca_eps)
        return cls(
            patch_size=patch_size,
            train=whitening.apply(train_pixel),
            test=whitening.apply(test_pixel),
            whitening=whitening,
            train_pixel=train_pixel,
            test_pixel=test_pixel,
        )


def _extract_patches(
    images: list[FloatArray],
    patch_size: int,
    n_patches: int,
    rng: np.random.Generator,
) -> FloatArray:
    height, width = images[0].shape
    max_y = height - patch_size + 1
    max_x = width - patch_size + 1
    patches = np.empty((n_patches, patch_size * patch_size), dtype=float)
    for i in range(n_patches):
        image = images[int(rng.integers(0, len(images)))]
        y = int(rng.integers(0, max_y))
        x = int(rng.integers(0, max_x))
        patches[i] = image[y : y + patch_size, x : x + patch_size].ravel()
    return patches


# ---------------------------------------------------------------------------
# Hierarchical predictive coding
# ---------------------------------------------------------------------------


@dataclass
class InferenceTrace:
    prediction: FloatArray
    error: FloatArray
    mse: list[float]


class HierarchicalPredictiveCoding:
    """Two-level network: r2 → r1 → pixels, errors flowing upward."""

    def __init__(
        self,
        n_input: int,
        n_level1: int = 64,
        n_level2: int = 12,
        infer_lr: float = 0.15,
        prior: float = 0.25,
        seed: int = DEFAULT_SEED,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_input = n_input
        self.n_level1 = n_level1
        self.n_level2 = n_level2
        self.infer_lr = infer_lr
        self.prior = prior
        self.U1 = rng.normal(0.0, 0.05, (n_input, n_level1))
        self.U2 = rng.normal(0.0, 0.05, (n_level1, n_level2))
        self._normalize()

    def _normalize(self) -> None:
        for weights in (self.U1, self.U2):
            norms = np.linalg.norm(weights, axis=0, keepdims=True)
            weights /= np.maximum(norms, 1e-8)

    def infer(
        self,
        x: FloatArray,
        n_steps: int = 40,
        record: bool = False,
    ) -> tuple[FloatArray, FloatArray, InferenceTrace | None]:
        if x.ndim == 1:
            x = x[None, :]
        batch = x.shape[0]
        r1 = np.zeros((batch, self.n_level1))
        r2 = np.zeros((batch, self.n_level2))
        mse: list[float] = []
        last_pred = np.zeros_like(x)
        last_err = np.zeros_like(x)

        for _ in range(n_steps):
            pred0 = r1 @ self.U1.T
            err0 = x - pred0
            pred1 = r2 @ self.U2.T
            err1 = r1 - pred1
            r1 = r1 + self.infer_lr * (err0 @ self.U1 - err1)
            r2 = r2 + self.infer_lr * (err1 @ self.U2 - self.prior * r2)
            last_pred, last_err = pred0, err0
            if record:
                mse.append(float(np.mean(err0**2)))

        trace = InferenceTrace(last_pred, last_err, mse) if record else None
        return r1, r2, trace

    def infer_history(
        self, x: FloatArray, n_steps: int = 40
    ) -> tuple[list[FloatArray], list[float]]:
        """Return per-step reconstructions (whitened) and MSE."""
        if x.ndim == 1:
            x = x[None, :]
        batch = x.shape[0]
        r1 = np.zeros((batch, self.n_level1))
        r2 = np.zeros((batch, self.n_level2))
        preds: list[FloatArray] = []
        mse: list[float] = []
        for _ in range(n_steps):
            pred0 = r1 @ self.U1.T
            err0 = x - pred0
            pred1 = r2 @ self.U2.T
            err1 = r1 - pred1
            r1 = r1 + self.infer_lr * (err0 @ self.U1 - err1)
            r2 = r2 + self.infer_lr * (err1 @ self.U2 - self.prior * r2)
            preds.append(pred0.copy())
            mse.append(float(np.mean(err0**2)))
        return preds, mse

    def learn(
        self,
        x: FloatArray,
        n_infer: int = 30,
        lr: float = 0.02,
    ) -> float:
        r1, r2, _ = self.infer(x, n_steps=n_infer)
        pred0 = r1 @ self.U1.T
        err0 = x - pred0
        pred1 = r2 @ self.U2.T
        err1 = r1 - pred1
        batch = x.shape[0]
        self.U1 += lr * (err0.T @ r1) / batch
        self.U2 += lr * (err1.T @ r2) / batch
        self._normalize()
        return float(np.mean(err0**2))

    def train(
        self,
        patches: FloatArray,
        n_epochs: int = 20,
        batch_size: int = 50,
        n_infer: int = 30,
        lr: float = 0.02,
        seed: int = DEFAULT_SEED,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        history: list[float] = []
        n = patches.shape[0]
        for epoch in range(n_epochs):
            order = rng.permutation(n)
            epoch_err: list[float] = []
            for start in range(0, n, batch_size):
                batch = patches[order[start : start + batch_size]]
                epoch_err.append(self.learn(batch, n_infer=n_infer, lr=lr))
            history.append(float(np.mean(epoch_err)))
            print(f"epoch {epoch + 1:02d}/{n_epochs}  recon MSE {history[-1]:.4f}")
        return history


# ---------------------------------------------------------------------------
# Paper figure
# ---------------------------------------------------------------------------


def _show_patch(ax: plt.Axes, patch: FloatArray, patch_size: int, title: str) -> None:
    vmax = np.max(np.abs(patch)) + 1e-8
    ax.imshow(
        patch.reshape(patch_size, patch_size),
        cmap="gray",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


def render_paper_figure(
    model: HierarchicalPredictiveCoding,
    data: PatchDataset,
    example_idx: tuple[int, int, int] = (0, 3, 7),
    n_rfs: int = 16,
    n_infer: int = 40,
) -> plt.Figure:
    pixel_rfs = data.whitening.to_pixel_basis(model.U1)
    energy = pixel_rfs.reshape(data.patch_size, data.patch_size, -1).var(axis=(0, 1))
    order = np.argsort(energy)[::-1][:n_rfs]

    examples = data.test[list(example_idx)]
    examples_pixel = data.test_pixel[list(example_idx)]
    per_example_mse: list[list[float]] = []
    snapshots_w = {1: [], 5: [], n_infer: []}
    for patch in examples:
        history, curve = model.infer_history(patch, n_steps=n_infer)
        per_example_mse.append(curve)
        snapshots_w[1].append(history[0][0])
        snapshots_w[5].append(history[min(4, len(history) - 1)][0])
        snapshots_w[n_infer].append(history[-1][0])
    snapshots = {
        step: data.whitening.invert(np.stack(patches))
        for step, patches in snapshots_w.items()
    }

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(6.3, 5.8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.04)
    sub_a, sub_b, sub_c = fig.subfigures(3, 1, height_ratios=[1.05, 1.35, 0.95])

    sub_a.suptitle("A   Learned first-level basis (pixel space)", fontsize=9, fontweight="bold")
    rf_axes = sub_a.subplots(2, 8)
    for ax, idx in zip(rf_axes.ravel(), order):
        _show_patch(ax, pixel_rfs[:, idx], data.patch_size, "")

    sub_b.suptitle("B   Explain-away on held-out patches", fontsize=9, fontweight="bold")
    recon_axes = sub_b.subplots(3, 5)
    col_titles = ["Input", "Pred. t=1", "Pred. t=5", f"Pred. t={n_infer}", "Residual"]
    for row, _ in enumerate(example_idx):
        panels = [
            examples_pixel[row],
            snapshots[1][row],
            snapshots[5][row],
            snapshots[n_infer][row],
            examples_pixel[row] - snapshots[n_infer][row],
        ]
        for col, (ax, patch) in enumerate(zip(recon_axes[row], panels)):
            title = col_titles[col] if row == 0 else ""
            _show_patch(ax, patch, data.patch_size, title)

    ax_err = sub_c.subplots(1, 1)
    steps = np.arange(1, n_infer + 1)
    for row, curve in enumerate(per_example_mse):
        ax_err.plot(steps, curve, lw=1.4, label=f"patch {row + 1}")
    mean_curve = np.mean(np.stack(per_example_mse), axis=0)
    ax_err.plot(steps, mean_curve, color="0.25", lw=1.8, label="mean of three")
    ax_err.set_xlabel("Inference step")
    ax_err.set_ylabel("Prediction-error MSE")
    ax_err.set_title(
        "C   Residual falls as the hierarchy explains the patch",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax_err.legend(frameon=False, fontsize=7, ncols=4)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)
    return fig


def train_demo(
    seed: int = DEFAULT_SEED,
    n_epochs: int = 25,
) -> tuple[HierarchicalPredictiveCoding, PatchDataset, list[float]]:
    data = PatchDataset.from_pink_noise(seed=seed)
    model = HierarchicalPredictiveCoding(
        n_input=data.train.shape[1],
        seed=seed,
    )
    history = model.train(data.train, n_epochs=n_epochs, seed=seed)
    return model, data, history


def write_figure(
    model: HierarchicalPredictiveCoding,
    data: PatchDataset,
    out_dir: Path | str | None = None,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(model, data)
    pdf = out / "predictive_coding_patches.pdf"
    png = out / "predictive_coding_patches.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    n_epochs: int = 25,
) -> Path:
    set_seed(seed)
    model, data, _ = train_demo(seed=seed, n_epochs=n_epochs)
    return write_figure(model, data, out_dir)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
