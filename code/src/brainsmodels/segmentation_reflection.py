"""Synthetic segmentation, annotation policies and feature interventions.

Two supervised, three-layer pixel networks learn different annotation policies.
Only NumPy is required here; the notebook handles plots. Nothing is downloaded
or written by this module. The images are designed phantoms, not simulated MRI
physics, patient data, or a clinical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    images: np.ndarray
    core: np.ndarray
    inclusive: np.ndarray
    case_seeds: np.ndarray


def make_dataset(n: int, seed: int, size: int = 48) -> Dataset:
    """Generate independent two-channel images and two geometric label policies."""
    case_seeds = np.random.default_rng(seed).choice(10_000_000, n, replace=False)
    images, cores, inclusive_masks = [], [], []
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    field = (x / 0.88) ** 2 + (y / 0.94) ** 2 <= 1
    for case_seed in case_seeds:
        rng = np.random.default_rng(int(case_seed))
        cx, cy = rng.uniform(-0.22, 0.22, 2)
        rx, ry = rng.uniform(0.22, 0.31, 2)
        radius = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        core, inclusive = radius <= 1, radius <= 1.35
        halo = inclusive & ~core
        texture = 0.025 * np.sin(9 * x + rng.uniform(0, 6)) * np.cos(7 * y)
        a = 0.20 + texture + 0.58 * core + 0.31 * halo
        b = 0.24 - texture + 0.48 * core + 0.28 * halo
        # A bright distractor in A has background-like signal in B.
        distractor = ((x + 0.58) ** 2 + (y - 0.48) ** 2 < 0.15**2)
        distractor &= field & ~inclusive
        a += 0.55 * distractor
        image = np.stack([a, b], axis=-1)
        image += rng.normal(0, 0.055, image.shape)
        image = np.where(field[..., None], image, 0.035)
        images.append(np.clip(image, 0, 1))
        cores.append(core)
        inclusive_masks.append(inclusive)
    return Dataset(np.array(images), np.array(cores), np.array(inclusive_masks), case_seeds)


def local_mean(images: np.ndarray) -> np.ndarray:
    """Fixed 3×3 local means; no learned spatial filter or coordinates."""
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    height, width = images.shape[1:3]
    return sum(padded[:, i:i + height, j:j + width] for i in range(3)
               for j in range(3)) / 9


def pixel_features(images: np.ndarray) -> np.ndarray:
    return np.concatenate([images, local_mean(images)], axis=-1)


def initialize(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parameters = {}
    for layer, (n_in, n_out) in enumerate(((4, 16), (16, 8), (8, 1)), 1):
        parameters[f"w{layer}"] = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_out))
        parameters[f"b{layer}"] = np.zeros(n_out)
    return parameters


def forward(x: np.ndarray, parameters: dict[str, np.ndarray]):
    h1 = np.tanh(x @ parameters["w1"] + parameters["b1"])
    h2 = np.tanh(h1 @ parameters["w2"] + parameters["b2"])
    logits = h2 @ parameters["w3"] + parameters["b3"]
    return logits, h1, h2


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -logits))


def loss_and_gradients(x: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray]):
    """Binary cross-entropy and analytic backpropagation for all three layers."""
    logits, h1, h2 = forward(x, parameters)
    loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
    delta = (sigmoid(logits) - y) / len(y)
    gradients = {"w3": h2.T @ delta, "b3": delta.sum(axis=0)}
    delta = (delta @ parameters["w3"].T) * (1 - h2**2)
    gradients.update(w2=h1.T @ delta, b2=delta.sum(axis=0))
    delta = (delta @ parameters["w2"].T) * (1 - h1**2)
    gradients.update(w1=x.T @ delta, b1=delta.sum(axis=0))
    return loss, gradients


@dataclass
class PixelNetwork:
    parameters: dict[str, np.ndarray]
    mean: np.ndarray
    scale: np.ndarray
    losses: np.ndarray

    def inspect(self, images: np.ndarray):
        features = pixel_features(images)
        x = (features.reshape(-1, 4) - self.mean) / self.scale
        logits, _, h2 = forward(x, self.parameters)
        return sigmoid(logits).reshape(images.shape[:3]), h2.reshape(*images.shape[:3], 8)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.inspect(images)[0]


def train(images: np.ndarray, labels: np.ndarray, seed: int = 7,
          steps: int = 350, batch_size: int = 512) -> PixelNetwork:
    """Fit using training images only; each call repeats the same initialization/batches."""
    features = pixel_features(images).reshape(-1, 4)
    mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-8
    x, y = (features - mean) / scale, labels.reshape(-1, 1).astype(float)
    parameters = initialize(seed)
    rng = np.random.default_rng(seed + 1)
    moment = {key: np.zeros_like(value) for key, value in parameters.items()}
    variance = {key: np.zeros_like(value) for key, value in parameters.items()}
    # A fixed training-only monitoring sample; held-out images never select an epoch.
    monitor = rng.choice(len(x), min(4096, len(x)), replace=False)
    losses = [loss_and_gradients(x[monitor], y[monitor], parameters)[0]]
    for step in range(1, steps + 1):
        batch = rng.choice(len(x), batch_size, replace=True)
        _, gradients = loss_and_gradients(x[batch], y[batch], parameters)
        for key, gradient in gradients.items():
            moment[key] = 0.9 * moment[key] + 0.1 * gradient
            variance[key] = 0.999 * variance[key] + 0.001 * gradient**2
            m_hat = moment[key] / (1 - 0.9**step)
            v_hat = variance[key] / (1 - 0.999**step)
            parameters[key] -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
        if step % 10 == 0 or step == steps:
            losses.append(loss_and_gradients(x[monitor], y[monitor], parameters)[0])
    return PixelNetwork(parameters, mean, scale, np.array(losses))


def dice(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """One Dice score per image; define two empty masks to have Dice 1."""
    predicted, reference = np.asarray(predicted, bool), np.asarray(reference, bool)
    axes = tuple(range(1, predicted.ndim))
    intersection = np.sum(predicted & reference, axis=axes)
    size = np.sum(predicted, axis=axes) + np.sum(reference, axis=axes)
    return np.divide(2 * intersection, size, out=np.ones_like(size, dtype=float), where=size != 0)


def channel_intervention(images: np.ndarray, channel: int, value: float = 0.24) -> np.ndarray:
    """Replace one entire synthetic channel; recompute features at prediction time."""
    altered = images.copy()
    altered[..., channel] = value
    return altered


def ablate_unit(model: PixelNetwork, images: np.ndarray, unit: int) -> np.ndarray:
    """Set a second-hidden-layer unit to zero before the trained output readout."""
    _, hidden = model.inspect(images)
    hidden = hidden.copy()
    hidden[..., unit] = 0
    logits = hidden @ model.parameters["w3"] + model.parameters["b3"]
    return sigmoid(logits[..., 0])


def gradient_error(seed: int = 91) -> float:
    """Check every trainable parameter against central finite differences."""
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(9, 4)), rng.integers(0, 2, (9, 1))
    parameters = initialize(seed)
    _, analytic = loss_and_gradients(x, y, parameters)
    largest, eps = 0.0, 1e-5
    for key, value in parameters.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus = loss_and_gradients(x, y, parameters)[0]
            value[index] = original - eps
            minus = loss_and_gradients(x, y, parameters)[0]
            value[index] = original
            largest = max(largest, abs((plus - minus) / (2 * eps) - analytic[key][index]))
    return float(largest)
