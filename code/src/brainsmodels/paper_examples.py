"""Exact numerical examples shared by the paper and its companion notebooks.

Pure NumPy calculations: no training, plotting, file writes, or downloads.
"""

from dataclasses import dataclass

import numpy as np


SHORTCUT_AGREEMENTS = {"training_like": 0.95, "independent": 0.5, "reversed": 0.05}
STABLE_ACCURACY = 0.8


@dataclass
class AttentionExample:
    query: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    contributions: np.ndarray
    output: float


def two_key_attention(values=(2.0, 10.0), query=(1.0, 0.0)) -> AttentionExample:
    """Scaled dot-product attention with two fixed orthogonal keys."""
    q, keys, v = np.asarray(query, dtype=float), np.eye(2), np.asarray(values, dtype=float)
    if q.shape != (2,) or v.shape != (2,):
        raise ValueError("The teaching example needs a two-dimensional query and two scalar values")
    scores = keys @ q / np.sqrt(2)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return AttentionExample(q, keys, v, scores, weights, weights * v, float(weights @ v))


@dataclass
class ShortcutExample:
    agreement: np.ndarray
    marker_accuracy: np.ndarray
    stable_accuracy: np.ndarray


def shortcut_dependence(agreement=None, stable_accuracy=STABLE_ACCURACY) -> ShortcutExample:
    """Expected accuracies of two fixed rules, not measurements of a trained model."""
    a = np.linspace(0, 1, 201) if agreement is None else np.asarray(agreement, dtype=float)
    if np.any((a < 0) | (a > 1)) or not 0 <= stable_accuracy <= 1:
        raise ValueError("Agreement and stable-rule accuracy must be between zero and one")
    return ShortcutExample(a, a.copy(), np.full_like(a, stable_accuracy))


@dataclass
class HopfieldExample:
    pattern: np.ndarray
    weights: np.ndarray
    states: np.ndarray
    energies: np.ndarray
    units: np.ndarray
    fields: np.ndarray


def hopfield_energy(state: np.ndarray, weights: np.ndarray) -> float:
    return float(-0.5 * state @ weights @ state)


def hopfield_12(flipped_bits=(0, 3, 7), sweeps=2) -> HopfieldExample:
    """One stored pattern, zero self-weights, ordered asynchronous updates.

    On a zero local field retain the current bit. Bit indices are zero-based.
    The defaults reproduce the paper's 25-state trajectory.
    """
    bits = tuple(flipped_bits)
    if len(set(bits)) != len(bits) or any(i not in range(12) for i in bits):
        raise ValueError("Choose distinct bit indices from 0 through 11")
    if not isinstance(sweeps, int) or sweeps < 0:
        raise ValueError("sweeps must be a nonnegative integer")
    pattern = np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1], dtype=float)
    weights = np.outer(pattern, pattern) / len(pattern)
    np.fill_diagonal(weights, 0)
    state = pattern.copy()
    state[list(bits)] *= -1
    states, energies, fields = [state.copy()], [hopfield_energy(state, weights)], []
    units = np.tile(np.arange(12), sweeps)
    for i in units:
        field = float(weights[i] @ state)
        if field != 0:
            state[i] = 1 if field > 0 else -1
        states.append(state.copy())
        energies.append(hopfield_energy(state, weights))
        fields.append(field)
    return HopfieldExample(pattern, weights, np.array(states), np.array(energies), units,
                           np.array(fields))
