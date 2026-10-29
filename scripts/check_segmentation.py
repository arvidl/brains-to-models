"""Numerical and scientific-scope checks for the segmentation notebook."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "src"))
from brainsmodels.segmentation_reflection import (
    PixelNetwork, ablate_unit, channel_intervention, dice, gradient_error,
    initialize, make_dataset, train,
)


def main():
    # An independent finite-difference check catches errors in the training rule.
    error = gradient_error()
    assert error < 1e-7, error
    empty = np.zeros((1, 3, 3), dtype=bool)
    assert dice(empty, empty)[0] == 1
    assert dice(~empty, empty)[0] == 0

    training, heldout = make_dataset(20, 7), make_dataset(6, 2026)
    assert not set(training.case_seeds) & set(heldout.case_seeds)
    assert np.all(training.core <= training.inclusive)
    assert np.any(training.core != training.inclusive)
    initial_images = training.images.copy()
    models = {policy: train(training.images, getattr(training, policy))
              for policy in ("core", "inclusive")}
    assert np.array_equal(training.images, initial_images)
    predictions = {policy: model.predict(heldout.images) for policy, model in models.items()}
    for policy, model in models.items():
        assert model.losses[-1] < model.losses[0] * 0.1
        own = dice(predictions[policy] >= 0.5, getattr(heldout, policy)).mean()
        other_policy = "inclusive" if policy == "core" else "core"
        other = dice(predictions[policy] >= 0.5, getattr(heldout, other_policy)).mean()
        assert own > 0.90 and own - other > 0.15, (policy, own, other)
    assert np.all((predictions["inclusive"] >= 0.5).sum(axis=(1, 2)) >
                  (predictions["core"] >= 0.5).sum(axis=(1, 2)))

    # Repeated fitting with the same seed must reproduce held-out predictions.
    repeat = train(training.images, training.core)
    assert np.array_equal(repeat.predict(heldout.images), predictions["core"])

    # Replacement must preserve the other input channel and the source array.
    altered = channel_intervention(heldout.images, 1)
    assert np.array_equal(altered[..., 0], heldout.images[..., 0])
    assert np.all(altered[..., 1] == 0.24)
    assert not np.shares_memory(altered, heldout.images)
    assert np.abs(models["inclusive"].predict(altered) - predictions["inclusive"]).mean() > 0.01

    model = models["inclusive"]
    base, hidden = model.inspect(heldout.images)
    effects = [np.abs(ablate_unit(model, heldout.images, unit) - base).mean()
               for unit in range(8)]
    assert max(effects) > 1e-5
    assert np.array_equal(model.predict(heldout.images), base), "Ablation mutated the model"

    # Internal unit identities are not unique: coordinated permutation preserves output.
    permutation = np.arange(8)[::-1]
    parameters = {key: value.copy() for key, value in model.parameters.items()}
    parameters["w2"] = parameters["w2"][:, permutation]
    parameters["b2"] = parameters["b2"][permutation]
    parameters["w3"] = parameters["w3"][permutation, :]
    permuted = PixelNetwork(parameters, model.mean, model.scale, model.losses)
    changed_output, changed_hidden = permuted.inspect(heldout.images)
    assert np.allclose(changed_output, base, atol=1e-12, rtol=0)
    assert np.allclose(changed_hidden, hidden[..., permutation], atol=1e-12, rtol=0)

    # Trainable parameter count stated in the notebook, rather than model-size guesswork.
    assert sum(value.size for value in initialize().values()) == 225
    print(f"PASS: segmentation gradients (max error {error:.3g}); image-level holdout; "
          "label-policy effects; deterministic fitting; input and hidden-unit interventions; "
          "equivalent internal parameterizations.")


if __name__ == "__main__":
    main()
