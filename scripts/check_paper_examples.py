"""Check the paper's exact numerical examples and notebook results."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
from brainsmodels.paper_examples import (
    SHORTCUT_AGREEMENTS, STABLE_ACCURACY, hopfield_12, hopfield_energy,
    shortcut_dependence, two_key_attention,
)


def main():
    values = json.loads((ROOT / "paper/figures/computed_values.json").read_text())
    appendix = json.loads((ROOT / "paper/figures/appendix_values.json").read_text())
    original, changed = two_key_attention(), two_key_attention(values=(2, -10))
    # Independent scalar expression for the two-key softmax, not another call to the helper.
    expected_weight = 1 / (1 + np.exp(-1 / np.sqrt(2)))
    assert np.allclose(original.weights, [expected_weight, 1 - expected_weight], atol=1e-14)
    assert np.array_equal(original.weights, changed.weights)
    assert np.isclose(original.output - changed.output, 20 * original.weights[1])
    assert np.allclose(original.weights, appendix["attention"]["weights"], atol=1e-14)
    assert original.output == appendix["attention"]["output"]
    assert changed.output == appendix["attention"]["changed_value_output"]

    shortcut = shortcut_dependence([*SHORTCUT_AGREEMENTS.values()])
    assert np.array_equal(shortcut.marker_accuracy, [.95, .5, .05])
    assert np.all(shortcut.stable_accuracy == STABLE_ACCURACY)
    assert values["shortcut_accuracy"] == {**SHORTCUT_AGREEMENTS, "stable_rule": STABLE_ACCURACY}

    result = hopfield_12()
    assert result.weights.shape == (12, 12) and result.states.shape == (25, 12)
    assert np.array_equal(result.weights, result.weights.T)
    assert np.all(np.diag(result.weights) == 0)
    assert np.flatnonzero(result.states[0] != result.pattern).tolist() == [0, 3, 7]
    assert np.array_equal(result.states[-1], result.pattern)
    assert np.allclose(result.energies, values["hopfield_energy"], atol=1e-14)
    assert np.allclose(result.energies, appendix["hopfield_energy"], atol=1e-14)
    for step, (unit, field) in enumerate(zip(result.units, result.fields)):
        before, after = result.states[step:step + 2]
        assert np.count_nonzero(before != after) <= 1
        delta = result.energies[step + 1] - result.energies[step]
        assert np.isclose(delta, -(after[unit] - before[unit]) * field, atol=1e-14)
        assert delta <= 1e-12
    inverse = hopfield_12(flipped_bits=tuple(range(12)))
    assert np.all(inverse.states == -result.pattern)
    assert hopfield_energy(-result.pattern, result.weights) == appendix["inverse_pattern_energy"]

    # All 4096 states: check the closed-form energy and every possible one-unit update.
    states = 2 * ((np.arange(4096)[:, None] >> np.arange(12)) & 1) - 1
    energies = -0.5 * np.einsum("bi,ij,bj->b", states, result.weights, states)
    overlap_sum = states @ result.pattern
    assert np.allclose(energies, -(overlap_sum**2 - 12) / 24, atol=1e-12)
    for unit in range(12):
        fields = states @ result.weights[unit]
        updated = states.copy()
        updated[:, unit] = np.where(fields == 0, states[:, unit], np.where(fields > 0, 1, -1))
        after = -0.5 * np.einsum("bi,ij,bj->b", updated, result.weights, updated)
        assert np.all(after <= energies + 1e-12)

    for filename in (
        "10_two_key_attention.ipynb", "11_shortcut_dependence.ipynb",
        "12_hopfield_12_units.ipynb",
    ):
        nb = json.loads((ROOT / "code/notebooks" / filename).read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))
        assert not [o for c in code for o in c.get("outputs", []) if o["output_type"] == "error"]
    assert (ROOT / "code/notebooks/08_attention_maps.ipynb").is_file()
    print("PASS: paper attention and shortcut values; exact 12-unit recall; "
          "49152 single-unit energy checks; saved execution of notebooks 10–12; "
          "attention-map companion notebook.")


if __name__ == "__main__":
    main()
