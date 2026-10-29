"""Regression checks for the teaching comparisons and their generated notes."""
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/notebooks"))
from brainsmodels.tmaze import (CENTER, CUE, LEFT, RIGHT, TMaze, QLearningAgent,
    ActiveInferenceAgent, run_trial_aif, run_trial_ql, cue_before_arm)
from brainsmodels.tiny_transformer import TinyTransformer
from _questions import ITEMS, NOTEBOOKS, QUESTIONS
from _responses import render
from _reading import READING


def main():
    q = QLearningAgent(alpha=1., gamma=.9)
    q.Q[:, LEFT] = 99.  # Terminal values must never be bootstrapped.
    q.update(CENTER, LEFT, 1., LEFT, 0)
    np.testing.assert_array_equal(q.Q[3, CENTER, 0], [0., 0., 1., 0.])
    q = QLearningAgent(alpha=1., gamma=.9)
    q.Q[2, CUE, 2, CENTER] = 2.
    q.update(CENTER, CUE, 0., CUE, 2)
    assert q.cue_mem == 2 and q.Q[3, CENTER, 0, CUE] == 1.8
    q.reset(n_steps=1)
    q.Q[0] = 99.
    q.update(CENTER, CENTER, 0., CENTER, 0)
    assert q.Q[1, CENTER, 0, CENTER] == 0., "Time limit was bootstrapped"
    assert not cue_before_arm([CENTER, LEFT, CUE])
    assert cue_before_arm([CENTER, CUE, CENTER, RIGHT])
    env = TMaze()
    env.reset(context=0)
    env.step(CUE)
    try:
        env.step(LEFT)
    except ValueError:
        pass
    else:
        raise AssertionError("Cue-to-arm teleport accepted")
    env.step(CENTER)
    env.step(LEFT)
    try:
        env.step(LEFT)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated terminal reward accepted")
    for weight in (0., 1.):
        planner = ActiveInferenceAgent(epistemic_weight=weight)
        scores = planner.action_scores(CENTER, np.array([.5, .5]), 3)
        assert np.isclose(scores[LEFT], scores[RIGHT])
        left = planner.expected_free_energy(CENTER, np.array([.5, .5]), np.array([CUE, CENTER, LEFT]))
        right = planner.expected_free_energy(CENTER, np.array([.5, .5]), np.array([CUE, CENTER, RIGHT]))
        assert np.isclose(left, right), (left, right)
        for context, arm in ((0, LEFT), (1, RIGHT)):
            trace = run_trial_aif(TMaze(), planner, context=context)
            assert trace.locations == [CENTER, CUE, CENTER, arm], trace
            assert sum(trace.rewards) == 1.
    short = ActiveInferenceAgent(horizon=1, epistemic_weight=0.)
    assert short.act(CENTER) == CENTER
    for context in (0, 1):
        trace = run_trial_ql(TMaze(), QLearningAgent(), context=context)
        assert len(trace.locations) <= 4
        assert sum(abs(r) for r in trace.rewards) <= 1
    learned = QLearningAgent(seed=7)
    training_maze = TMaze(seed=7)
    for context in np.random.default_rng(7).integers(0, 2, 5000):
        run_trial_ql(training_maze, learned, context=int(context))
    learned.epsilon = 0.
    for context in (0, 1):
        trace = run_trial_ql(training_maze, learned, context=context, learn=False)
        assert trace.visited_cue_first and sum(trace.rewards) == 1., trace
    # Handwritten backprop: sample each parameter array against an independent
    # central difference, keeping a copy of the unperturbed analytic gradients.
    model = TinyTransformer(5, d_model=4, n_heads=2, d_ff=6, block_size=3, seed=8)
    x = np.array([[0, 1, 2], [2, 3, 4]])
    y = np.array([[1, 2, 3], [3, 4, 0]])
    model.loss_and_backward(x, y)
    gradients = [g.copy() for g in model._grads()]
    rng = np.random.default_rng(1)
    errors = []
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        for index in rng.choice(parameter.size, min(5, parameter.size), replace=False):
            old = parameter.flat[index]
            parameter.flat[index] = old + 1e-5
            plus = model.loss_and_backward(x, y)
            parameter.flat[index] = old - 1e-5
            minus = model.loss_and_backward(x, y)
            parameter.flat[index] = old
            errors.append(abs((plus-minus)/2e-5-gradient.flat[index]))
    assert max(errors) < 1e-7, max(errors)
    assert (ROOT / "code/notebooks/questions_responses.md").read_text() == render()
    for key, filename in NOTEBOOKS.items():
        nb = json.loads((ROOT / "code/notebooks" / filename).read_text())
        cells = {c['id']: ''.join(c['source']) for c in nb['cells']}
        assert len(ITEMS[key]) == 5
        assert cells['questions001'] == QUESTIONS[key], filename
        assert cells['reading001'] == READING[key], filename
    print(f"PASS: action-specific TD update; terminal rewards; maze topology; "
          f"cue ordering; outcome symmetry; reward-only control; Transformer "
          f"gradients (max {max(errors):.2g}); generated discussion/reading consistency.")


if __name__ == "__main__":
    main()
