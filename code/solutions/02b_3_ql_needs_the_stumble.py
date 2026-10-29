"""Standalone solution; also %load-able from the companion notebook."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    _SRC = Path(__file__).resolve().parents[1] / "src"
except NameError:
    _cwd = Path.cwd().resolve()
    if (_cwd / "src" / "brainsmodels").is_dir():
        _SRC = _cwd / "src"
    elif (_cwd.parent / "src" / "brainsmodels").is_dir():
        _SRC = _cwd.parent / "src"
    else:
        _SRC = _cwd / "code" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from brainsmodels.repro import DEFAULT_SEED, set_seed

SEED = set_seed(DEFAULT_SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

from brainsmodels.tmaze import CENTER, QLearningAgent, TMaze, run_trial_ql, run_experiment
traces = run_experiment(n_trials=400, seed=SEED)
for name, trials in traces.items():
    print(name, "cue-first", np.mean([t.visited_cue_first for t in trials]), "return", np.mean([sum(t.rewards) for t in trials]))
agent = QLearningAgent(seed=SEED)
env = TMaze(seed=SEED)
for context in np.random.default_rng(SEED).integers(0, 2, 400):
    run_trial_ql(env, agent, context=int(context))
print("Learned Q(center, unknown):", agent.Q[agent.max_steps, CENTER, 0])
print("The learner begins without the transition and observation model supplied to both planners.")
