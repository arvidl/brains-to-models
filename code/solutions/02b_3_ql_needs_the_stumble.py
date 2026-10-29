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

from brainsmodels.tmaze import (
    CENTER,
    CUE,
    LEFT,
    RIGHT,
    ActiveInferenceAgent,
    QLearningAgent,
    TMaze,
    run_trial_aif,
    run_trial_ql,
)
n = 80
env_a = TMaze(seed=SEED)
env_q = TMaze(seed=SEED + 1)
aif = ActiveInferenceAgent()
ql = QLearningAgent(seed=SEED + 2, epsilon=0.25)
aif_hits = sum(int(run_trial_aif(env_a, aif, n_steps=3).visited_cue_first) for _ in range(n))
ql_hits = sum(int(run_trial_ql(env_q, ql, n_steps=3).visited_cue_first) for _ in range(n))
q0 = ql.Q[CENTER, 0]
print(f"AIF cue-first over {n}: {aif_hits / n:.2f}")
print(f"QL  cue-first over {n}: {ql_hits / n:.2f}")
print(f"Q(start, unknown)  cue {q0[CUE]:+.2f}  left {q0[LEFT]:+.2f}  right {q0[RIGHT]:+.2f}")
print(f"greedy from start: {('center', 'cue', 'left', 'right')[int(np.argmax(q0))]}")
print(
    "\nActive inference visits the cue on the first trial. Q-learning "
    "has only the stumbles that ε-greedy happened to take, and a three-step "
    "return is a harsh credit assignment. Even if a longer budget made "
    "the cue greedy, the goal would still be an external scalar, not a "
    "preferred observation, and not a body that must keep going."
)
