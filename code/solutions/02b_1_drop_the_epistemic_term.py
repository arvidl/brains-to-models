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

from brainsmodels.tmaze import ActiveInferenceAgent, TMaze, run_trial_aif
for horizon in (1, 3):
    for weight in (0., 1.):
        agent = ActiveInferenceAgent(horizon=horizon, epistemic_weight=weight)
        trace = run_trial_aif(TMaze(seed=SEED), agent, context=1)
        print(f"horizon={horizon}, IG weight={weight}: {trace.locations}, return={sum(trace.rewards):+.1f}")
print("The three-move reward-only planner can value information instrumentally. Both planners know the model.")
