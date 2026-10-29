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
def cue_rate(agent, n=40):
    env = TMaze(seed=SEED)
    return sum(int(run_trial_aif(env, agent, n_steps=3).visited_cue_first) for _ in range(n)) / n

def no_ig(horizon):
    agent = ActiveInferenceAgent(horizon=horizon)
    agent._info_gain_one_channel = lambda *args, **kwargs: 0.0
    return agent

print(f"horizon 2, full EFE     cue-first {cue_rate(ActiveInferenceAgent(horizon=2)):.2f}")
print(f"horizon 2, no IG        cue-first {cue_rate(no_ig(2)):.2f}")
print(f"horizon 1, no IG        cue-first {cue_rate(no_ig(1)):.2f}")
print(
    "\nDropping the epistemic bonus is not enough if the horizon can still "
    "plan *cue then arm*: the cue remains a means to a later preferred "
    "observation. A one-step agent with no information gain has no such "
    "plan and stays put. Two ingredients, not one slogan. Still a "
    "four-location teaching model, not a rat."
)
