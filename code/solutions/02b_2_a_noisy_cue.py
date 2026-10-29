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
def score(acc, n=40):
    env = TMaze(cue_accuracy=acc, seed=SEED)
    agent = ActiveInferenceAgent(cue_accuracy=acc)
    cues, rews = [], []
    for _ in range(n):
        tr = run_trial_aif(env, agent, n_steps=3)
        cues.append(tr.visited_cue_first)
        rews.append(sum(tr.rewards))
    return float(np.mean(cues)), float(np.mean(rews))

for acc in (1.0, 0.6, 0.5):
    cue, rew = score(acc)
    print(f"cue accuracy {acc:.1f}  cue-first {cue:.2f}  mean return {rew:+.2f}")
print(
    "\nAt 0.5 the cue is a coin flip and information gain collapses. The "
    "policy is then a bet on an arm. Epistemic value tracks how much an "
    "observation would change the context belief, not a virtue of looking. "
    "A noisy instrument in clinic is the same arithmetic: a test that does "
    "not change the posterior is not worth the step, and that is still not "
    "a rat."
)
