"""T-maze: active inference versus Q-learning.

Four locations form a star: center connects to cue and two terminal arms.
A hidden context places reward in one arm. Known-model belief planners
are compared with a tabular learner that initially knows no action values.
The reward-only planner controls for the instrumental value of information.
This simplified task is inspired by T-maze examples, not a pymdp replica
or a controlled comparison of active inference with reinforcement learning.

References: Friston et al., Neural Computation (2017); Sajid et al. (2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

CENTER, CUE, LEFT, RIGHT = 0, 1, 2, 3
LOC_NAMES = ("center", "cue", "left", "right")
N_LOC = 4
REWARD_LEFT, REWARD_RIGHT = 0, 1

# Observations
# cue: 0 = none, 1 = reward-left, 2 = reward-right
# rew: 0 = neutral, 1 = reward, 2 = punish
N_CUE, N_REW = 3, 3


def available_actions(loc: int) -> tuple[int, ...]:
    """Stay or traverse one edge; entering an arm ends the trial."""
    return {CENTER: (CENTER, CUE, LEFT, RIGHT), CUE: (CENTER, CUE),
            LEFT: (), RIGHT: ()}[loc]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


def _softmax(x: FloatArray) -> FloatArray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _entropy(p: FloatArray, axis: int = -1) -> FloatArray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class TMaze:
    """Deterministic T-maze; context is sampled each trial."""

    cue_accuracy: float = 1.0
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.location = CENTER
        self.context = REWARD_LEFT

    def reset(self, context: int | None = None) -> tuple[int, int, int]:
        self.location = CENTER
        self.context = (
            int(self.rng.integers(0, 2)) if context is None else int(context)
        )
        return self.observe()

    def observe(self) -> tuple[int, int, int]:
        loc = self.location
        if loc == CUE:
            true = 1 if self.context == REWARD_LEFT else 2
            if self.rng.random() < self.cue_accuracy:
                cue = true
            else:
                cue = 2 if true == 1 else 1
        else:
            cue = 0
        if loc == LEFT:
            rew = 1 if self.context == REWARD_LEFT else 2
        elif loc == RIGHT:
            rew = 1 if self.context == REWARD_RIGHT else 2
        else:
            rew = 0
        return loc, cue, rew

    def step(self, action: int) -> tuple[tuple[int, int, int], float]:
        if action not in available_actions(self.location):
            raise ValueError("Move along one maze edge; arms are terminal.")
        self.location = int(action)
        loc, cue, rew = self.observe()
        reward = {0: 0.0, 1: 1.0, 2: -1.0}[rew]
        return (loc, cue, rew), reward


# ---------------------------------------------------------------------------
# Active inference (receding-horizon EFE)
# ---------------------------------------------------------------------------


class ActiveInferenceAgent:
    """Finite-horizon belief planner: negative utility minus information gain.

    Future actions depend on the observations received. All possible outcomes
    are averaged, rather than treating one imagined observation as certain.
    This explicit utility/IG objective is a teaching approximation to EFE.
    """

    def __init__(
        self,
        horizon: int = 3,
        cue_accuracy: float = 1.0,
        prefer_reward: float = 1.0,
        prefer_punish: float = -1.0,
        epistemic_weight: float = 1.0,
    ) -> None:
        self.horizon = horizon
        if horizon < 1 or not 0 <= cue_accuracy <= 1:
            raise ValueError("Use a positive horizon and cue accuracy in [0, 1].")
        self.cue_accuracy = cue_accuracy
        self.epistemic_weight = epistemic_weight
        self.C_rew = np.array([0.0, prefer_reward, prefer_punish])
        self.qs_ctx = np.array([0.5, 0.5])

    def reset(self) -> None:
        self.qs_ctx = np.array([0.5, 0.5])

    def infer_context(self, loc: int, cue: int, rew: int) -> FloatArray:
        logp = np.log(self.qs_ctx + 1e-12)
        for ctx in (REWARD_LEFT, REWARD_RIGHT):
            logp[ctx] += np.log(self._p_cue(loc, ctx, cue) + 1e-12)
            logp[ctx] += np.log(self._p_rew(loc, ctx, rew) + 1e-12)
        self.qs_ctx = _softmax(logp)
        return self.qs_ctx

    def _p_cue(self, loc: int, ctx: int, cue: int) -> float:
        if loc != CUE:
            return 1.0 if cue == 0 else 0.0
        true = 1 if ctx == REWARD_LEFT else 2
        if cue == 0:
            return 0.0
        return self.cue_accuracy if cue == true else 1.0 - self.cue_accuracy

    def _p_rew(self, loc: int, ctx: int, rew: int) -> float:
        if loc in (CENTER, CUE):
            return 1.0 if rew == 0 else 0.0
        good = (loc == LEFT and ctx == REWARD_LEFT) or (
            loc == RIGHT and ctx == REWARD_RIGHT
        )
        want = 1 if good else 2
        return 1.0 if rew == want else 0.0

    def _predict_obs(self, loc: int, qs_ctx: FloatArray) -> tuple[FloatArray, FloatArray]:
        p_cue = np.zeros(N_CUE)
        p_rew = np.zeros(N_REW)
        for ctx, p in enumerate(qs_ctx):
            for cue in range(N_CUE):
                p_cue[cue] += p * self._p_cue(loc, ctx, cue)
            for rew in range(N_REW):
                p_rew[rew] += p * self._p_rew(loc, ctx, rew)
        return p_cue, p_rew

    def _info_gain_one_channel(self, loc: int, qs_ctx: FloatArray) -> float:
        """IG from the single informative observation at this location."""
        h_prior = float(_entropy(qs_ctx))
        if loc == CUE:
            p_obs = self._predict_obs(loc, qs_ctx)[0]
            likelihood = lambda o, ctx: self._p_cue(loc, ctx, o)
        elif loc in (LEFT, RIGHT):
            p_obs = self._predict_obs(loc, qs_ctx)[1]
            likelihood = lambda o, ctx: self._p_rew(loc, ctx, o)
        else:
            return 0.0
        h_post = 0.0
        for o, po in enumerate(p_obs):
            if po < 1e-12:
                continue
            logp = np.log(qs_ctx + 1e-12)
            for ctx in (REWARD_LEFT, REWARD_RIGHT):
                logp[ctx] += np.log(likelihood(o, ctx) + 1e-12)
            h_post += po * float(_entropy(_softmax(logp)))
        return h_prior - h_post

    def expected_free_energy(self, loc: int, qs_ctx: FloatArray, policy: IntArray) -> float:
        """Expected score of a fixed open-loop policy (for inspection)."""
        if not len(policy) or not available_actions(loc):
            return 0.0
        action = int(policy[0])
        if action not in available_actions(loc):
            return float("inf")
        return self._immediate(action, qs_ctx) + sum(
            probability * self.expected_free_energy(action, posterior, policy[1:])
            for probability, posterior in self._outcomes(action, qs_ctx)
        )

    def _outcomes(self, action: int, belief: FloatArray):
        for cue in range(N_CUE):
            for rew in range(N_REW):
                joint = belief * np.array([
                    self._p_cue(action, ctx, cue) * self._p_rew(action, ctx, rew)
                    for ctx in (REWARD_LEFT, REWARD_RIGHT)
                ])
                probability = float(joint.sum())
                if probability > 1e-12:
                    yield probability, joint / probability

    def _immediate(self, action: int, belief: FloatArray) -> float:
        utility = float(self._predict_obs(action, belief)[1] @ self.C_rew)
        return -utility - self.epistemic_weight * self._info_gain_one_channel(action, belief)

    def action_scores(self, loc: int, belief: FloatArray, horizon: int) -> dict[int, float]:
        def continuation(action, posterior):
            if horizon <= 1 or not available_actions(action):
                return 0.0
            return min(self.action_scores(action, posterior, horizon - 1).values())
        return {
            action: self._immediate(action, belief) + sum(
                probability * continuation(action, posterior)
                for probability, posterior in self._outcomes(action, belief)
            )
            for action in available_actions(loc)
        }

    def act(self, loc: int, horizon: int | None = None) -> int:
        scores = self.action_scores(loc, self.qs_ctx, self.horizon if horizon is None else horizon)
        if not scores:
            raise ValueError("The trial has ended.")
        return min(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------


class QLearningAgent:
    """ε-greedy Q-learning on (moves left, location, cue memory)."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 1.0,
        epsilon: float = 0.25,
        seed: int = DEFAULT_SEED,
        max_steps: int = 3,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        # cue memory: 0 = unknown, 1 = left, 2 = right
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.Q = np.zeros((max_steps + 1, N_LOC, 3, N_LOC))
        self.cue_mem = 0

    def reset(self, n_steps: int | None = None) -> None:
        self.steps_left = self.max_steps if n_steps is None else n_steps
        if not 1 <= self.steps_left <= self.max_steps:
            raise ValueError("Trial length exceeds the configured Q-table horizon.")
        self.cue_mem = 0

    def state(self, loc: int) -> tuple[int, int, int]:
        return self.steps_left, loc, self.cue_mem

    def act(self, loc: int) -> int:
        actions = np.array(available_actions(loc))
        if not len(actions) or self.steps_left == 0:
            raise ValueError("The trial has ended.")
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        q = self.Q[self.steps_left, loc, self.cue_mem, actions]
        best = actions[np.flatnonzero(q >= q.max() - 1e-12)]
        return int(self.rng.choice(best))

    def update(self, loc: int, action: int, reward: float, loc_next: int, cue: int) -> None:
        s = self.state(loc)
        if cue in (1, 2):
            self.cue_mem = cue
        self.steps_left -= 1
        s2 = self.state(loc_next)
        actions = available_actions(loc_next)
        future = float(np.max(self.Q[s2][list(actions)])) if actions and self.steps_left else 0.0
        chosen = (*s, action)
        td = reward + self.gamma * future - self.Q[chosen]
        self.Q[chosen] += self.alpha * td


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@dataclass
class TrialTrace:
    locations: list[int]
    rewards: list[float]
    visited_cue_first: bool
    context: int


def cue_before_arm(locations: list[int]) -> bool:
    for loc in locations[1:]:
        if loc == CUE:
            return True
        if loc in (LEFT, RIGHT):
            return False
    return False


def run_trial_aif(
    env: TMaze,
    agent: ActiveInferenceAgent,
    n_steps: int = 3,
    context: int | None = None,
) -> TrialTrace:
    loc, cue, rew = env.reset(context=context)
    agent.reset()
    agent.infer_context(loc, cue, rew)
    locations = [loc]
    rewards = [0.0]
    for step in range(n_steps):
        if not available_actions(loc):
            break
        action = agent.act(loc, horizon=min(agent.horizon, n_steps - step))
        (loc, cue, rew), r = env.step(action)
        agent.infer_context(loc, cue, rew)
        locations.append(loc)
        rewards.append(r)
    return TrialTrace(locations, rewards, cue_before_arm(locations), env.context)


def run_trial_ql(
    env: TMaze,
    agent: QLearningAgent,
    n_steps: int = 3,
    context: int | None = None,
    learn: bool = True,
) -> TrialTrace:
    loc, cue, rew = env.reset(context=context)
    agent.reset(n_steps=n_steps)
    if cue in (1, 2):
        agent.cue_mem = cue
    locations = [loc]
    rewards = [0.0]
    for _ in range(n_steps):
        if not available_actions(loc):
            break
        action = agent.act(loc)
        (loc2, cue, rew), r = env.step(action)
        if learn:
            agent.update(loc, action, r, loc2, cue)
        else:
            agent.steps_left -= 1
            if cue in (1, 2):
                agent.cue_mem = cue
        loc = loc2
        locations.append(loc)
        rewards.append(r)
    return TrialTrace(locations, rewards, cue_before_arm(locations), env.context)


def run_experiment(
    n_trials: int = 400,
    n_steps: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[TrialTrace]]:
    contexts = np.random.default_rng(seed).integers(0, 2, size=n_trials)
    env_aif = TMaze(seed=seed)
    env_ql = TMaze(seed=seed)
    env_reward = TMaze(seed=seed)
    aif = ActiveInferenceAgent()
    ql = QLearningAgent(seed=seed + 2, max_steps=n_steps)
    reward_only = ActiveInferenceAgent(epistemic_weight=0.0)
    traces_aif = [run_trial_aif(env_aif, aif, n_steps=n_steps, context=int(c)) for c in contexts]
    traces_ql = [run_trial_ql(env_ql, ql, n_steps=n_steps, context=int(c)) for c in contexts]
    traces_reward = [run_trial_aif(env_reward, reward_only, n_steps=n_steps, context=int(c)) for c in contexts]
    return {"aif": traces_aif, "ql": traces_ql, "reward_only": traces_reward}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_XY = {
    CENTER: (0.0, 0.0),
    CUE: (0.0, -1.15),
    LEFT: (-1.15, 0.85),
    RIGHT: (1.15, 0.85),
}


def _draw_maze(ax: plt.Axes, title: str) -> None:
    for a, b in ((CENTER, CUE), (CENTER, LEFT), (CENTER, RIGHT)):
        x0, y0 = _XY[a]
        x1, y1 = _XY[b]
        ax.plot([x0, x1], [y0, y1], color="0.75", lw=6, solid_capstyle="round", zorder=0)
    colors = {CENTER: "0.45", CUE: "#2a6f97", LEFT: "#c44536", RIGHT: "#c44536"}
    labels = {CENTER: "start", CUE: "cue", LEFT: "L", RIGHT: "R"}
    for loc, (x, y) in _XY.items():
        ax.add_patch(Circle((x, y), 0.22, fc="white", ec=colors[loc], lw=1.6, zorder=2))
        ax.text(x, y, labels[loc], ha="center", va="center", fontsize=7, zorder=3)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.65, 1.35)


def _draw_path(ax: plt.Axes, locations: list[int], color: str) -> None:
    pts = np.array([_XY[loc] for loc in locations], dtype=float)
    # slight offset so overlapping paths stay visible
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, alpha=0.9, zorder=1)
    ax.scatter(pts[0, 0], pts[0, 1], s=18, color=color, zorder=4)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=28, color=color, marker="*", zorder=4)


def render_paper_figure(
    traces: dict[str, list[TrialTrace]] | None = None,
    seed: int = DEFAULT_SEED,
) -> plt.Figure:
    if traces is None:
        traces = run_experiment(seed=seed)
    aif, ql = traces["aif"], traces["ql"]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig = plt.figure(figsize=(6.3, 5.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1.0])
    ax_aif, ax_ql = top.subplots(1, 2)
    ax_cue, ax_rew = bottom.subplots(1, 2)

    _draw_maze(ax_aif, "A   Known-model utility + IG, first trial")
    _draw_path(ax_aif, aif[0].locations, "#1b6ca8")
    ctx = "reward left" if aif[0].context == REWARD_LEFT else "reward right"
    ax_aif.text(0, -1.58, f"context: {ctx}", ha="center", fontsize=7, color="0.3")

    _draw_maze(ax_ql, "A′  Q-learning, first trial")
    _draw_path(ax_ql, ql[0].locations, "#b23a48")
    ctx = "reward left" if ql[0].context == REWARD_LEFT else "reward right"
    ax_ql.text(0, -1.58, f"context: {ctx}", ha="center", fontsize=7, color="0.3")

    trials = np.arange(1, len(aif) + 1)
    cue_aif = np.array([t.visited_cue_first for t in aif], dtype=float)
    cue_ql = np.array([t.visited_cue_first for t in ql], dtype=float)
    rew_aif = np.array([sum(t.rewards) for t in aif])
    rew_ql = np.array([sum(t.rewards) for t in ql])

    def _cummean(x: FloatArray) -> FloatArray:
        return np.cumsum(x) / np.arange(1, len(x) + 1)

    ax_cue.plot(trials, _cummean(cue_aif), color="#1b6ca8", lw=1.6, label="known model, utility + IG")
    ax_cue.plot(trials, _cummean(cue_ql), color="#b23a48", lw=1.6, label="Q-learning")
    ax_cue.set_ylim(-0.05, 1.05)
    ax_cue.set_xlabel("Trial")
    ax_cue.set_ylabel("Cumulative cue-first rate")
    ax_cue.set_title("B   Information seeking", loc="left", fontsize=9, fontweight="bold")
    ax_cue.legend(frameon=False, fontsize=7)
    ax_cue.spines["top"].set_visible(False)
    ax_cue.spines["right"].set_visible(False)

    ax_rew.plot(trials, _cummean(rew_aif), color="#1b6ca8", lw=1.6, label="known model, utility + IG")
    ax_rew.plot(trials, _cummean(rew_ql), color="#b23a48", lw=1.6, label="Q-learning")
    if "reward_only" in traces:
        control = traces["reward_only"]
        ax_cue.plot(trials, _cummean(np.array([t.visited_cue_first for t in control], dtype=float)),
                    color="#397548", ls="--", label="known model, reward only")
        ax_rew.plot(trials, _cummean(np.array([sum(t.rewards) for t in control])),
                   color="#397548", ls="--", label="known model, reward only")
        ax_cue.legend(frameon=False, fontsize=6)
    ax_rew.set_xlabel("Trial")
    ax_rew.set_ylabel("Cumulative mean return")
    ax_rew.set_title("C   Reward collected", loc="left", fontsize=9, fontweight="bold")
    ax_rew.legend(frameon=False, fontsize=7)
    ax_rew.spines["top"].set_visible(False)
    ax_rew.spines["right"].set_visible(False)
    return fig


def write_figure(
    traces: dict[str, list[TrialTrace]] | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(traces, seed=seed)
    pdf = out / "tmaze_aif_vs_ql.pdf"
    png = out / "tmaze_aif_vs_ql.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def save_paper_figure(out_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> Path:
    set_seed(seed)
    traces = run_experiment(seed=seed)
    aif0 = traces["aif"][0]
    print(
        f"AIF first trial: { [LOC_NAMES[i] for i in aif0.locations] }  "
        f"return {sum(aif0.rewards):+.0f}"
    )
    ql0 = traces["ql"][0]
    print(
        f"QL  first trial: { [LOC_NAMES[i] for i in ql0.locations] }  "
        f"return {sum(ql0.rewards):+.0f}"
    )
    return write_figure(traces, out_dir, seed=seed)


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
