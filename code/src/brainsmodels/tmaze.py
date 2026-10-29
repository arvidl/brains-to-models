"""T-maze: active inference versus Q-learning.

Classic four-location maze (center, cue, left arm, right arm). A hidden
context places the reward in one arm; the cue location reveals it without
risk. Active inference visits the cue first because expected free energy
has an epistemic term. ε-greedy Q-learning must stumble into the same
structure through reward.

The task is the one used in the pymdp T-maze demo; the EFE loop here is
NumPy so the figure stays dependency-light.

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
        self.location = int(action)
        loc, cue, rew = self.observe()
        reward = {0: 0.0, 1: 1.0, 2: -1.0}[rew]
        return (loc, cue, rew), reward


# ---------------------------------------------------------------------------
# Active inference (receding-horizon EFE)
# ---------------------------------------------------------------------------


class ActiveInferenceAgent:
    """Known generative model; policies scored by expected free energy."""

    def __init__(
        self,
        horizon: int = 2,
        gamma: float = 8.0,
        cue_accuracy: float = 1.0,
        prefer_reward: float = 2.0,
        prefer_punish: float = -4.0,
    ) -> None:
        self.horizon = horizon
        self.gamma = gamma
        self.cue_accuracy = cue_accuracy
        self.C_rew = np.array([0.0, prefer_reward, prefer_punish])
        self.qs_ctx = np.array([0.5, 0.5])
        self.policies = np.array(
            np.meshgrid(*[np.arange(N_LOC)] * horizon)
        ).T.reshape(-1, horizon)

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
        g = 0.0
        here = loc
        belief = qs_ctx.copy()
        for action in policy:
            here = int(action)
            _, p_rew = self._predict_obs(here, belief)
            pragmatic = float(p_rew @ self.C_rew)
            epistemic = self._info_gain_one_channel(here, belief)
            g += -pragmatic - epistemic
            # update predicted belief as if the most likely informative obs arrived
            if here == CUE:
                p_cue, _ = self._predict_obs(here, belief)
                o = int(np.argmax(p_cue))
                logp = np.log(belief + 1e-12)
                for ctx in (REWARD_LEFT, REWARD_RIGHT):
                    logp[ctx] += np.log(self._p_cue(here, ctx, o) + 1e-12)
                belief = _softmax(logp)
            elif here in (LEFT, RIGHT):
                o = int(np.argmax(p_rew))
                logp = np.log(belief + 1e-12)
                for ctx in (REWARD_LEFT, REWARD_RIGHT):
                    logp[ctx] += np.log(self._p_rew(here, ctx, o) + 1e-12)
                belief = _softmax(logp)
        return g

    def act(self, loc: int) -> int:
        scores = np.array(
            [self.expected_free_energy(loc, self.qs_ctx, pi) for pi in self.policies]
        )
        # minimize G; softmax over -gamma G
        q_pi = _softmax(-self.gamma * scores)
        return int(self.policies[int(np.argmax(q_pi)), 0])


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------


class QLearningAgent:
    """ε-greedy tabular Q-learning. State = (location, cue memory)."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.9,
        epsilon: float = 0.25,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        # cue memory: 0 = unknown, 1 = left, 2 = right
        self.Q = np.zeros((N_LOC, 3, N_LOC))
        self.cue_mem = 0

    def reset(self) -> None:
        self.cue_mem = 0

    def state(self, loc: int) -> tuple[int, int]:
        return loc, self.cue_mem

    def act(self, loc: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_LOC))
        q = self.Q[loc, self.cue_mem]
        best = np.flatnonzero(q >= q.max() - 1e-12)
        return int(self.rng.choice(best))

    def update(self, loc: int, action: int, reward: float, loc_next: int, cue: int) -> None:
        s = (loc, self.cue_mem)
        if cue in (1, 2):
            self.cue_mem = cue
        s2 = (loc_next, self.cue_mem)
        td = reward + self.gamma * np.max(self.Q[s2]) - self.Q[s]
        self.Q[s] += self.alpha * td


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@dataclass
class TrialTrace:
    locations: list[int]
    rewards: list[float]
    visited_cue_first: bool
    context: int


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
    for _ in range(n_steps):
        action = agent.act(loc)
        (loc, cue, rew), r = env.step(action)
        agent.infer_context(loc, cue, rew)
        locations.append(loc)
        rewards.append(r)
    return TrialTrace(locations, rewards, CUE in locations[1:], env.context)


def run_trial_ql(
    env: TMaze,
    agent: QLearningAgent,
    n_steps: int = 3,
    context: int | None = None,
    learn: bool = True,
) -> TrialTrace:
    loc, cue, rew = env.reset(context=context)
    agent.reset()
    if cue in (1, 2):
        agent.cue_mem = cue
    locations = [loc]
    rewards = [0.0]
    for _ in range(n_steps):
        action = agent.act(loc)
        (loc2, cue, rew), r = env.step(action)
        if learn:
            agent.update(loc, action, r, loc2, cue)
        else:
            if cue in (1, 2):
                agent.cue_mem = cue
        loc = loc2
        locations.append(loc)
        rewards.append(r)
    return TrialTrace(locations, rewards, CUE in locations[1:], env.context)


def run_experiment(
    n_trials: int = 40,
    n_steps: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[TrialTrace]]:
    env_aif = TMaze(seed=seed)
    env_ql = TMaze(seed=seed + 1)
    aif = ActiveInferenceAgent()
    ql = QLearningAgent(seed=seed + 2)
    traces_aif = [run_trial_aif(env_aif, aif, n_steps=n_steps) for _ in range(n_trials)]
    traces_ql = [run_trial_ql(env_ql, ql, n_steps=n_steps) for _ in range(n_trials)]
    return {"aif": traces_aif, "ql": traces_ql}


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

    _draw_maze(ax_aif, "A   Active inference, first trial")
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

    ax_cue.plot(trials, _cummean(cue_aif), color="#1b6ca8", lw=1.6, label="active inference")
    ax_cue.plot(trials, _cummean(cue_ql), color="#b23a48", lw=1.6, label="Q-learning")
    ax_cue.set_ylim(-0.05, 1.05)
    ax_cue.set_xlabel("Trial")
    ax_cue.set_ylabel("Cumulative cue-first rate")
    ax_cue.set_title("B   Epistemic foraging", loc="left", fontsize=9, fontweight="bold")
    ax_cue.legend(frameon=False, fontsize=7)
    ax_cue.spines["top"].set_visible(False)
    ax_cue.spines["right"].set_visible(False)

    ax_rew.plot(trials, _cummean(rew_aif), color="#1b6ca8", lw=1.6, label="active inference")
    ax_rew.plot(trials, _cummean(rew_ql), color="#b23a48", lw=1.6, label="Q-learning")
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
