"""Write exercise solution files and append Exercises cells to notebooks 01–07."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOL = Path(__file__).resolve().parent
NB = ROOT / "code" / "notebooks"

HEADER = '''\
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
plt.rcParams.update({{"figure.dpi": 120, "font.size": 10}})
'''

# notebook_stem -> list of (slug, title, prompt, extra_imports, body)
EXERCISES: dict[str, list[tuple[str, str, str, str, str]]] = {}


def add(nb: str, slug: str, title: str, prompt: str, imports: str, body: str) -> None:
    EXERCISES.setdefault(nb, []).append((slug, title, prompt, imports, body))


add(
    "01_tiny_transformer",
    "01_1_drop_the_causal_mask",
    "Drop the causal mask",
    """The black upper triangle in panel A is not decoration: position $t$ is forbidden from seeing $t{+}1,t{+}2,\\ldots$. Recompute the attention weights from the trained $Q$ and $K$ *without* that mask. How much mass does a mid-prompt query put on characters that have not been written yet? What would go wrong at generation time if the model had been allowed to learn that look-ahead?""",
    """from brainsmodels.tiny_transformer import _softmax, train_demo
""",
    '''
model, tokenizer, text, _ = train_demo(seed=SEED, n_steps=200)
prompt = "the cat sat on the "
ids = tokenizer.encode(prompt)
logits, causal = model.forward(ids, record_attn=True)
q, k = model._cache["q"], model._cache["k"]
scale = np.sqrt(model._d_head)
scores = np.einsum("bhtd,bhsd->bhts", q, k) / scale
leaky = _softmax(scores, axis=-1)

t = len(prompt)
# Query at the space after "cat" (index of that space).
q_at = prompt.index(" ", 4)
future = slice(q_at + 1, t)
leaked = float(leaky[0, :, q_at, future].sum() / model.n_heads)
blocked = float(causal[0, :, q_at, future].sum() / model.n_heads)
print(f"query at index {q_at} ({prompt[q_at]!r})")
print(f"mean mass on later characters, causal mask on : {blocked:.3f}")
print(f"mean mass on later characters, mask dropped   : {leaked:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
labels = [("␣" if c == " " else c) for c in prompt]
for ax, mat, title in (
    (axes[0], causal[0, 0, :t, :t], "head 1, causal"),
    (axes[1], leaky[0, 0, :t, :t], "head 1, no mask"),
):
    ax.imshow(mat, cmap="magma", vmin=0, vmax=max(0.2, float(mat.max())))
    ax.set_xticks(range(t), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(t), labels, fontsize=7)
    ax.set_title(title, fontsize=9)
fig.suptitle("Without the mask, a query can look at characters that do not yet exist")
plt.tight_layout()
plt.show()
print(
    "Generation writes one character at a time. A look-ahead weight is a weight "
    "on a blank. Training with the mask dropped would let the loss cheat on the "
    "training window (the future tokens are already in the batch) and would not "
    "transfer to left-to-right sampling. The mask is what makes next-token "
    "prediction a well-posed online objective, not a better picture of a cat."
)
''',
)

add(
    "01_tiny_transformer",
    "01_2_a_unigram_is_not_enough",
    "A unigram is not enough",
    """Panel B falls well below $\\ln|\\mathcal{{V}}|$. How much of that drop is just “some letters are common”? Fit a unigram and a bigram on the same designed corpus and compare their next-character NLL on held-out windows to the trained transformer. What does the remaining gap say about what attention is for here?""",
    """from brainsmodels.tiny_transformer import CharTokenizer, make_corpus, train_demo
""",
    '''
model, tokenizer, text, _ = train_demo(seed=SEED, n_steps=200)
ids = tokenizer.encode(text)
n = len(ids)
cut = int(0.85 * n)
train_ids, test_ids = ids[:cut], ids[cut:]

unigram = np.ones(tokenizer.vocab_size)
for i in train_ids:
    unigram[int(i)] += 1.0
unigram = unigram / unigram.sum()

bigram = np.ones((tokenizer.vocab_size, tokenizer.vocab_size))
for a, b in zip(train_ids[:-1], train_ids[1:]):
    bigram[int(a), int(b)] += 1.0
bigram = bigram / bigram.sum(axis=1, keepdims=True)

def nll_unigram(seq):
    return float(-np.mean(np.log(unigram[seq])))

def nll_bigram(seq):
    return float(-np.mean(np.log(bigram[seq[:-1], seq[1:]])))

def nll_model(seq, block=32):
    losses = []
    for start in range(0, len(seq) - block - 1, block):
        ctx = seq[start : start + block]
        tgt = seq[start + 1 : start + 1 + block]
        logits, _ = model.forward(ctx)
        p = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        losses.append(-np.log(np.clip(p[np.arange(block), tgt], 1e-9, 1.0)).mean())
    return float(np.mean(losses))

print(f"chance (ln V)   {np.log(tokenizer.vocab_size):.3f}")
print(f"unigram  test   {nll_unigram(test_ids):.3f}")
print(f"bigram   test   {nll_bigram(test_ids):.3f}")
print(f"1-layer  test   {nll_model(test_ids):.3f}")
print(
    "\\nThe unigram knows that space and e are common. The bigram knows that "
    "'t' often follows 'a' in 'cat' / 'sat' / 'mat'. The transformer still "
    "wins: the shared sentence frame is longer than one previous character. "
    "Attention is a look-back over a window, not a better unigram. It is still "
    "not a world model."
)
''',
)

add(
    "01_tiny_transformer",
    "01_3_temperature_is_not_a_belief",
    "Temperature is not a belief",
    """`generate` has a `temperature` argument. Sample 20 continuations of `the cat sat on the ` at $T=0$ (argmax), $T=0.4$, and $T=1.2$. How does the set of object-words change? Why is a sharper sample not a more committed belief about a cat?""",
    """from brainsmodels.tiny_transformer import train_demo
""",
    '''
model, tokenizer, _, _ = train_demo(seed=SEED, n_steps=400)
prompt = "the cat sat on the "
temps = (0.0, 0.4, 1.2)
objects = ("mat", "hat", "log", "box")

for T in temps:
    texts = [
        model.generate(prompt, tokenizer, n_tokens=12, temperature=T, seed=SEED + k)
        for k in range(20)
    ]
    tails = [t[len(prompt) :].split()[0] if t[len(prompt) :].split() else "" for t in texts]
    hits = {w: sum(w in tail for tail in tails) for w in objects}
    uniq = len(set(tails))
    print(f"T={T:<4}  unique first-words {uniq:2d}  object hits {hits}")
    print("   e.g.", texts[0][len(prompt) : 28])

print(
    "\\nT=0 repeats the mode. Raising T spreads mass; more object-words appear, "
    "and so do broken spellings. Temperature rescales a categorical. It does not "
    "add a world that could make one completion true of a particular cat. A "
    "confident sample is a peaked next-character distribution, not a belief."
)
''',
)

add(
    "02_predictive_coding",
    "02_1_inference_not_learning",
    "Inference is not learning",
    """Panel C is the residual falling *inside one patch*, with the weights held fixed. Freeze a trained network, then reconstruct a held-out patch after 1, 5, and 40 inference steps. Then take the *untrained* initial weights and do the same. How much of the falling residual is the hierarchy explaining the patch, and how much is just “any basis plus enough iterations”?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '''
data = PatchDataset.from_pink_noise(n_train=500, n_test=16, n_images=8, seed=SEED)
x = data.test[0]
fresh = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)
trained = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)
trained.train(data.train, n_epochs=6, n_infer=20, seed=SEED)

def curve(model, steps=40):
    _, mse = model.infer_history(x, n_steps=steps)
    return np.asarray(mse)

c0, c1 = curve(fresh), curve(trained)
print(f"untrained  MSE t=1 {c0[0]:.3f}  t=40 {c0[-1]:.3f}")
print(f"trained    MSE t=1 {c1[0]:.3f}  t=40 {c1[-1]:.3f}")
print(f"drop untrained {c0[0] - c0[-1]:.3f}   drop trained {c1[0] - c1[-1]:.3f}")

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(c0, label="untrained weights")
ax.plot(c1, label="after 6 epochs")
ax.set_xlabel("inference step")
ax.set_ylabel("prediction-error MSE")
ax.legend(frameon=False)
ax.set_title("Same infer loop; only the descending weights changed")
plt.tight_layout()
plt.show()
print(
    "Both curves fall: inference is gradient descent on the current error, "
    "and even random columns of U1 span something. The first step is similar "
    "because r starts at zero; the trained curve ends lower. Learning moved "
    "the basis; inference explains the present patch in that basis. This is "
    "still not V1."
)
''',
)

add(
    "02_predictive_coding",
    "02_2_one_level",
    "One level",
    """The network is two-level: $r_2$ predicts $r_1$, $r_1$ predicts pixels. Train a copy with `n_level2=1` (almost no higher code) and one with the default 12 units, same data and seed. Compare held-out reconstruction MSE after the same number of inference steps. What does the higher level buy on these 8×8 patches, and what can it not buy?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '''
data = PatchDataset.from_pink_noise(n_train=500, n_test=24, n_images=8, seed=SEED)
shallow = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=1, seed=SEED)
deep = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=12, seed=SEED)
shallow.train(data.train, n_epochs=6, n_infer=20, seed=SEED)
deep.train(data.train, n_epochs=6, n_infer=20, seed=SEED + 1)

def recon_mse(model):
    _, _, tr = model.infer(data.test, n_steps=40, record=True)
    return float(np.mean(tr.error**2))

print(f"n_level2=1   held-out MSE {recon_mse(shallow):.4f}")
print(f"n_level2=12  held-out MSE {recon_mse(deep):.4f}")
print(
    "\\nOn 8×8 pink-noise patches the second level is a small extra code for "
    "the first-level coefficients, not a scene. A better reconstruction is "
    "not evidence of a world model, and a flat second level is not evidence "
    "that cortex is one layer. The paper's contrast is the *job*: predict "
    "the current sensory vector, not the next character of a corpus."
)
''',
)

add(
    "02_predictive_coding",
    "02_3_skip_the_whitening",
    "Skip the whitening",
    """The descending weights are learned in ZCA-whitened coordinates, then mapped back to pixels for panel A. Train the same architecture on the *unwhitened* pixel patches (`data.train_pixel`). Do the first-level columns still look localized and oriented when you plot them as 8×8 images? What did whitening buy, and what did it not invent?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '''
data = PatchDataset.from_pink_noise(n_train=500, n_test=16, n_images=8, seed=SEED)
pix = HierarchicalPredictiveCoding(n_input=data.train_pixel.shape[1], seed=SEED)
zca = HierarchicalPredictiveCoding(n_input=data.train.shape[1], seed=SEED)
pix.train(data.train_pixel, n_epochs=6, n_infer=20, seed=SEED)
zca.train(data.train, n_epochs=6, n_infer=20, seed=SEED)

rf_pix = pix.U1
rf_zca = data.whitening.to_pixel_basis(zca.U1)
energy = rf_zca.reshape(data.patch_size, data.patch_size, -1).var(axis=(0, 1))
order = np.argsort(energy)[::-1][:8]

fig, axes = plt.subplots(2, 8, figsize=(9.5, 2.8))
for col, i in enumerate(order):
    for row, bank, name in ((0, rf_pix, "pixels"), (1, rf_zca, "ZCA→pixels")):
        patch = bank[:, i].reshape(data.patch_size, data.patch_size)
        vmax = np.max(np.abs(patch)) + 1e-8
        axes[row, col].imshow(patch, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=8)
fig.suptitle("Whitening changes the coordinates the Hebbian rule sees")
plt.tight_layout()
plt.show()
print(
    "Without whitening the columns spend their variance on low-frequency "
    "blobs (the 1/f spectrum). ZCA equalizes those coordinates so a local "
    "rule can spend capacity on oriented structure that is already in the "
    "gratings. Whitening is a change of basis, not a retina, and the "
    "dictionary is still not V1."
)
''',
)

add(
    "02b_active_inference_vs_rl",
    "02b_1_drop_the_epistemic_term",
    "Drop the epistemic term",
    """Expected free energy is $-$pragmatic value $-$ information gain. Zero `_info_gain_one_channel` on a horizon-2 agent, then again on a horizon-1 agent. Who still visits the cue? What two ingredients were doing the work?""",
    """from brainsmodels.tmaze import ActiveInferenceAgent, TMaze, run_trial_aif
""",
    '''
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
    "\\nDropping the epistemic bonus is not enough if the horizon can still "
    "plan *cue then arm*: the cue remains a means to a later preferred "
    "observation. A one-step agent with no information gain has no such "
    "plan and stays put. Two ingredients, not one slogan. Still a "
    "four-location teaching model, not a rat."
)
''',
)

add(
    "02b_active_inference_vs_rl",
    "02b_2_a_noisy_cue",
    "A noisy cue",
    """The environment and the agent both default to `cue_accuracy=1`. Set both to $0.6$ and compare cue-first rate and mean reward over 40 trials against the perfect-cue agent. When does “go to the cue first” stop being worth the step?""",
    """from brainsmodels.tmaze import ActiveInferenceAgent, TMaze, run_trial_aif
""",
    '''
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
    "\\nAt 0.5 the cue is a coin flip and information gain collapses. The "
    "policy is then a bet on an arm. Epistemic value tracks how much an "
    "observation would change the context belief, not a virtue of looking. "
    "A noisy instrument in clinic is the same arithmetic: a test that does "
    "not change the posterior is not worth the step, and that is still not "
    "a rat."
)
''',
)

add(
    "02b_active_inference_vs_rl",
    "02b_3_ql_needs_the_stumble",
    "Q-learning needs the stumble",
    """ε-greedy Q-learning has no information-gain term. After 80 trials, compare its cue-first rate to active inference’s, and print $Q$ from the start state for cue vs the two arms. Has the cue become the greedy action? What is still missing even if it later does?""",
    """from brainsmodels.tmaze import (
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
""",
    '''
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
    "\\nActive inference visits the cue on the first trial. Q-learning "
    "has only the stumbles that ε-greedy happened to take, and a three-step "
    "return is a harsh credit assignment. Even if a longer budget made "
    "the cue greedy, the goal would still be an external scalar, not a "
    "preferred observation, and not a body that must keep going."
)
''',
)

add(
    "03_bio_vs_artificial",
    "03_1_xor_needs_a_hidden_layer",
    "XOR needs a hidden layer",
    """The notebook already trains both learners on the four-blob XOR. Report train accuracy for the spiking perceptron and for the MLP, then for an MLP with `n_hidden=1`. Which failure is about *local spike-count updates*, and which is about *having only a linear readout*?""",
    """from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_xor
""",
    '''
x, y = make_xor(n=200, seed=SEED)
spk = SpikingPerceptron(seed=SEED)
mlp = SmallMLP(n_hidden=16, seed=SEED)
thin = SmallMLP(n_hidden=1, seed=SEED)
spk.train(x, y, n_epochs=30, seed=SEED)
mlp.train(x, y, n_epochs=30)
thin.train(x, y, n_epochs=30)
print(f"LIF perceptron (2→2, local delta)     acc {np.mean(spk.predict(x, seed=SEED) == y):.2f}")
print(f"MLP 2→16→2, SGD                       acc {np.mean(mlp.predict(x) == y):.2f}")
print(f"MLP 2→1→2, SGD                        acc {np.mean(thin.predict(x) == y):.2f}")
print(
    "\\nThe thin MLP is still SGD, and it still cannot draw the XOR "
    "partition: one hidden unit is a single half-plane. The LIF net is "
    "a different failure: two output cells and a local count rule, no "
    "hidden layer to assign credit through. Same points, two reasons to "
    "miss. Neither miss is a synapse in cortex."
)
''',
)

add(
    "03_bio_vs_artificial",
    "03_2_same_points_different_update",
    "Same points, different update",
    """On the linearly separable Gaussians both learners succeed. After training, scatter the two weight columns of the LIF net and the first-layer weights of the MLP (plot the 2-D input weights). The decision is similar; the objects that were updated are not. What was incremented in each case?""",
    """from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_linear
""",
    '''
x, y = make_linear(n=200, seed=SEED)
spk = SpikingPerceptron(seed=SEED)
mlp = SmallMLP(n_hidden=16, seed=SEED)
spk.train(x, y, n_epochs=25, seed=SEED)
mlp.train(x, y, n_epochs=25)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, pts, c, lab in (
    (axes[0], x[y == 0], "#4a7c8c", "class 0"),
    (axes[0], x[y == 1], "#c47a3a", "class 1"),
):
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=c, label=lab, alpha=0.7)
for k, col in enumerate(("#2c5f6e", "#9a4a2a")):
    w = spk.w[:, k]
    axes[0].arrow(0, 0, w[0], w[1], color=col, width=0.02, head_width=0.12)
    axes[0].set_title("LIF: two columns of W (2×2)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False)
axes[0].set_aspect("equal")

im = axes[1].imshow(mlp.w1.T, cmap="coolwarm", aspect="auto")
axes[1].set_title("MLP: W1 (2 → 16), updated by SGD", fontsize=9)
axes[1].set_xlabel("input dim")
axes[1].set_ylabel("hidden unit")
fig.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.show()
print(
    "LIF: each synapse saw a pre-count and a post-count. MLP: every hidden "
    "weight saw a scalar that had already been multiplied by the downstream "
    "error. Same point cloud; the elementary increment is not the same "
    "object. That is the §4 claim, not a verdict on which rule is 'more biological'."
)
''',
)

add(
    "03_bio_vs_artificial",
    "03_3_leak_is_not_depth",
    "Leak is not depth",
    """The LIF cell has a `leak` (how fast $v$ forgets) and a `threshold`. Retrain on the linear task at `leak=0.05` and `leak=0.80`, same seed. Does accuracy move more than it did when you removed the MLP's hidden layer in exercise 1? What kind of difference is leak?""",
    """from brainsmodels.bio_vs_artificial import SpikingPerceptron, make_linear
""",
    '''
x, y = make_linear(n=200, seed=SEED)
for leak in (0.05, 0.20, 0.80):
    net = SpikingPerceptron(leak=leak, seed=SEED)
    net.train(x, y, n_epochs=25, seed=SEED)
    acc = float(np.mean(net.predict(x, seed=SEED) == y))
    print(f"leak {leak:.2f}  train acc {acc:.2f}")
print(
    "\\nLeak changes how long a spike is remembered inside one trial. It is "
    "a time-constant, not a layer. On a linearly separable cloud it barely "
    "moves the score; XOR would still be out of reach. Do not read a "
    "membrane parameter as architectural depth, and do not read this cell "
    "as cortex."
)
''',
)

add(
    "04_quadrants_biases",
    "04_1_interpolate_the_corpus",
    "Interpolate the corpus",
    """The notebook trains a fully biased corpus and a balanced one. Train four word-level models with `gain_p_sure` in $\\{1.0, 0.75, 0.5, 0.25\\}$ (keep `loss_p_sure = 1 - gain_p_sure`). Plot $P(\\texttt{{sure}}\\mid\\text{{gain probe}})$ against the corpus rate. How close is the next-token answer to a copy of the training frequency?""",
    """from brainsmodels.biases import make_crawl, train_on_crawl
""",
    '''
rates = (1.0, 0.75, 0.5, 0.25)
model_p = []
for p in rates:
    crawl = make_crawl(n_sentences=240, seed=SEED, gain_p_sure=p, loss_p_sure=1.0 - p)
    run = train_on_crawl(crawl, seed=SEED, n_steps=350)
    model_p.append(run.gain_sure)
    print(f"corpus P(sure|gain) {p:.2f}  model {run.gain_sure:.2f}")

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.plot([0, 1], [0, 1], color="0.75", lw=1)
ax.plot(rates, model_p, "o-", color="#2c5f6e")
ax.set_xlabel("corpus P(sure | gain frame)")
ax.set_ylabel("model P(sure | gain probe)")
ax.set_title("Next-token mass tracks the designed frequency")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()
print(
    "The probe is held out as a *choice*, not as a sentence type. The model "
    "still copies the frequency it was trained on. That is the §5 mechanism, "
    "not a re-run of Tversky and Kahneman, and not a bias in a person."
)
''',
)

add(
    "04_quadrants_biases",
    "04_2_a_held_out_wording",
    "A held-out wording",
    """The training templates always continue with `most people choose the {{choice}} option`. Probe with a wording built from the same vocabulary but a new continuation: `if program one is adopted two hundred people will be saved . people choose the`. Does $P(\\texttt{{sure}})$ survive the omitted `most`?""",
    """from brainsmodels.biases import make_crawl, next_token_probs, sure_rate, train_on_crawl
""",
    '''
crawl = make_crawl(n_sentences=300, seed=SEED, gain_p_sure=1.0, loss_p_sure=0.0)
run = train_on_crawl(crawl, seed=SEED, n_steps=350)
novel = "if program one is adopted two hundred people will be saved . people choose the"
probs = next_token_probs(run.model, run.tokenizer, novel)
print("trained probe  P(sure|gain)", f"{run.gain_sure:.2f}")
ps = float(probs[run.tokenizer.stoi["sure"]])
pr = float(probs[run.tokenizer.stoi["risky"]])
top = sorted(run.tokenizer.stoi, key=lambda w: -float(probs[run.tokenizer.stoi[w]]))[:6]
print(f"novel probe    P(sure) {ps:.3f}  P(risky) {pr:.3f}")
print("novel probe top words:", ", ".join(f"{w} {probs[run.tokenizer.stoi[w]]:.3f}" for w in top))
print(
    "\\nDropping `most` moves the mass onto other training words "
    f"({top[0]!r}), not onto `sure`. The sure/risky ratio among a tiny "
    "remainder is not the bias surviving. The framing pattern was a "
    "regularity of the designed sentences, not a property of the "
    "architecture, and not a person facing lives saved or lost."
)
''',
)

add(
    "04_quadrants_biases",
    "04_3_swap_the_labels",
    "Swap the labels",
    """Train a corpus that is still *deterministic*, but reversed: gain frames always continue with `risky`, loss frames with `sure`. Does the model invert the classic pattern? What would have to be true of a person for this to be a “debiasing” result?""",
    """from brainsmodels.biases import make_crawl, train_on_crawl
""",
    '''
classic = make_crawl(n_sentences=300, seed=SEED, gain_p_sure=1.0, loss_p_sure=0.0)
flipped = make_crawl(n_sentences=300, seed=SEED + 1, gain_p_sure=0.0, loss_p_sure=1.0)
a = train_on_crawl(classic, seed=SEED, n_steps=350)
b = train_on_crawl(flipped, seed=SEED + 1, n_steps=350)
print(f"{'corpus':<12}{'P(sure|gain)':>14}{'P(sure|loss)':>14}")
print(f"{'classic':<12}{a.gain_sure:>14.2f}{a.loss_sure:>14.2f}")
print(f"{'reversed':<12}{b.gain_sure:>14.2f}{b.loss_sure:>14.2f}")
print(
    "\\nThe architecture did not prefer the textbook answers. It preferred "
    "the completions it was given. Calling the reverse run 'debiasing' "
    "would be a category error: nothing in a person was changed. A "
    "preference-tuned LLM can be pushed the same way. That is still not "
    "a value, and not Tversky and Kahneman's subjects."
)
''',
)

add(
    "04b_four_views_of_AI",
    "04b_1_one_system_four_scores",
    "One system, four scores",
    """The map is four *questions*, not four machines. For each cell, print `lookup` on a word from that cell’s verdict (`errors`, `proof`, `imitation`, `goals`). Confirm you get one cell each time. Then write one sentence — not a new experiment — stating how a single clinical note-completer can be scored in all four cells at once.""",
    """from brainsmodels.four_views import format_quadrant, lookup
""",
    '''
for query in ("errors", "proof", "imitation", "goals"):
    hits = lookup(query)
    print(f"query {query!r}  →  {len(hits)} cell(s)")
    for cell in hits:
        print(" ", cell.title, "—", cell.verdict)
    print()
print(
    "A note-completer can copy a documented bias from discharge text "
    "(thinking humanly), emit a fluent but unproved plan (thinking "
    "rationally), pass a short Turing-style screen (acting humanly), and "
    "still have its objective added after pretraining (acting rationally). "
    "One architecture; four scores. Notebook 04 is the upper-left toy. "
    "The other three cells are verdicts, not extra nets."
)
''',
)

add(
    "04b_four_views_of_AI",
    "04b_2_the_empty_companion",
    "The empty companion",
    """Which cell has no companion notebook? Print it. Why did the paper refuse a fourth toy there, and what would a misleading toy have looked like (a longer prefix, a chain-of-thought string)?""",
    """from brainsmodels.four_views import QUADRANTS, format_quadrant
""",
    '''
for cell in QUADRANTS:
    flag = cell.companion or "(none)"
    print(f"{cell.title:<22}  companion {flag}")
print()
empty = [c for c in QUADRANTS if not c.companion]
print(format_quadrant(empty[0]))
print(
    "\\nThinking rationally has no extra toy because a scratchpad is still "
    "next-token prediction with a longer prefix. A notebook that printed "
    "a 'therefore' and called it a proof would be the misleading toy: it "
    "would look like System 2 and implement the same objective as "
    "notebook 01. The empty companion is the claim."
)
''',
)

add(
    "04b_four_views_of_AI",
    "04b_3_lookup_is_not_a_proof",
    "Lookup is not a proof",
    """`lookup` is a substring match on the catalogue text. Try `FOCUS = \"human\"` and `FOCUS = \"not a word in the catalogue\"`. How many cells fire, and what does that tell you about treating the map as an experiment?""",
    """from brainsmodels.four_views import format_quadrant, lookup
""",
    '''
for q in ("human", "not a word in the catalogue", "04", "02b"):
    hits = lookup(q)
    print(f"{q!r:40}  {len(hits)} hit(s):", ", ".join(c.title for c in hits) or "—")
print()
print(
    "The word 'human' sits in two cells by construction (the left column). "
    "A nonsense string sits in none. This notebook is an index, like "
    "notebook 00. It does not train, and a lookup hit is not evidence "
    "that a foundation model 'is' that cell. Notebook 04 is where a "
    "designed corpus is put on the table."
)
''',
)

add(
    "05_multistable_dynamics",
    "05_1_corrupt_until_it_fails",
    "Corrupt until it fails",
    """`corrupt(pattern, frac)` flips a fraction of bits, then `hopfield_async` descends. Sweep `frac` from 0 to 0.6 and record whether the final overlap with pattern 0 exceeds 0.8. Where does retrieval break, and why is that not a Necker cube?""",
    """from brainsmodels.dynamics import corrupt, hebb_weights, hopfield_async, make_patterns, overlaps
""",
    '''
patterns = make_patterns(n=32, seed=SEED)
W = hebb_weights(patterns)
fracs = np.linspace(0.0, 0.6, 13)
ok = []
for frac in fracs:
    cue = corrupt(patterns[0], frac=float(frac), seed=SEED + 3)
    recalled, _, _ = hopfield_async(cue, W, n_sweeps=8, seed=SEED)
    ov = float(overlaps(recalled, patterns)[0])
    ok.append(ov)
    print(f"frac {frac:.2f}  overlap with ξ0 {ov:+.2f}")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(fracs, ok, "o-")
ax.axhline(0.8, color="0.6", ls="--", lw=1)
ax.set_xlabel("flip fraction")
ax.set_ylabel("final overlap with ξ0")
ax.set_title("A Hopfield basin has a radius; past it, recall fails")
plt.tight_layout()
plt.show()
print(
    "Inside the basin the corrupted cue is still closer to ξ0 than to ξ1, "
    "and asynchronous sign updates finish the job. Past a radius the cue "
    "is a different start and may fall into the other well or a mixture. "
    "That is associative memory, not bistable perception: there is no "
    "cube, and no viewer."
)
''',
)

add(
    "05_multistable_dynamics",
    "05_2_beta_is_the_temperature",
    "β is the temperature",
    """`attention_retrieve(query, keys, beta)` is one modern-Hopfield step. Use the notebook’s mixed query ($0.55\\,\\xi_1+0.45\\,\\xi_2$) and print the softmax weights at $\\beta \\in \\{0.5, 1, 4, 16\\}$. When does the mass collapse onto one key, and why is that still not a transformer layer?""",
    """from brainsmodels.dynamics import attention_retrieve, make_patterns
""",
    '''
keys = make_patterns(n=32, seed=SEED)
query = 0.55 * keys[0] + 0.45 * keys[1]
query = query / float(np.linalg.norm(query))
print(f"{'beta':>6}  w(ξ0)   w(ξ1)")
ws = []
for beta in (0.5, 1.0, 4.0, 16.0):
    _, w = attention_retrieve(query, keys, beta)
    ws.append(w)
    print(f"{beta:6.1f}  {w[0]:.3f}   {w[1]:.3f}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot((0.5, 1, 4, 16), [w[0] for w in ws], "o-", label="ξ0")
ax.plot((0.5, 1, 4, 16), [w[1] for w in ws], "s-", label="ξ1")
ax.set_xlabel("β")
ax.set_ylabel("softmax weight")
ax.legend(frameon=False)
ax.set_title("Soft β shares; sharp β retrieves")
plt.tight_layout()
plt.show()
print(
    "Ramsauer et al.: one attention step with keys = values = stored "
    "patterns. Large β is a hard lookup, not a deeper network. Notebook "
    "01's maps are many such steps, trained, with a causal mask. This "
    "exercise is the *update*, not a language model and not a percept."
)
''',
)

add(
    "05_multistable_dynamics",
    "05_3_noise_switches_the_well",
    "Noise switches the well",
    """`necker_walk` is Langevin dynamics on $E(x)=\\frac{{1}}{{4}}(x^2-1)^2$. Run two walks from $x_0=0$ with `sigma=0.05` and `sigma=0.55` (same seed). Count sign changes of a 40-step moving average. When is the state trapped, and when does it switch? What is still not a cube?""",
    """from brainsmodels.dynamics import necker_walk
""",
    '''
def n_switches(xs, win=40):
    ma = np.convolve(xs, np.ones(win) / win, mode="valid")
    return int(np.sum(np.diff(np.sign(ma)) != 0))

fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.0), sharex=True)
for ax, sig in zip(axes, (0.05, 0.55)):
    xs = necker_walk(sigma=sig, x0=0.0, seed=SEED)
    ax.plot(xs, lw=0.8)
    ax.axhline(1, color="0.7", ls="--", lw=1)
    ax.axhline(-1, color="0.7", ls="--", lw=1)
    ax.set_ylabel("x")
    ax.set_title(f"σ = {sig}   switches ≈ {n_switches(xs)}", fontsize=9)
axes[-1].set_xlabel("step")
fig.suptitle("The same double well; only the noise scale changed")
plt.tight_layout()
plt.show()
print(
    "Small noise: the walk falls into one well and stays. Larger noise: "
    "it crosses the saddle. That is the teaching picture of bistable "
    "percepts as two attractors plus a perturbation. There is no cube, "
    "no rivalry experiment, and no claim that next-token sampling is "
    "seeing."
)
''',
)

add(
    "05b_multistable_animation",
    "05b_1_nearby_starts_two_wells",
    "Nearby starts, two wells",
    """The teaching surface $U(x,y)$ has several listed attractors. Descend from `(-2.50, -3.50)` and from `(-2.18, -3.50)` — 0.32 apart. Which well does each path enter? What does a ridge between nearby starts say about a prompt that sits near a saddle?""",
    """from brainsmodels.landscape import ATTRACTORS, descend, energy, nearest_attractor
""",
    '''
starts = ((-2.50, -3.50), (-2.18, -3.50), (0.80, 4.80))
print(f"{'start':<16}{'end':<16}{'U(end)':>8}  well")
for s in starts:
    path = descend(s)
    end = tuple(float(v) for v in path[-1])
    well = nearest_attractor(end, radius=0.6)
    print(f"{s!s:<16}{end!s:<16}{float(energy(*end)):>8.3f}  {well}")

print(
    "\\nA small shift in the start can change the basin. On this surface "
    "that is a ridge in U. In the paper's reading, a slight rewording "
    "of a prompt is a perturbation near a saddle — a teaching analogy, "
    "not a measurement of a frontier model, and not a Hopfield net "
    "(that is notebook 05)."
)
''',
)

add(
    "05b_multistable_animation",
    "05b_2_step_size",
    "Step size",
    """`descend` takes `lr`. From the default start, compare `lr=0.02` and `lr=0.25`: path length, final energy, and whether `nearest_attractor` still returns a listed well. When does a large step leave the teaching picture?""",
    """from brainsmodels.landscape import DEFAULT_START, descend, energy, nearest_attractor
""",
    '''
for lr in (0.02, 0.08, 0.25):
    path = descend(DEFAULT_START, lr=lr, n_steps=200)
    end = path[-1]
    well = nearest_attractor(tuple(end), radius=0.6)
    print(
        f"lr {lr:.2f}  steps {len(path):3d}  U={float(energy(*end)):+.3f}  "
        f"well {well}"
    )
print(
    "\\nA small step crawls; the default step is the figure; a large step "
    "can overshoot a well or leave the plotted square (the path is "
    "clipped). The surface is a teaching slice of activity space, not "
    "SGD on a transformer loss. Notebook 05 is the controlled "
    "demonstration; this one is the map."
)
''',
)

add(
    "05b_multistable_animation",
    "05b_3_energy_along_the_path",
    "Energy along the path",
    """Plot $U$ along the default descent. Is the sequence monotone? If a step rises, what in `descend` allowed it — and what would a true gradient flow have done?""",
    """from brainsmodels.landscape import DEFAULT_START, descend, energy
""",
    '''
path = descend(DEFAULT_START)
u = np.array([float(energy(x, y)) for x, y in path])
ups = int(np.sum(np.diff(u) > 1e-9))
print(f"steps {len(path)}  ΔU {u[-1] - u[0]:+.3f}  uphill steps {ups}")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(u)
ax.set_xlabel("descent step")
ax.set_ylabel("U(x, y)")
ax.set_title("Teaching descent on a plotted square")
plt.tight_layout()
plt.show()
print(
    "Discrete steps with a fixed lr are not an ODE. A rise, if it "
    "happens, is a too-large Euler step or the clip to the square. "
    "The animation is a schematic of falling into a basin, not a "
    "proof that training or perception is this surface."
)
''',
)

add(
    "05c_analog_brain_waves",
    "05c_1_swap_the_stencils",
    "Swap the stencils",
    """Hold the two object drives fixed and exchange the two context stencils. What happens to unit 2’s expressed activity? Why is that not a weight update?""",
    """from brainsmodels.analog_waves import (
    expressed,
    run_experiment,
    selectivity_table,
)
""",
    '''
result = run_experiment(seed=SEED)
before = selectivity_table(result)
# Exchange stencils; drives stay put.
result.stencil_first, result.stencil_second = (
    result.stencil_second,
    result.stencil_first,
)
after = selectivity_table(result)
print("unit 2 before / after swap")
for name in before:
    print(f"  {name:10s}  {before[name][1]:.3f} → {after[name][1]:.3f}")
print(
    "Unit 2’s high condition follows the open region. The drives are "
    "unchanged and no W was fitted. A swapped stencil is a changed "
    "control pattern, not a new synapse and not a measured LFP."
)
''',
)

add(
    "05c_analog_brain_waves",
    "05c_2_subtraction_is_not_addition",
    "Subtraction is not addition",
    """Panel B uses the mean of two control waves. Replace the mean with the absolute difference. Do the four pairs still give four distinct spatial gates? What computation have you removed?""",
    """from brainsmodels.analog_waves import combine_waves, control_waves, positions
""",
    '''
x = positions()
waves = control_waves(x)
pairs = (
    ("first", "match"),
    ("first", "recall"),
    ("second", "match"),
    ("second", "recall"),
)
added = np.stack([combine_waves(waves[a], waves[b]) for a, b in pairs])
subbed = np.stack([np.abs(waves[a] - waves[b]) for a, b in pairs])

def mean_offdiag(stack):
    diffs = [np.mean(np.abs(stack[i] - stack[j]))
             for i in range(4) for j in range(i + 1, 4)]
    return float(np.mean(diffs))

print(f"mean |gate_i − gate_j|  addition     {mean_offdiag(added):.3f}")
print(f"mean |gate_i − gate_j|  |difference| {mean_offdiag(subbed):.3f}")
print(
    "Addition is the analog combination rule the figure uses: two "
    "contexts sum to a third pattern. Absolute difference is another "
    "pointwise map; it is not the teaching claim. Neither map is a "
    "learned decoder, and neither is a cortex."
)
''',
)

add(
    "05c_analog_brain_waves",
    "05c_3_freeze_the_window",
    "Freeze the traveling window",
    """Recompute the space–time map with $\\mu(t)$ held at the sensory locus. What claim about relocation is gone, and what claim about gating remains?""",
    """from brainsmodels.analog_waves import gaussian, positions, traveling_open
""",
    '''
x = positions()
drive = gaussian(x, 0.50, 0.32)
times = np.linspace(0.0, 1.0, 24)
moving = np.stack([drive * traveling_open(x, t) for t in times])
frozen = np.stack([drive * traveling_open(x, 0.0) for _ in times])
peak_move = x[np.argmax(moving, axis=1)]
peak_frz = x[np.argmax(frozen, axis=1)]
print(f"moving  peak x: {peak_move[0]:.2f} → {peak_move[-1]:.2f}")
print(f"frozen  peak x: {peak_frz[0]:.2f} → {peak_frz[-1]:.2f}")
print(
    "A frozen window is a static stencil: drive is still gated, but "
    "activity no longer travels. Relocation was the traveling-wave "
    "claim. Both versions are designed functions, not LFPs, and not "
    "phenomenal experience."
)
''',
)

add(
    "06_global_workspace",
    "06_1_raise_the_threshold",
    "Raise the threshold",
    """Ignition is `activity >= THRESHOLD` (default 0.55). Hold a mid-strength pulse (`peak=0.35`, no runner) and re-run at thresholds 0.40, 0.55, and 0.70. Who occupies the workspace, and at which threshold does the same drive stay local?""",
    """import brainsmodels.workspace as ws
""",
    '''
old = ws.THRESHOLD
try:
    for thr in (0.40, 0.55, 0.70):
        ws.THRESHOLD = thr
        trial = ws.run_trial("mid", 0.35, 0.0)
        print(
            f"θ={thr:.2f}  ignited={trial.ignited}  "
            f"t={trial.ignition_t}  occupants={trial.occupants}"
        )
finally:
    ws.THRESHOLD = old
print(
    "\\nThe pulse is the same. Crossing a higher bar is a different event "
    "from being slightly more active. Below threshold the computation "
    "stays in the modules; above it, a slot opens. That is the "
    "schematic of ignition, not a measurement of consciousness, and "
    "not Claude."
)
''',
)

add(
    "06_global_workspace",
    "06_2_one_slot",
    "One slot",
    """Capacity is two. Set `CAPACITY = 1` and re-run the strong trial. Which module is dropped, and what does a one-slot workspace say about “everything that is active is reportable”?""",
    """import brainsmodels.workspace as ws
""",
    '''
old = ws.CAPACITY
try:
    two = ws.run_trial("cap2", ws.STRONG_PEAK, ws.RUNNER_PEAK)
    ws.CAPACITY = 1
    one = ws.run_trial("cap1", ws.STRONG_PEAK, ws.RUNNER_PEAK)
finally:
    ws.CAPACITY = old
print("capacity 2 occupants:", two.occupants)
print("capacity 1 occupants:", one.occupants)
print(
    "\\nThe runner-up can be active and still fail to be written back. "
    "Reportability in this toy is occupancy of a scarce buffer, not "
    "strength of a local trace. A verbalizable slice is not phenomenal "
    "experience — that is the §7 limit, and this exercise only makes "
    "the scarcity visible."
)
''',
)

add(
    "06_global_workspace",
    "06_3_lesion_the_winner",
    "Lesion the winner",
    """The intended winner is `count` (module 3). Run a trial whose pulse is only on the runner (`peak=0`, `runner=STRONG_PEAK`). Who ignites? What does a lesion of the would-be winner show about “the workspace chose the meaning”?""",
    """import brainsmodels.workspace as ws
""",
    '''
intact = ws.run_trial("intact", ws.STRONG_PEAK, ws.RUNNER_PEAK)
lesion = ws.run_trial("lesion count", 0.0, ws.STRONG_PEAK)
print("intact :", intact.occupants, "ignition_t", intact.ignition_t)
print("lesion :", lesion.occupants, "ignition_t", lesion.ignition_t)
print(
    "\\nWhoever crosses the bar is broadcast. Remove the original winner's "
    "drive and the runner occupies the slots. The workspace did not "
    "interpret the pulse; it admitted the largest local activities. "
    "That is access as a bottleneck, not a mind, and not a clinical "
    "neglect syndrome."
)
''',
)

add(
    "07_values_probing",
    "07_1_balance_erases_the_split",
    "Balance erases the split",
    """The notebook already trains biased vs balanced corpora. Confirm the numbers, then train a *third* corpus with `switch_p_act=0.0`, `foot_p_act=1.0` (the reverse of Greene’s usual summary). Does the model invert the split? What would it mean to call that inversion a change of values?""",
    """from brainsmodels.values import make_crawl, train_on_crawl
""",
    '''
specs = {
    "biased (usual)": dict(switch_p_act=1.0, foot_p_act=0.0),
    "balanced": dict(switch_p_act=0.5, foot_p_act=0.5),
    "reversed": dict(switch_p_act=0.0, foot_p_act=1.0),
}
print(f"{'corpus':<18}{'P(act|switch)':>16}{'P(act|foot)':>14}")
for name, kw in specs.items():
    run = train_on_crawl(make_crawl(n_sentences=280, seed=SEED, **kw), seed=SEED, n_steps=400)
    print(f"{name:<18}{run.switch_act:>16.2f}{run.foot_act:>14.2f}")
print(
    "\\nThe split tracks the designed frequencies, including the reverse. "
    "Nothing in the architecture preferred the textbook trolley answers. "
    "Calling the reverse run a change of values would name a corpus "
    "edit as a moral conversion. It is next-token mass. It is not "
    "Greene's subjects, and not a body that can be the one."
)
''',
)

add(
    "07_values_probing",
    "07_2_paraphrase_is_not_the_outcome",
    "Paraphrase is not the outcome",
    """`SAVE_PROBE` and `KILL_PROBE` describe one actuarial outcome in two wordings. Train a corpus with `save_p_act=1`, `kill_p_act=0` (and switch/foot balanced). Read $P(\\texttt{{act}})$ after each probe. Why can two names of the same outcome flip the answer?""",
    """from brainsmodels.values import KILL_PROBE, SAVE_PROBE, make_crawl, train_on_crawl
""",
    '''
crawl = make_crawl(
    n_sentences=280,
    seed=SEED,
    switch_p_act=0.5,
    foot_p_act=0.5,
    save_p_act=1.0,
    kill_p_act=0.0,
)
run = train_on_crawl(crawl, seed=SEED, n_steps=400)
print(f"P(act | 'to save five…')  {run.save_act:.2f}")
print(f"P(act | 'to kill one…')   {run.kill_act:.2f}")
print(
    "\\nThe two probes are paraphrases of one five-versus-one arithmetic. "
    "The corpus tied `save` to `act` and `kill` to `refrain`. The model "
    "follows the words it was given, not the outcome those words name. "
    "That is notebook 04's framing result on a trolley. It is not a "
    "value, and not dumbfounding in a person."
)
''',
)

add(
    "07_values_probing",
    "07_3_the_bars_are_not_a_sample",
    "The bars are not a sample",
    """Panel A’s 0.90 / 0.10 bars are a textbook summary, not Greene et al.’s measured rates. Change `HUMAN_SWITCH_ACT` and `HUMAN_FOOT_ACT` to 0.7 and 0.3, rebuild the figure’s human pair in memory (do not write the paper PDF), and print the new gap. What in the *model* panels should move if those bars were a real re-measurement — and what must not?""",
    """from brainsmodels import values as V
""",
    '''
print(f"library summary  switch {V.HUMAN_SWITCH_ACT:.2f}  foot {V.HUMAN_FOOT_ACT:.2f}")
V.HUMAN_SWITCH_ACT = 0.70
V.HUMAN_FOOT_ACT = 0.30
print(f"edited labels    switch {V.HUMAN_SWITCH_ACT:.2f}  foot {V.HUMAN_FOOT_ACT:.2f}")
print(f"gap (switch − foot) {V.HUMAN_SWITCH_ACT - V.HUMAN_FOOT_ACT:.2f}")
print(
    "\\nOnly panel A's reference bars would move. Panels B–D are trained "
    "on a designed corpus; they do not read HUMAN_*. If 0.90/0.10 were "
    "mistaken for Greene's sample, the error would be in the caption, "
    "not in the next-token mechanism. Restore the constants before "
    "`make figures` or the paper figure's panel A will be wrong."
)
V.HUMAN_SWITCH_ACT = 0.90
V.HUMAN_FOOT_ACT = 0.10
''',
)


def write_py(slug: str, imports: str, body: str) -> str:
    text = HEADER.format() + "\n" + imports + body.lstrip("\n")
    if not text.endswith("\n"):
        text += "\n"
    (SOL / f"{slug}.py").write_text(text)
    return text


def details_block(slug: str, title: str, code: str) -> str:
    return (
        f"<!-- solution-box: {slug}.py -->\n"
        f"<details>\n"
        f"<summary><strong>Show solution</strong> — {title}</summary>\n\n"
        f"Takes seconds on a laptop. Copy the block below into a new cell. "
        f"From a clone of the repository, `%load ../solutions/{slug}.py` "
        f"inserts the same code together with the few lines of setup it "
        f"needs to run outside the notebook.\n\n"
        f"```python\n"
        f"{code.rstrip()}\n"
        f"```\n\n"
        f"</details>\n"
        f"<!-- /solution-box -->\n"
    )


def exercises_markdown(items: list[tuple[str, str, str, str, str]]) -> str:
    parts = ["### Exercises\n"]
    for i, (slug, title, prompt, imports, body) in enumerate(items, start=1):
        code = write_py(slug, imports, body)
        parts.append(f"{i}. **{title}.** {prompt}\n")
        parts.append(details_block(slug, title, code))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def append_cell(nb_path: Path, source: str) -> None:
    nb = json.loads(nb_path.read_text())
    marker = "### Exercises\n"
    # Drop a previous generated exercises cell if re-run.
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and src.startswith(marker):
            continue
        cells.append(cell)
    cells.append(
        {
            "cell_type": "markdown",
            "id": "exercises001",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")[:-1]] + (
                [source.split("\n")[-1]] if source.endswith("\n") else []
            ),
        }
    )
    # Keep a trailing newline in the last source line like other cells.
    src_lines = source.splitlines(keepends=True)
    cells[-1]["source"] = src_lines
    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    mapping = {
        "01_tiny_transformer": "01_tiny_transformer.ipynb",
        "02_predictive_coding": "02_predictive_coding.ipynb",
        "02b_active_inference_vs_rl": "02b_active_inference_vs_rl.ipynb",
        "03_bio_vs_artificial": "03_bio_vs_artificial.ipynb",
        "04_quadrants_biases": "04_quadrants_biases.ipynb",
        "04b_four_views_of_AI": "04b_four_views_of_AI.ipynb",
        "05_multistable_dynamics": "05_multistable_dynamics.ipynb",
        "05b_multistable_animation": "05b_multistable_animation.ipynb",
        "05c_analog_brain_waves": "05c_analog_brain_waves.ipynb",
        "06_global_workspace": "06_global_workspace.ipynb",
        "07_values_probing": "07_values_probing.ipynb",
    }
    for key, filename in mapping.items():
        md = exercises_markdown(EXERCISES[key])
        append_cell(NB / filename, md)
        print("wrote", key, len(EXERCISES[key]), "exercises")


if __name__ == "__main__":
    main()
