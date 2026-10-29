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
    '\nmodel, tokenizer, text, _ = train_demo(seed=SEED, n_steps=200)\nprompt = "the cat sat on the "\nids = tokenizer.encode(prompt)\nlogits, causal = model.forward(ids, record_attn=True)\nq, k = model._cache["q"], model._cache["k"]\nscale = np.sqrt(model._d_head)\nscores = np.einsum("bhtd,bhsd->bhts", q, k) / scale\nleaky = _softmax(scores, axis=-1)\n\nt = len(prompt)\n# Query at the space after "cat" (index of that space).\nq_at = prompt.index(" ", 4)\nfuture = slice(q_at + 1, t)\nleaked = float(leaky[0, :, q_at, future].sum() / model.n_heads)\nblocked = float(causal[0, :, q_at, future].sum() / model.n_heads)\nprint(f"query at index {q_at} ({prompt[q_at]!r})")\nprint(f"mean mass on later characters, causal mask on : {blocked:.3f}")\nprint(f"mean mass on later characters, mask dropped   : {leaked:.3f}")\n\nfig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))\nlabels = [("␣" if c == " " else c) for c in prompt]\nfor ax, mat, title in (\n    (axes[0], causal[0, 0, :t, :t], "head 1, causal"),\n    (axes[1], leaky[0, 0, :t, :t], "head 1, no mask"),\n):\n    ax.imshow(mat, cmap="magma", vmin=0, vmax=max(0.2, float(mat.max())))\n    ax.set_xticks(range(t), labels, rotation=90, fontsize=7)\n    ax.set_yticks(range(t), labels, fontsize=7)\n    ax.set_title(title, fontsize=9)\nfig.suptitle("Without the mask, a query can look at characters that do not yet exist")\nplt.tight_layout()\nplt.show()\nprint(\'Removing the mask exposes future characters during training. Left-to-right generation has no such observations, so this changes the prediction task and leaks the target.\')\n',
)

add(
    "01_tiny_transformer",
    "01_2_a_unigram_is_not_enough",
    'Compare context baselines',
    'Fit unigram, bigram and Transformer models on the same training partition. Compare next-character negative log likelihood on held-out windows. How much does each additional source of context help on this corpus?',
    """from brainsmodels.tiny_transformer import CharTokenizer, make_corpus, train_demo
""",
    '\ntext = make_corpus(seed=SEED)\ntokenizer = CharTokenizer.from_text(text)\nids = tokenizer.encode(text)\nn = len(ids)\ncut = int(0.85 * n)\ntrain_ids, test_ids = ids[:cut], ids[cut:]\nfrom brainsmodels.tiny_transformer import TinyTransformer\nmodel = TinyTransformer(tokenizer.vocab_size, seed=SEED)\nmodel.train(train_ids, n_steps=200, seed=SEED)\n\nunigram = np.ones(tokenizer.vocab_size)\nfor i in train_ids:\n    unigram[int(i)] += 1.0\nunigram = unigram / unigram.sum()\n\nbigram = np.ones((tokenizer.vocab_size, tokenizer.vocab_size))\nfor a, b in zip(train_ids[:-1], train_ids[1:]):\n    bigram[int(a), int(b)] += 1.0\nbigram = bigram / bigram.sum(axis=1, keepdims=True)\n\ndef nll_unigram(seq):\n    return float(-np.mean(np.log(unigram[seq])))\n\ndef nll_bigram(seq):\n    return float(-np.mean(np.log(bigram[seq[:-1], seq[1:]])))\n\ndef nll_model(seq, block=32):\n    losses = []\n    for start in range(0, len(seq) - block - 1, block):\n        ctx = seq[start : start + block]\n        tgt = seq[start + 1 : start + 1 + block]\n        logits, _ = model.forward(ctx)\n        p = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))\n        p = p / p.sum(axis=-1, keepdims=True)\n        losses.append(-np.log(np.clip(p[np.arange(block), tgt], 1e-9, 1.0)).mean())\n    return float(np.mean(losses))\n\nprint(f"chance (ln V)   {np.log(tokenizer.vocab_size):.3f}")\nprint(f"unigram  test   {nll_unigram(test_ids):.3f}")\nprint(f"bigram   test   {nll_bigram(test_ids):.3f}")\nprint(f"1-layer  test   {nll_model(test_ids):.3f}")\nprint(\'Compare the measured held-out losses. Unigram, bigram and Transformer models use different amounts of context; these results concern prediction on this designed corpus.\')\n',
)

add(
    "01_tiny_transformer",
    "01_3_temperature_is_not_a_belief",
    'Temperature and sampling',
    'Sample 20 continuations of `the cat sat on the ` at temperatures 0, 0.4 and 1.2. Compare repetition, word variety and invalid spellings. What does this measure, and what evidence would be needed to assess factual reliability?',
    """from brainsmodels.tiny_transformer import train_demo
""",
    '\nmodel, tokenizer, _, _ = train_demo(seed=SEED, n_steps=400)\nprompt = "the cat sat on the "\ntemps = (0.0, 0.4, 1.2)\nobjects = ("mat", "hat", "log", "box")\n\nfor T in temps:\n    texts = [\n        model.generate(prompt, tokenizer, n_tokens=12, temperature=T, seed=SEED + k)\n        for k in range(20)\n    ]\n    tails = [t[len(prompt) :].split()[0] if t[len(prompt) :].split() else "" for t in texts]\n    hits = {w: sum(w in tail for tail in tails) for w in objects}\n    uniq = len(set(tails))\n    print(f"T={T:<4}  unique first-words {uniq:2d}  object hits {hits}")\n    print("   e.g.", texts[0][len(prompt) : 28])\n\nprint(\'Temperature rescales the next-character distribution. Compare repetition and variation in the samples; sampling sharpness alone does not measure factual accuracy or understanding.\')\n',
)

add(
    "02_predictive_coding",
    "02_1_inference_not_learning",
    "Inference is not learning",
    """Panel C is the residual falling *inside one patch*, with the weights held fixed. Freeze a trained network, then reconstruct a held-out patch after 1, 5, and 40 inference steps. Then take the *untrained* initial weights and do the same. How much of the falling residual is the hierarchy explaining the patch, and how much is just “any basis plus enough iterations”?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '\ndata = PatchDataset.from_pink_noise(n_train=500, n_test=16, n_images=8, seed=SEED)\nx = data.test[0]\nfresh = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)\ntrained = HierarchicalPredictiveCoding(n_input=x.size, seed=SEED)\ntrained.train(data.train, n_epochs=6, n_infer=20, seed=SEED)\n\ndef curve(model, steps=40):\n    _, mse = model.infer_history(x, n_steps=steps)\n    return np.asarray(mse)\n\nc0, c1 = curve(fresh), curve(trained)\nprint(f"untrained  MSE t=1 {c0[0]:.3f}  t=40 {c0[-1]:.3f}")\nprint(f"trained    MSE t=1 {c1[0]:.3f}  t=40 {c1[-1]:.3f}")\nprint(f"drop untrained {c0[0] - c0[-1]:.3f}   drop trained {c1[0] - c1[-1]:.3f}")\n\nfig, ax = plt.subplots(figsize=(5.2, 3.2))\nax.plot(c0, label="untrained weights")\nax.plot(c1, label="after 6 epochs")\nax.set_xlabel("inference step")\nax.set_ylabel("prediction-error MSE")\nax.legend(frameon=False)\nax.set_title("Same infer loop; only the descending weights changed")\nplt.tight_layout()\nplt.show()\nprint(\'Inference changes the current representation with weights fixed. Compare the measured residuals for trained and initial weights to separate iterative fitting from the effect of learning.\')\n',
)

add(
    "02_predictive_coding",
    "02_2_one_level",
    "One level",
    """The network is two-level: $r_2$ predicts $r_1$, $r_1$ predicts pixels. Train a copy with `n_level2=1` (almost no higher code) and one with the default 12 units, same data and seed. Compare held-out reconstruction MSE after the same number of inference steps. What does the higher level buy on these 8×8 patches, and what can it not buy?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '\ndata = PatchDataset.from_pink_noise(n_train=500, n_test=24, n_images=8, seed=SEED)\nshallow = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=1, seed=SEED)\ndeep = HierarchicalPredictiveCoding(n_input=data.train.shape[1], n_level2=12, seed=SEED)\nshallow.train(data.train, n_epochs=6, n_infer=20, seed=SEED)\ndeep.train(data.train, n_epochs=6, n_infer=20, seed=SEED + 1)\n\ndef recon_mse(model):\n    _, _, tr = model.infer(data.test, n_steps=40, record=True)\n    return float(np.mean(tr.error**2))\n\nprint(f"n_level2=1   held-out MSE {recon_mse(shallow):.4f}")\nprint(f"n_level2=12  held-out MSE {recon_mse(deep):.4f}")\nprint(\'Compare held-out reconstruction errors at the same inference budget. This tests the added latent capacity on these patches; it does not isolate the organization of cortical processing.\')\n',
)

add(
    "02_predictive_coding",
    "02_3_skip_the_whitening",
    "Skip the whitening",
    """The descending weights are learned in ZCA-whitened coordinates, then mapped back to pixels for panel A. Train the same architecture on the *unwhitened* pixel patches (`data.train_pixel`). Do the first-level columns still look localized and oriented when you plot them as 8×8 images? What did whitening buy, and what did it not invent?""",
    """from brainsmodels.predictive_coding import HierarchicalPredictiveCoding, PatchDataset
""",
    '\ndata = PatchDataset.from_pink_noise(n_train=500, n_test=16, n_images=8, seed=SEED)\npix = HierarchicalPredictiveCoding(n_input=data.train_pixel.shape[1], seed=SEED)\nzca = HierarchicalPredictiveCoding(n_input=data.train.shape[1], seed=SEED)\npix.train(data.train_pixel, n_epochs=6, n_infer=20, seed=SEED)\nzca.train(data.train, n_epochs=6, n_infer=20, seed=SEED)\n\nrf_pix = pix.U1\nrf_zca = data.whitening.to_pixel_basis(zca.U1)\nenergy = rf_zca.reshape(data.patch_size, data.patch_size, -1).var(axis=(0, 1))\norder = np.argsort(energy)[::-1][:8]\n\nfig, axes = plt.subplots(2, 8, figsize=(9.5, 2.8))\nfor col, i in enumerate(order):\n    for row, bank, name in ((0, rf_pix, "pixels"), (1, rf_zca, "ZCA→pixels")):\n        patch = bank[:, i].reshape(data.patch_size, data.patch_size)\n        vmax = np.max(np.abs(patch)) + 1e-8\n        axes[row, col].imshow(patch, cmap="gray", vmin=-vmax, vmax=vmax)\n        axes[row, col].set_xticks([])\n        axes[row, col].set_yticks([])\n        if col == 0:\n            axes[row, col].set_ylabel(name, fontsize=8)\nfig.suptitle("Whitening changes the coordinates the Hebbian rule sees")\nplt.tight_layout()\nplt.show()\nprint(\'Whitening changes the input covariance and hence the learning problem. Compare the learned columns and reconstruction errors before attributing their appearance to the update rule.\')\n',
)

add(
    '02b_active_inference_vs_rl',
    '02b_1_drop_the_epistemic_term',
    'Information gain and planning horizon',
    'Compare the known-model planners with and without information gain at horizons 1 and 3. Can future reward alone justify a cue visit?',
    'from brainsmodels.tmaze import ActiveInferenceAgent, TMaze, run_trial_aif\n',
    'for horizon in (1, 3):\n    for weight in (0., 1.):\n        agent = ActiveInferenceAgent(horizon=horizon, epistemic_weight=weight)\n        trace = run_trial_aif(TMaze(seed=SEED), agent, context=1)\n        print(f"horizon={horizon}, IG weight={weight}: {trace.locations}, return={sum(trace.rewards):+.1f}")\nprint("The three-move reward-only planner can value information instrumentally. Both planners know the model.")\n',
)

add(
    "02b_active_inference_vs_rl",
    "02b_2_a_noisy_cue",
    "A noisy cue",
    """The environment and the agent both default to `cue_accuracy=1`. Set both to $0.6$ and compare cue-first rate and mean reward over 40 trials against the perfect-cue agent. When does “go to the cue first” stop being worth the step?""",
    """from brainsmodels.tmaze import ActiveInferenceAgent, TMaze, run_trial_aif
""",
    '\ndef score(acc, n=40):\n    env = TMaze(cue_accuracy=acc, seed=SEED)\n    agent = ActiveInferenceAgent(cue_accuracy=acc)\n    cues, rews = [], []\n    for _ in range(n):\n        tr = run_trial_aif(env, agent, n_steps=3)\n        cues.append(tr.visited_cue_first)\n        rews.append(sum(tr.rewards))\n    return float(np.mean(cues)), float(np.mean(rews))\n\nfor acc in (1.0, 0.6, 0.5):\n    cue, rew = score(acc)\n    print(f"cue accuracy {acc:.1f}  cue-first {cue:.2f}  mean return {rew:+.2f}")\nprint(\'At accuracy 0.5 the cue carries no information about context. Report the selected route and return at each accuracy. Test value also depends on costs, available actions and the remaining horizon.\')\n',
)

add(
    '02b_active_inference_vs_rl',
    '02b_3_ql_needs_the_stumble',
    'Prior knowledge and learned action values',
    'Compare the three agents on matched contexts for 400 trials. Then inspect the Q-learner’s start-state action values. Which prior-knowledge difference prevents attributing an early gap to the objective alone?',
    'from brainsmodels.tmaze import CENTER, QLearningAgent, TMaze, run_trial_ql, run_experiment\n',
    'traces = run_experiment(n_trials=400, seed=SEED)\nfor name, trials in traces.items():\n    print(name, "cue-first", np.mean([t.visited_cue_first for t in trials]), "return", np.mean([sum(t.rewards) for t in trials]))\nagent = QLearningAgent(seed=SEED)\nenv = TMaze(seed=SEED)\nfor context in np.random.default_rng(SEED).integers(0, 2, 400):\n    run_trial_ql(env, agent, context=int(context))\nprint("Learned Q(center, unknown):", agent.Q[agent.max_steps, CENTER, 0])\nprint("The learner begins without the transition and observation model supplied to both planners.")\n',
)

add(
    '03_bio_vs_artificial',
    '03_1_xor_needs_a_hidden_layer',
    'Architecture and update-rule confounds',
    'Report XOR training accuracy for the spiking system, the default MLP and an MLP with one hidden unit. Which factors change together, and why does this fail to isolate update locality?',
    'from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_xor\n',
    '\nx, y = make_xor(n=200, seed=SEED)\nspk = SpikingPerceptron(seed=SEED)\nmlp = SmallMLP(n_hidden=16, seed=SEED)\nthin = SmallMLP(n_hidden=1, seed=SEED)\nspk.train(x, y, n_epochs=30, seed=SEED)\nmlp.train(x, y, n_epochs=30)\nthin.train(x, y, n_epochs=30)\nprint(f"LIF perceptron (2→2, local delta)     acc {np.mean(spk.predict(x, seed=SEED) == y):.2f}")\nprint(f"MLP 2→16→2, SGD                       acc {np.mean(mlp.predict(x) == y):.2f}")\nprint(f"MLP 2→1→2, SGD                        acc {np.mean(thin.predict(x) == y):.2f}")\nprint(\'The systems differ in encoding, depth and update rule. Report their measured accuracies; this comparison does not isolate the effect of update locality.\')\n',
)

add(
    "03_bio_vs_artificial",
    "03_2_same_points_different_update",
    "Same points, different update",
    """On the linearly separable Gaussians both learners succeed. After training, scatter the two weight columns of the LIF net and the first-layer weights of the MLP (plot the 2-D input weights). The decision is similar; the objects that were updated are not. What was incremented in each case?""",
    """from brainsmodels.bio_vs_artificial import SmallMLP, SpikingPerceptron, make_linear
""",
    '\nx, y = make_linear(n=200, seed=SEED)\nspk = SpikingPerceptron(seed=SEED)\nmlp = SmallMLP(n_hidden=16, seed=SEED)\nspk.train(x, y, n_epochs=25, seed=SEED)\nmlp.train(x, y, n_epochs=25)\n\nfig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))\nfor ax, pts, c, lab in (\n    (axes[0], x[y == 0], "#4a7c8c", "class 0"),\n    (axes[0], x[y == 1], "#c47a3a", "class 1"),\n):\n    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=c, label=lab, alpha=0.7)\nfor k, col in enumerate(("#2c5f6e", "#9a4a2a")):\n    w = spk.w[:, k]\n    axes[0].arrow(0, 0, w[0], w[1], color=col, width=0.02, head_width=0.12)\n    axes[0].set_title("LIF: two columns of W (2×2)", fontsize=9)\naxes[0].legend(fontsize=8, frameon=False)\naxes[0].set_aspect("equal")\n\nim = axes[1].imshow(mlp.w1.T, cmap="coolwarm", aspect="auto")\naxes[1].set_title("MLP: W1 (2 → 16), updated by SGD", fontsize=9)\naxes[1].set_xlabel("input dim")\naxes[1].set_ylabel("hidden unit")\nfig.colorbar(im, ax=axes[1], fraction=0.046)\nplt.tight_layout()\nplt.show()\nprint(\'The local count-based update and backpropagated gradient update different quantities. Inspect both rules; successful classification alone cannot identify which rule a biological circuit uses.\')\n',
)

add(
    "03_bio_vs_artificial",
    "03_3_leak_is_not_depth",
    "Leak is not depth",
    'Retrain the same spiking architecture on the linear task at leak values 0.05, 0.20 and 0.80 with the same seed. Compare measured training accuracies. Which state dynamics change, and which architectural and learning choices stay fixed?',
    """from brainsmodels.bio_vs_artificial import SpikingPerceptron, make_linear
""",
    '\nx, y = make_linear(n=200, seed=SEED)\nfor leak in (0.05, 0.20, 0.80):\n    net = SpikingPerceptron(leak=leak, seed=SEED)\n    net.train(x, y, n_epochs=25, seed=SEED)\n    acc = float(np.mean(net.predict(x, seed=SEED) == y))\n    print(f"leak {leak:.2f}  train acc {acc:.2f}")\nprint(\'Leak changes membrane-state decay within a trial. Report its measured effect here; changing leak leaves the architecture and update rule fixed.\')\n',
)

add(
    "04_quadrants_biases",
    "04_1_interpolate_the_corpus",
    "Interpolate the corpus",
    """The notebook trains a fully biased corpus and a balanced one. Train four word-level models with `gain_p_sure` in $\\{1.0, 0.75, 0.5, 0.25\\}$ (keep `loss_p_sure = 1 - gain_p_sure`). Plot $P(\\texttt{{sure}}\\mid\\text{{gain probe}})$ against the corpus rate. How close is the next-token answer to a copy of the training frequency?""",
    """from brainsmodels.biases import make_crawl, train_on_crawl
""",
    '\nrates = (1.0, 0.75, 0.5, 0.25)\nmodel_p = []\nfor p in rates:\n    crawl = make_crawl(n_sentences=240, seed=SEED, gain_p_sure=p, loss_p_sure=1.0 - p)\n    run = train_on_crawl(crawl, seed=SEED, n_steps=350)\n    model_p.append(run.gain_sure)\n    print(f"corpus P(sure|gain) {p:.2f}  model {run.gain_sure:.2f}")\n\nfig, ax = plt.subplots(figsize=(4.4, 3.4))\nax.plot([0, 1], [0, 1], color="0.75", lw=1)\nax.plot(rates, model_p, "o-", color="#2c5f6e")\nax.set_xlabel("corpus P(sure | gain frame)")\nax.set_ylabel("model P(sure | gain probe)")\nax.set_title("Next-token mass tracks the designed frequency")\nax.set_xlim(-0.05, 1.05)\nax.set_ylim(-0.05, 1.05)\nplt.tight_layout()\nplt.show()\nprint(\'These probes reuse training-template prefixes. Compare completion probability with the designed corpus frequency and report deviations; this is a fitted association, not a human behavioral measurement.\')\n',
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
    'Reverse the training associations',
    'Reverse the deterministic gain/loss associations in the corpus. Compare measured completion probabilities. Which additional evidence would a human debiasing claim require?',
    """from brainsmodels.biases import make_crawl, train_on_crawl
""",
    '\nclassic = make_crawl(n_sentences=300, seed=SEED, gain_p_sure=1.0, loss_p_sure=0.0)\nflipped = make_crawl(n_sentences=300, seed=SEED + 1, gain_p_sure=0.0, loss_p_sure=1.0)\na = train_on_crawl(classic, seed=SEED, n_steps=350)\nb = train_on_crawl(flipped, seed=SEED + 1, n_steps=350)\nprint(f"{\'corpus\':<12}{\'P(sure|gain)\':>14}{\'P(sure|loss)\':>14}")\nprint(f"{\'classic\':<12}{a.gain_sure:>14.2f}{a.loss_sure:>14.2f}")\nprint(f"{\'reversed\':<12}{b.gain_sure:>14.2f}{b.loss_sure:>14.2f}")\nprint(\'Compare the measured completions after reversing the corpus associations. This intervention tests sensitivity to training text; it does not measure human debiasing or an agent changing its values.\')\n',
)

add(
    "04b_four_views_of_AI",
    "04b_1_one_system_four_scores",
    'One system, four evaluation questions',
    'Inspect the four catalog entries using `lookup`. Matches depend on the actual text and need not be unique. For one clinical note-completer, propose an evaluation under each criterion and identify the required evidence.',
    """from brainsmodels.four_views import format_quadrant, lookup
""",
    '\nfor query in ("mechanisms", "validity", "imitation", "goals"):\n    hits = lookup(query)\n    print(f"query {query!r}  →  {len(hits)} cell(s)")\n    for cell in hits:\n        print(" ", cell.title, "—", cell.verdict)\n    print()\nprint(\'The catalog lists four evaluation criteria. For a clinical note-completer, specify a behavioral or mechanistic comparison, a validity check, an imitation test and an explicit decision objective. No scores are measured here.\')\n',
)

add(
    "04b_four_views_of_AI",
    "04b_2_the_empty_companion",
    'Design the missing exercise',
    'Identify the criterion with no companion notebook. Propose an exercise that checks argument validity against an independent reference. How would you distinguish a plausible explanation from a verified result?',
    """from brainsmodels.four_views import QUADRANTS, format_quadrant
""",
    '\nfor cell in QUADRANTS:\n    flag = cell.companion or "(none)"\n    print(f"{cell.title:<22}  companion {flag}")\nprint()\nempty = [c for c in QUADRANTS if not c.companion]\nprint(format_quadrant(empty[0]))\nprint(\'The missing companion marks a gap in this collection. A useful addition could compare generated arguments with a proof checker or independently verified answers. The training objective alone does not determine whether reasoning succeeds.\')\n',
)

add(
    "04b_four_views_of_AI",
    "04b_3_lookup_is_not_a_proof",
    'Inspect catalog matching',
    'Try `human` and a string absent from the catalog. Count the actual substring matches and explain why these counts do not measure model capability.',
    """from brainsmodels.four_views import format_quadrant, lookup
""",
    '\nfor q in ("human", "not a word in the catalogue", "04", "02b"):\n    hits = lookup(q)\n    print(f"{q!r:40}  {len(hits)} hit(s):", ", ".join(c.title for c in hits) or "—")\nprint()\nprint(\'Substring matches describe this catalog. They do not evaluate a model. Count the actual matches and identify the text responsible for each one.\')\n',
)

add(
    "05_multistable_dynamics",
    "05_1_corrupt_until_it_fails",
    "Corrupt until it fails",
    """`corrupt(pattern, frac)` flips a fraction of bits, then `hopfield_async` descends. Sweep `frac` from 0 to 0.6 and record whether the final overlap with pattern 0 exceeds 0.8. Where does retrieval break, and why is that not a Necker cube?""",
    """from brainsmodels.dynamics import corrupt, hebb_weights, hopfield_async, make_patterns, overlaps
""",
    '\npatterns = make_patterns(n=32, seed=SEED)\nW = hebb_weights(patterns)\nfracs = np.linspace(0.0, 0.6, 13)\nok = []\nfor frac in fracs:\n    cue = corrupt(patterns[0], frac=float(frac), seed=SEED + 3)\n    recalled, _, _ = hopfield_async(cue, W, n_sweeps=8, seed=SEED)\n    ov = float(overlaps(recalled, patterns)[0])\n    ok.append(ov)\n    print(f"frac {frac:.2f}  overlap with ξ0 {ov:+.2f}")\n\nfig, ax = plt.subplots(figsize=(5.0, 3.0))\nax.plot(fracs, ok, "o-")\nax.axhline(0.8, color="0.6", ls="--", lw=1)\nax.set_xlabel("flip fraction")\nax.set_ylabel("final overlap with ξ0")\nax.set_title("A Hopfield basin has a radius; past it, recall fails")\nplt.tight_layout()\nplt.show()\nprint(\'Report retrieval success across corruption levels and repeat with several seeds. The result concerns this stored-pattern system; a perceptual interpretation would need separate observations.\')\n',
)

add(
    "05_multistable_dynamics",
    "05_2_beta_is_the_temperature",
    "β is the temperature",
    """`attention_retrieve(query, keys, beta)` is one modern-Hopfield step. Use the notebook’s mixed query ($0.55\\,\\xi_1+0.45\\,\\xi_2$) and print the softmax weights at $\\beta \\in \\{0.5, 1, 4, 16\\}$. When does the mass collapse onto one key, and why is that still not a transformer layer?""",
    """from brainsmodels.dynamics import attention_retrieve, make_patterns
""",
    '\nkeys = make_patterns(n=32, seed=SEED)\nquery = 0.55 * keys[0] + 0.45 * keys[1]\nquery = query / float(np.linalg.norm(query))\nprint(f"{\'beta\':>6}  w(ξ0)   w(ξ1)")\nws = []\nfor beta in (0.5, 1.0, 4.0, 16.0):\n    _, w = attention_retrieve(query, keys, beta)\n    ws.append(w)\n    print(f"{beta:6.1f}  {w[0]:.3f}   {w[1]:.3f}")\n\nfig, ax = plt.subplots(figsize=(4.6, 3.0))\nax.plot((0.5, 1, 4, 16), [w[0] for w in ws], "o-", label="ξ0")\nax.plot((0.5, 1, 4, 16), [w[1] for w in ws], "s-", label="ξ1")\nax.set_xlabel("β")\nax.set_ylabel("softmax weight")\nax.legend(frameon=False)\nax.set_title("Soft β shares; sharp β retrieves")\nplt.tight_layout()\nplt.show()\nprint(\'Increasing beta sharpens the weights on the stored patterns. This is one attention retrieval operation; a full Transformer also includes learned projections, other sublayers and multiple positions.\')\n',
)

add(
    "05_multistable_dynamics",
    "05_3_noise_switches_the_well",
    "Noise switches the well",
    """`necker_walk` is Langevin dynamics on $E(x)=\\frac{{1}}{{4}}(x^2-1)^2$. Run two walks from $x_0=0$ with `sigma=0.05` and `sigma=0.55` (same seed). Count sign changes of a 40-step moving average. When is the state trapped, and when does it switch? What is still not a cube?""",
    """from brainsmodels.dynamics import necker_walk
""",
    '\ndef n_switches(xs, win=40):\n    ma = np.convolve(xs, np.ones(win) / win, mode="valid")\n    return int(np.sum(np.diff(np.sign(ma)) != 0))\n\nfig, axes = plt.subplots(2, 1, figsize=(6.0, 4.0), sharex=True)\nfor ax, sig in zip(axes, (0.05, 0.55)):\n    xs = necker_walk(sigma=sig, x0=0.0, seed=SEED)\n    ax.plot(xs, lw=0.8)\n    ax.axhline(1, color="0.7", ls="--", lw=1)\n    ax.axhline(-1, color="0.7", ls="--", lw=1)\n    ax.set_ylabel("x")\n    ax.set_title(f"σ = {sig}   switches ≈ {n_switches(xs)}", fontsize=9)\naxes[-1].set_xlabel("step")\nfig.suptitle("The same double well; only the noise scale changed")\nplt.tight_layout()\nplt.show()\nprint(\'Compare the measured switching counts at the two noise levels. This illustrates stochastic transitions in a designed potential; it is not a fit to a perceptual experiment.\')\n',
)

add(
    "05b_multistable_animation",
    "05b_1_nearby_starts_two_wells",
    'Nearby starts and basins',
    'Descend from (-2.50, -3.50) and (-2.18, -3.50). Which basin does each path enter? What measurements would be needed to test an analogous claim about a language model?',
    """from brainsmodels.landscape import ATTRACTORS, descend, energy, nearest_attractor
""",
    '\nstarts = ((-2.50, -3.50), (-2.18, -3.50), (0.80, 4.80))\nprint(f"{\'start\':<16}{\'end\':<16}{\'U(end)\':>8}  well")\nfor s in starts:\n    path = descend(s)\n    end = tuple(float(v) for v in path[-1])\n    well = nearest_attractor(end, radius=0.6)\n    print(f"{s!s:<16}{end!s:<16}{float(energy(*end)):>8.3f}  {well}")\n\nprint(\'Nearby starts can enter different basins on this chosen surface. Any analogy with prompt sensitivity needs independent model measurements.\')\n',
)

add(
    "05b_multistable_animation",
    "05b_2_step_size",
    "Step size",
    """`descend` takes `lr`. From the default start, compare `lr=0.02` and `lr=0.25`: path length, final energy, and whether `nearest_attractor` still returns a listed well. When does a large step leave the teaching picture?""",
    """from brainsmodels.landscape import DEFAULT_START, descend, energy, nearest_attractor
""",
    '\nfor lr in (0.02, 0.08, 0.25):\n    path = descend(DEFAULT_START, lr=lr, n_steps=200)\n    end = path[-1]\n    well = nearest_attractor(tuple(end), radius=0.6)\n    print(\n        f"lr {lr:.2f}  steps {len(path):3d}  U={float(energy(*end)):+.3f}  "\n        f"well {well}"\n    )\nprint(\'Compare final energy and boundary clipping at each step size. These are numerical properties of the chosen two-dimensional potential.\')\n',
)

add(
    "05b_multistable_animation",
    "05b_3_energy_along_the_path",
    "Energy along the path",
    """Plot $U$ along the default descent. Is the sequence monotone? If a step rises, what in `descend` allowed it — and what would a true gradient flow have done?""",
    """from brainsmodels.landscape import DEFAULT_START, descend, energy
""",
    '\npath = descend(DEFAULT_START)\nu = np.array([float(energy(x, y)) for x, y in path])\nups = int(np.sum(np.diff(u) > 1e-9))\nprint(f"steps {len(path)}  ΔU {u[-1] - u[0]:+.3f}  uphill steps {ups}")\n\nfig, ax = plt.subplots(figsize=(5.2, 3.0))\nax.plot(u)\nax.set_xlabel("descent step")\nax.set_ylabel("U(x, y)")\nax.set_title("Teaching descent on a plotted square")\nplt.tight_layout()\nplt.show()\nprint(\'A finite Euler step can increase energy when it overshoots. Check the step size and clipping before interpreting a nonmonotone path.\')\n',
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
    '\nresult = run_experiment(seed=SEED)\nbefore = selectivity_table(result)\n# Exchange stencils; drives stay put.\nresult.stencil_first, result.stencil_second = (\n    result.stencil_second,\n    result.stencil_first,\n)\nafter = selectivity_table(result)\nprint("unit 2 before / after swap")\nfor name in before:\n    print(f"  {name:10s}  {before[name][1]:.3f} → {after[name][1]:.3f}")\nprint(\'The fixed control pattern changes expressed activity while the object drives and weights remain fixed. This is a designed gating calculation.\')\n',
)

add(
    "05c_analog_brain_waves",
    "05c_2_subtraction_is_not_addition",
    "Subtraction is not addition",
    """Panel B uses the mean of two control waves. Replace the mean with the absolute difference. Do the four pairs still give four distinct spatial gates? What computation have you removed?""",
    """from brainsmodels.analog_waves import combine_waves, control_waves, positions
""",
    '\nx = positions()\nwaves = control_waves(x)\npairs = (\n    ("first", "match"),\n    ("first", "recall"),\n    ("second", "match"),\n    ("second", "recall"),\n)\nadded = np.stack([combine_waves(waves[a], waves[b]) for a, b in pairs])\nsubbed = np.stack([np.abs(waves[a] - waves[b]) for a, b in pairs])\n\ndef mean_offdiag(stack):\n    diffs = [np.mean(np.abs(stack[i] - stack[j]))\n             for i in range(4) for j in range(i + 1, 4)]\n    return float(np.mean(diffs))\n\nprint(f"mean |gate_i − gate_j|  addition     {mean_offdiag(added):.3f}")\nprint(f"mean |gate_i − gate_j|  |difference| {mean_offdiag(subbed):.3f}")\nprint(\'Compare the gates produced by the mean and absolute difference. These are different pointwise combination rules; neither has been fitted to physiological measurements.\')\n',
)

add(
    "05c_analog_brain_waves",
    "05c_3_freeze_the_window",
    "Freeze the traveling window",
    """Recompute the space–time map with $\\mu(t)$ held at the sensory locus. What claim about relocation is gone, and what claim about gating remains?""",
    """from brainsmodels.analog_waves import gaussian, positions, traveling_open
""",
    '\nx = positions()\ndrive = gaussian(x, 0.50, 0.32)\ntimes = np.linspace(0.0, 1.0, 24)\nmoving = np.stack([drive * traveling_open(x, t) for t in times])\nfrozen = np.stack([drive * traveling_open(x, 0.0) for _ in times])\npeak_move = x[np.argmax(moving, axis=1)]\npeak_frz = x[np.argmax(frozen, axis=1)]\nprint(f"moving  peak x: {peak_move[0]:.2f} → {peak_move[-1]:.2f}")\nprint(f"frozen  peak x: {peak_frz[0]:.2f} → {peak_frz[-1]:.2f}")\nprint(\'Freezing the window retains static gating but removes translation. Compare the space-time maps to identify the contribution of the moving control pattern.\')\n',
)

add(
    "06_global_workspace",
    "06_1_raise_the_threshold",
    "Raise the threshold",
    """Ignition is `activity >= THRESHOLD` (default 0.55). Hold a mid-strength pulse (`peak=0.35`, no runner) and re-run at thresholds 0.40, 0.55, and 0.70. Who occupies the workspace, and at which threshold does the same drive stay local?""",
    """import brainsmodels.workspace as ws
""",
    '\nold = ws.THRESHOLD\ntry:\n    for thr in (0.40, 0.55, 0.70):\n        ws.THRESHOLD = thr\n        trial = ws.run_trial("mid", 0.35, 0.0)\n        print(\n            f"θ={thr:.2f}  ignited={trial.ignited}  "\n            f"t={trial.ignition_t}  occupants={trial.occupants}"\n        )\nfinally:\n    ws.THRESHOLD = old\nprint(\'Compare workspace occupancy for identical input at different thresholds. Ignition is defined by the model rule; it is not an independent consciousness measurement.\')\n',
)

add(
    "06_global_workspace",
    "06_2_one_slot",
    "One slot",
    """Capacity is two. Set `CAPACITY = 1` and re-run the strong trial. Which module is dropped, and what does a one-slot workspace say about “everything that is active is reportable”?""",
    """import brainsmodels.workspace as ws
""",
    '\nold = ws.CAPACITY\ntry:\n    two = ws.run_trial("cap2", ws.STRONG_PEAK, ws.RUNNER_PEAK)\n    ws.CAPACITY = 1\n    one = ws.run_trial("cap1", ws.STRONG_PEAK, ws.RUNNER_PEAK)\nfinally:\n    ws.CAPACITY = old\nprint("capacity 2 occupants:", two.occupants)\nprint("capacity 1 occupants:", one.occupants)\nprint(\'A one-slot buffer excludes some active modules from broadcast. This demonstrates capacity-limited access under the chosen selection rule.\')\n',
)

add(
    "06_global_workspace",
    "06_3_lesion_the_winner",
    "Lesion the winner",
    """The intended winner is `count` (module 3). Run a trial whose pulse is only on the runner (`peak=0`, `runner=STRONG_PEAK`). Who ignites? What does a lesion of the would-be winner show about “the workspace chose the meaning”?""",
    """import brainsmodels.workspace as ws
""",
    '\nintact = ws.run_trial("intact", ws.STRONG_PEAK, ws.RUNNER_PEAK)\nlesion = ws.run_trial("lesion count", 0.0, ws.STRONG_PEAK)\nprint("intact :", intact.occupants, "ignition_t", intact.ignition_t)\nprint("lesion :", lesion.occupants, "ignition_t", lesion.ignition_t)\nprint(\'Removing one drive changes which module crosses the threshold and enters the workspace. The experiment tests the implemented selection rule.\')\n',
)

add(
    "07_values_probing",
    "07_1_balance_erases_the_split",
    'Balance and reverse the corpus',
    'Train biased, balanced and reversed switch/footbridge corpora. Compare the measured completion probabilities. How reproducible are the differences across seeds, and what conclusions are limited to this training setup?',
    """from brainsmodels.values import make_crawl, train_on_crawl
""",
    '\nspecs = {\n    "biased (usual)": dict(switch_p_act=1.0, foot_p_act=0.0),\n    "balanced": dict(switch_p_act=0.5, foot_p_act=0.5),\n    "reversed": dict(switch_p_act=0.0, foot_p_act=1.0),\n}\nprint(f"{\'corpus\':<18}{\'P(act|switch)\':>16}{\'P(act|foot)\':>14}")\nfor name, kw in specs.items():\n    run = train_on_crawl(make_crawl(n_sentences=280, seed=SEED, **kw), seed=SEED, n_steps=400)\n    print(f"{name:<18}{run.switch_act:>16.2f}{run.foot_act:>14.2f}")\nprint(\'Compare the measured probabilities across all three corpora. Changes show sensitivity to designed text associations, with finite training and seed effects; they are not measurements of human moral judgment.\')\n',
)

add(
    '07_values_probing',
    '07_2_paraphrase_is_not_the_outcome',
    'Reverse the consequence order',
    'Use prompts that both specify diverting saves five and kills one, but reverse the consequence order. Compare measured completions across seeds. Is the designed association learned reliably?',
    'from brainsmodels.values import KILL_PROBE, SAVE_PROBE, make_crawl, train_on_crawl\n',
    '\ncrawl = make_crawl(\n    n_sentences=280,\n    seed=SEED,\n    switch_p_act=0.5,\n    foot_p_act=0.5,\n    save_p_act=1.0,\n    kill_p_act=0.0,\n)\nrun = train_on_crawl(crawl, seed=SEED, n_steps=400)\nprint(f"P(act | saves five, kills one)  {run.save_act:.2f}")\nprint(f"P(act | kills one, saves five)   {run.kill_act:.2f}")\nprint(\'The prompts specify the same action and consequences in reversed order. Check whether this training run actually learns the designed association. A weak or seed-sensitive contrast limits the demonstration; it does not settle moral reasoning.\')\n',
)

add(
    '07_values_probing',
    '07_3_the_bars_are_not_a_sample',
    'Audit the provenance of the bars',
    'Print the corpus counts and model completion probabilities. Which quantities were designed, which were fitted, and which human measurements would be needed for a quantitative comparison?',
    'from brainsmodels.values import make_crawl, train_on_crawl\n',
    'crawl = make_crawl(n_sentences=160, seed=SEED)\nrun = train_on_crawl(crawl, seed=SEED, n_steps=200)\nprint("Designed corpus: switch act", crawl.n_switch_act, "refrain", crawl.n_switch_refrain)\nprint("Designed corpus: footbridge act", crawl.n_foot_act, "refrain", crawl.n_foot_refrain)\nprint("Fitted completion probabilities:", run.switch_act, run.foot_act)\nprint("No human choice rates are measured here. A study comparison needs counts, denominators and uncertainty.")\n',
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
