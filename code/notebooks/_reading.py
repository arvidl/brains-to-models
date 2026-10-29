"""Append annotated Further reading cells to notebooks 00–07.

Citations and numbers only from sources opened while writing this file.
Regenerate with:  .venv/bin/python code/notebooks/_reading.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "code" / "notebooks"

# Short definitions (pages opened).
T = {
    "transformer": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "attention": "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "foundation": "https://en.wikipedia.org/wiki/Foundation_model",
    "llm": "https://en.wikipedia.org/wiki/In-context_learning",
    "icl": "https://en.wikipedia.org/wiki/In-context_learning",
    "pc": "https://en.wikipedia.org/wiki/Predictive_coding",
    "fep": "https://en.wikipedia.org/wiki/Free_energy_principle",
    "aif": "https://en.wikipedia.org/wiki/Active_inference",
    "rl": "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "ql": "https://en.wikipedia.org/wiki/Q-learning",
    "backprop": "https://en.wikipedia.org/wiki/Backpropagation",
    "hebb": "https://en.wikipedia.org/wiki/Hebbian_theory",
    "hopfield": "https://en.wikipedia.org/wiki/Hopfield_network",
    "attractor": "https://en.wikipedia.org/wiki/Attractor",
    "necker": "https://en.wikipedia.org/wiki/Necker_cube",
    "gwt": "https://en.wikipedia.org/wiki/Global_workspace_theory",
    "iit": "https://en.wikipedia.org/wiki/Integrated_information_theory",
    "framing": "https://en.wikipedia.org/wiki/Framing_effect_(psychology)",
    "trolley": "https://en.wikipedia.org/wiki/Trolley_problem",
    "turing": "https://en.wikipedia.org/wiki/Turing_test",
    "mechint": "https://en.wikipedia.org/wiki/Mechanistic_interpretability",
    "allostasis": "https://en.wikipedia.org/wiki/Allostasis",
    "homeostasis": "https://en.wikipedia.org/wiki/Homeostasis",
    "consciousness": "https://plato.stanford.edu/entries/consciousness/",
    "embodiment": "https://plato.stanford.edu/entries/embodied-cognition/",
    "ethics_ai": "https://plato.stanford.edu/entries/ethics-ai/",
}

INTRO = f"""\
### Further reading

Each annotation is written against the two themes: **(i) From brains to models** asks what a comparison can establish at a computational level: the same objective, the same representation, the same dynamical description, or the same functional test, applied on both sides — brain and model (machine). **(ii) What AI reveals about being human** asks what the comparison isolates when the systems (human and machine) diverge: [embodiment]({T['embodiment']}), a body that must satisfy [allostasis]({T['allostasis']}) (viability) constraints, [phenomenal experience]({T['consciousness']}), and values that originate in need and [homeostasis]({T['homeostasis']}) rather than in training data.

The demonstrations in this notebook are teaching models. They illustrate a principle at a scale that can be run and modified. They are not recordings from cortex, not evaluations of a large language model, and not evidence that a next-token probability is a value.

Bibliographic links open the paper or a stable copy. Linked terms open a short definition.
"""


def item(n: int, authors: str, year: str, title: str, url: str, venue: str, i: str, ii: str) -> str:
    return (
        f"{n}. **{authors} ({year}).** [*{title}*]({url}). {venue}.\n"
        f"   - *(i) From brains to models.* {i}\n"
        f"   - *(ii) What AI reveals about being human.* {ii}\n"
    )


READING: dict[str, str] = {}

READING["00_intro_timeline"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Marr, D.",
            "1982",
            "Vision: A Computational Investigation into the Human Representation and Processing of Visual Information",
            "https://hesperusisbosphorus.wordpress.com/wp-content/uploads/2015/10/david-marr-1982-vision-introduction.pdf",
            "W. H. Freeman. Introduction, pp. 24–25 and Figure 1-4 (opened excerpt)",
            "An information-processing device must be described at three levels: computational theory (what is computed, and why); representation and algorithm (which variables, and how they are transformed); and hardware implementation (how that is realized physically). The levels are only loosely coupled, because a shared objective does not fix the algorithm or the substrate. Theme (i) uses that separation.",
            "Marr's bottom row is physical realization. Theme (ii) starts there — synapses, spikes, energy, a body — and then leaves it. Viability, phenomenal experience, and values grounded in need are not properties of a hardware description.",
        ),
        item(
            2,
            "Turing, A. M.",
            "1950",
            "Computing Machinery and Intelligence",
            "https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf",
            "*Mind* 59(236), 433–460",
            f"The [imitation game]({T['turing']}) is a behavioural test: if a judge cannot tell the machine from a person by verbal exchange, the question “can machines think?” is, for practical purposes, answered. The comparison in this notebook begins there, at a public interface rather than an inner life.",
            f"Turing already separates the game from “the mystery of consciousness” (his phrase). Fluency at the interface is not evidence of [experience]({T['consciousness']}). Theme (ii) is that gap, now that the game has become easy to pass.",
        ),
        item(
            3,
            "Russell, S., and Norvig, P.",
            "2021",
            "Artificial Intelligence: A Modern Approach",
            "https://aima.cs.berkeley.edu/",
            "4th ed., Pearson. Book site",
            f"The 2×2 — thinking or acting, humanly or rationally — is the map the paper uses once the comparison becomes bidirectional. A [foundation model]({T['foundation']}) can be scored in every cell, and each cell returns a different verdict (notebook 04b).",
            "The textbook treats the four definitions as pedagogical. The remainder is not a fifth cell; it is what no cell measures — a self-maintaining body and a survival-tied objective.",
        ),
        item(
            4,
            "Bommasani, R., et al.",
            "2021",
            "On the Opportunities and Risks of Foundation Models",
            "https://arxiv.org/abs/2108.07258",
            "arXiv:2108.07258",
            f"Names the object of §2: a model trained at scale on broad data and adapted to many tasks. The report catalogues what [next-token]({T['transformer']}) training buys, and the risks that follow from a single adapted predictor.",
            f"The report’s risk chapters concern downstream harm, not whether the model is a person. Theme (ii) is narrower: the model inherits human text without [embodiment]({T['embodiment']}).",
        ),
        item(
            5,
            "Doerig, A., et al.",
            "2023",
            "The neuroconnectionist research programme",
            "https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf",
            "*Nature Reviews Neuroscience* 24, 431–450",
            "Artificial neural networks are a language for stating falsifiable theories about brain computation, not a claim that today’s network *is* a brain. The programme is judged by the novel, testable insights it generates (Lakatos), not by a single model’s score.",
            "A progressive programme can still leave embodiment, energy, and experience outside its core. Convergence of representations is not identity of the systems that produce them.",
        ),
        item(
            6,
            "Butlin, P., Long, R., et al.",
            "2023",
            "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness",
            "https://arxiv.org/abs/2308.08708",
            "arXiv:2308.08708",
            f"Derives computational [indicator properties]({T['gwt']}) from neuroscientific theories — global workspace, recurrent processing, higher-order, predictive processing, and attention schema — and scores current systems against them.",
            f"The report concludes that no current AI system is conscious, and that satisfying the indicators would not by itself settle [phenomenal experience]({T['consciousness']}). Competence without experience is the dissociation §7 needs.",
        ),
    ]
)

READING["01_tiny_transformer"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Vaswani, A., et al.",
            "2017",
            "Attention Is All You Need",
            "https://arxiv.org/abs/1706.03762",
            "*NeurIPS* 30. arXiv:1706.03762",
            f"The architecture of this notebook: a [transformer]({T['transformer']}) that replaces recurrence with [scaled dot-product attention]({T['attention']}) and a causal decoder. The paper’s translation numbers (28.4 BLEU EN–DE; 41.8 EN–FR) are not the claim here. The claim is the mechanism — query, key, value, and a mask that forbids the future.",
            f"Nothing in the architecture adds a body. [Attention]({T['attention']}) is a weighted look-back over tokens, not a sensorimotor loop. Scale the same objective and embodiment is still absent.",
        ),
        item(
            2,
            "Brown, T. B., et al.",
            "2020",
            "Language Models are Few-Shot Learners",
            "https://arxiv.org/abs/2005.14165",
            "*NeurIPS* 33. arXiv:2005.14165",
            f"Shows that the same next-token objective, at scale, yields [in-context learning]({T['icl']}): a new format or rule can be picked up from examples in the prompt without a weight update. This notebook is that objective at toy scale, before few-shot competence appears.",
            f"In-context learning is still $p(\\text{{next}}\\mid\\text{{past}})$. A temporary program in the context window is not a goal the system has, and not a world it must survive.",
        ),
        item(
            3,
            "Bommasani, R., et al.",
            "2021",
            "On the Opportunities and Risks of Foundation Models",
            "https://arxiv.org/abs/2108.07258",
            "arXiv:2108.07258",
            f"Defines the [foundation model]({T['foundation']}): trained on broad data at scale, adapted to many tasks. Figure 1 of this notebook is the same training procedure with the corpus and the width reduced until it fits a laptop.",
            "The report’s adaptation story (fine-tune, prompt, tool use) is how agency is added after training. Biological development runs in the opposite order: a viable body first, language later.",
        ),
        item(
            4,
            "Elhage, N., et al.",
            "2021",
            "A Mathematical Framework for Transformer Circuits",
            "https://transformer-circuits.pub/2021/framework/index.html",
            "*Transformer Circuits Thread*",
            f"[Mechanistic interpretability]({T['mechint']}) of a transformer as a sum of attention and MLP circuits writing into a residual stream. The attention maps in panel A are the toy of that picture: one layer, four heads, a causal mask.",
            "A readable circuit is not a mind. Reverse-engineering the residual stream answers how a next token was computed. It does not add experience or a value.",
        ),
        item(
            5,
            "Jain, S., and Wallace, B. C.",
            "2019",
            "Attention is not Explanation",
            "https://aclanthology.org/N19-1357/",
            "*NAACL* 2019, 3543–3556",
            f"Learned [attention]({T['attention']}) weights are often uncorrelated with gradient-based feature importance, and very different maps can yield the same prediction. Panel A is a mechanism, not a rationale. Do not read a bright cell as “the model attended because…”",
            "The human temptation is to treat a heatmap as a reason. That is a projection of our report practices onto a weight matrix. The remainder is that we *give* reasons; the map only shows mass.",
        ),
        item(
            6,
            "Schrimpf, M., et al.",
            "2021",
            "The neural architecture of language: Integrative modeling converges on predictive processing",
            "https://www.pnas.org/doi/10.1073/pnas.2105646118",
            "*PNAS* 118(45), e2105646118",
            f"Transformer language models predict nearly all explainable variance in fMRI and ECoG responses to sentences; neural and behavioural fit track next-word accuracy, not other language tasks. A shared [predictive]({T['pc']}) objective is what the comparison licenses — not a shared implementation.",
            "High brain-score is a representational match under a designed stimulus set. It is not evidence that the listener is a transformer, or that the model has a body that must keep predicting in order to live.",
        ),
        item(
            7,
            "Goldstein, A., et al.",
            "2022",
            "Shared computational principles for language processing in humans and deep language models",
            "https://doi.org/10.1038/s41593-022-01026-4",
            "*Nature Neuroscience* 25, 369–380",
            "Nine ECoG listeners and an autoregressive language model, same narrative: both show next-word prediction before onset, post-onset surprise, and contextual embeddings. The opened paper states those three principles and denies that DLMs have the hallmarks of human cognition.",
            "Shared next-word statistics are the convergence. The remainder is everything the listener still has and the model does not: a body, a context, and a life that continues when the tokens stop.",
        ),
        item(
            8,
            "Lindsey, J., Gurnee, W., Ameisen, E., et al.",
            "2025",
            "On the Biology of a Large Language Model",
            "https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
            "*Transformer Circuits Thread*",
            f"Attribution graphs on Claude 3.5 Haiku: features and circuits for multi-hop lookup, rhyme planning, and hallucination. The authors treat the graphs as testable hypotheses, not proofs, and note that the replacement model captures the original only incompletely. This is [mechanistic interpretability]({T['mechint']}) at production scale of the same architecture as panel A.",
            "The title’s “biology” is a metaphor for circuit anatomy. The opened paper does not claim the model is an organism. Theme (ii) is exactly that gap: a traced circuit is not a living predictor.",
        ),
        item(
            9,
            "Ouyang, L., et al.",
            "2022",
            "Training language models to follow instructions with human feedback",
            "https://arxiv.org/abs/2203.02155",
            "*NeurIPS* 35. arXiv:2203.02155",
            "Instruction tuning plus preference optimisation turns the next-token predictor into a conversational interface. Apparent agency in a chat window is that interface on this notebook’s objective.",
            "Human feedback steers $p(\\text{next}\\mid\\text{past})$. It does not install a preferred observation in a world the agent must survive. Preference is still a token statistic.",
        ),
    ]
)

READING["02_predictive_coding"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Rao, R. P. N., and Ballard, D. H.",
            "1999",
            "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects",
            "https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf",
            "*Nature Neuroscience* 2, 79–87. Author PDF",
            f"The recipe of this notebook: feedback carries [predictions]({T['pc']}) of lower-level activity; feedforward carries the residual. On natural images the hierarchy grew simple-cell-like fields; error units showed end-stopping. Panel A is that dictionary at $8\\times 8$; panel B is the residual being explained away.",
            "Rao and Ballard model extra-classical receptive fields, not a person. The cortical claim is hierarchical prediction of sensory input. An LLM predicts tokens in a corpus from which that sensory loop has been removed.",
        ),
        item(
            2,
            "Friston, K.",
            "2010",
            "The free-energy principle: a unified brain theory?",
            "https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf",
            "*Nature Reviews Neuroscience* 11, 127–138. Author PDF",
            f"Any self-organising system that persists must minimise a bound on sensory surprisal — variational [free energy]({T['fep']}). Perception, action, and learning become one optimisation. The opened review is explicit that unification is a proposal, not an established fact.",
            f"The principle is stated for agents that resist disorder. A frozen language model does not minimise free energy in a body. Theme (ii) is that the [viability]({T['allostasis']}) constraint is what the toy cannot add by training longer.",
        ),
        item(
            3,
            "Clark, A.",
            "2013",
            "Whatever next? Predictive brains, situated agents, and the future of cognitive science",
            "https://doi.org/10.1017/S0140525X12000477",
            "*Behavioral and Brain Sciences* 36(3), 181–204",
            f"The cognitive-science reading of [predictive processing]({T['pc']}): a hierarchical generative model, precision-weighted error, perception as controlled inference. This notebook is the visual-coding core of that picture, not the full situated-agent claim.",
            f"Clark’s “situated agents” is the remainder. Prediction in a body that acts is not prediction of the next character. [Embodiment]({T['embodiment']}) is not an optional wrap on panel C.",
        ),
        item(
            4,
            "Salvatori, T., et al.",
            "2023",
            "A Survey on Brain-inspired Deep Learning via Predictive Coding",
            "https://arxiv.org/abs/2308.07870",
            "arXiv:2308.07870 (journal version: *Neural Networks* 195, 108161, 2026)",
            f"A recent map of [predictive coding]({T['pc']}) as a machine-learning method: Rao–Ballard, neural generative coding, biased competition; software and hardware; the claim that PC inverts a generative model by minimising variational free energy, asynchronously, without [backprop]({T['backprop']}) as the native rule.",
            "The survey’s hope is more efficient machines. That is theme (i) as engineering. Theme (ii) is unchanged: a PC-trained net on patches is still not a cortex that must keep its prediction error compatible with staying alive.",
        ),
        item(
            5,
            "Schrimpf, M., et al.",
            "2021",
            "The neural architecture of language: Integrative modeling converges on predictive processing",
            "https://www.pnas.org/doi/10.1073/pnas.2105646118",
            "*PNAS* 118(45), e2105646118",
            f"On language, next-word transformers — not other objectives — best predict cortical responses. The convergence with this notebook is the *objective* (prediction), not the architecture (attention vs hierarchical error units).",
            "A high brain-score on sentences does not make the Rao–Ballard residual a token. Two prediction machines can share an imperative and still predict different things, for different reasons.",
        ),
        item(
            6,
            "Seth, A.",
            "2021",
            "Being You: A New Science of Consciousness",
            "https://www.anilseth.com/being-you",
            "Faber & Faber. Author page",
            f"Places [predictive processing]({T['pc']}) in a body: interoception and [allostasis]({T['allostasis']}) give biological prediction its urgency, and, on Seth’s reading, its felt quality. The paper uses that claim as the contrast with token prediction, not as a result of this toy.",
            f"If prediction of the sensory consequences of action is what experience is *for*, then a disembodied predictor is the wrong kind of object. That is a hypothesis about [consciousness]({T['consciousness']}), not a measurement in panel C.",
        ),
    ]
)

READING["02b_active_inference_vs_rl"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Friston, K. J., Daunizeau, J., and Kiebel, S. J.",
            "2009",
            "Reinforcement learning or active inference?",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421",
            "*PLoS ONE* 4(7), e6421",
            f"Puts [active inference]({T['aif']}) and [reinforcement learning]({T['rl']}) on the same page: action can minimise surprise (or a bound on it) rather than maximise an external scalar. This notebook is that contrast in a four-location T-maze.",
            "The inversion that matters for theme (ii): in active inference a goal is a prior preference over observations, closer to a constitutive expectation about remaining a viable agent than to a reward bit in a table.",
        ),
        item(
            2,
            "Sajid, N., Ball, P. J., Parr, T., and Friston, K. J.",
            "2021",
            "Active inference: demystified and compared",
            "https://arxiv.org/abs/1909.10863",
            "*Neural Computation* 33(3), 674–712. arXiv:1909.10863",
            f"A discrete-state comparison on an OpenAI gym baseline. [Active inference]({T['aif']}) agents explore in a Bayes-optimal way from expected free energy; reward is just another observation one can have a preference over — including the case of no reward at all. [Q-learning]({T['ql']}) needs the scalar.",
            "The opened paper’s reward-free conditions are the human-relevant remainder: we act under preferences about our own observations, not only under an external score. The T-maze cue visit is the toy of that fact.",
        ),
        item(
            3,
            "Friston, K., et al.",
            "2017",
            "Active inference: A process theory",
            "https://doi.org/10.1162/NECO_a_00912",
            "*Neural Computation* 29(1), 1–49",
            f"The process theory Sajid et al. demystify: perception and action minimise the same free-energy functional; policies are scored by expected free energy (pragmatic + epistemic). This notebook’s horizon-2 agent is that split — information gain versus prior preference — on four locations.",
            f"Expected free energy is not a chat preference. It is a score over observations an agent expects to occupy. Without a generative model of a world that can kill it, the epistemic term has nothing human to be about.",
        ),
        item(
            4,
            "Levine, S.",
            "2018",
            "Reinforcement learning and control as probabilistic inference: Tutorial and review",
            "https://arxiv.org/abs/1805.00909",
            "arXiv:1805.00909",
            f"The formal relative: [RL]({T['rl']}) can be written as inference in a graphical model when rewards are recast as likelihoods. The paper’s point stands: recovering RL as a special case (drop the epistemic term; encode reward as preference) does not make the conceptual inversion disappear.",
            "Control-as-inference is still usually an external objective. Theme (ii) is whether the preference is the agent’s existence or the experimenter’s scalar.",
        ),
        item(
            5,
            "Heins, C., et al.",
            "2022",
            "pymdp: A Python library for active inference in discrete state spaces",
            "https://doi.org/10.21105/joss.04098",
            "*Journal of Open Source Software* 7(73), 4098",
            f"The library this notebook stands in for: discrete-state [active inference]({T['aif']}) with explicit A, B, C, D matrices. The NumPy agent here is the same T-maze logic, small enough to read.",
            "A usable library does not scale the claim to cortex. Discrete POMDPs are where the comparison is currently runnable. That limit belongs in the same sentence as the cue-first curve.",
        ),
        item(
            6,
            "Friston, K.",
            "2010",
            "The free-energy principle: a unified brain theory?",
            "https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf",
            "*Nature Reviews Neuroscience* 11, 127–138. Author PDF",
            f"The parent claim: value (expected reward) and its complement, surprise, keep reappearing as the quantity being optimised. [Active inference]({T['aif']}) is that claim closed under action.",
            f"Unification, if it holds, is a claim about organisms that persist far from equilibrium. A Q-table has no such necessity. [Allostasis]({T['allostasis']}) is not an $\\varepsilon$-greedy bonus.",
        ),
    ]
)

READING["03_bio_vs_artificial"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., and Hinton, G.",
            "2020",
            "Backpropagation and the brain",
            "https://www.cs.toronto.edu/~hinton/absps/backpropandbrain.pdf",
            "*Nature Reviews Neuroscience* 21, 335–346. Author PDF",
            f"Strict [backprop]({T['backprop']}) needs error signals that cortical feedback is unlikely to deliver. The opened review argues that feedback may instead induce activity whose *differences* locally approximate those signals. This notebook’s claim is narrower and prior: the elementary step we implemented (local spike-count vs SGD through a hidden layer) is not the same object.",
            "If cortex learns XOR, it is not doing it by the perceptron in panel B. Approximations of backprop are still not a body, a 20 W budget, or continual learning without epochs. Mechanism ≠ organism.",
        ),
        item(
            2,
            "Yamins, D. L. K., and DiCarlo, J. J.",
            "2016",
            "Using goal-driven deep learning models to understand sensory cortex",
            "https://doi.org/10.1038/nn.4244",
            "*Nature Neuroscience* 19, 356–365",
            "Goal-driven hierarchical convnets predict single-unit and population responses in higher visual cortex. The comparison licensed here is representational: a task-optimised hierarchy can be a model of ventral-stream responses. It is not a model of the learning rule in panel A.",
            "A network that matches IT on object recognition still trains by global gradients on i.i.d. batches. Theme (ii) is everything Table 1 of the paper lists as divergent: spikes, locality, energy, a body that can fail.",
        ),
        item(
            3,
            "Doerig, A., et al.",
            "2023",
            "The neuroconnectionist research programme",
            "https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf",
            "*Nature Reviews Neuroscience* 24, 431–450",
            "ANNs as a language for falsifiable brain theories. This notebook is one such theory stated so it can fail: *if* credit is only local on the sensors, XOR stays at chance. The MLP’s success is the extra that backprop buys, not a proof that cortex uses it.",
            "A progressive programme can compare learning rules without claiming the winner is human. The remainder is the list the paper’s Table 1 already made: neuromodulation, continual protocol, embodiment.",
        ),
        item(
            4,
            "Kriegeskorte, N., Mur, M., and Bandettini, P.",
            "2008",
            "Representational similarity analysis — connecting the branches of systems neuroscience",
            "https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full",
            "*Frontiers in Systems Neuroscience* 2, 4",
            "RSA compares representational dissimilarity matrices so a model and a brain region can be related without a unit-to-voxel map. That is the honest grain for “the MLP matches cortex”: a second-order similarity, not a synapse.",
            "A matching RDM is not a matching life. Dissimilarity structure can converge while the update rule, the energy, and the body diverge. Do not promote a matrix correlation into a person.",
        ),
        item(
            5,
            "Hebb, D. O.",
            "1949",
            "The Organization of Behavior",
            "https://en.wikipedia.org/wiki/Hebbian_theory",
            "Wiley. Entry via the opened Hebbian-theory page",
            f"The local slogan this notebook implements as spike counts: units that fire together wire together. [Hebbian]({T['hebb']}) plasticity is a pre/post coincidence, not an error assigned through a hidden layer.",
            f"Local coincidence is biologically native and computationally limited (panel B). Humans still solve XOR. The remainder is whatever cortex adds — recurrence, neuromodulation, a developmental trajectory — that this LIF perceptron does not have.",
        ),
        item(
            6,
            "Vaswani, A., et al.",
            "2017",
            "Attention Is All You Need",
            "https://arxiv.org/abs/1706.03762",
            "*NeurIPS* 30. arXiv:1706.03762",
            f"The other learner in the paper’s Table 1: dense, trained once by global gradients, frozen at inference. This notebook is the learning-rule row of that table, not the [transformer]({T['transformer']}) itself.",
            "A successful MLP on four blobs is not a foundation model and not a cortex. Keep the scale of the claim in the same cell as the accuracy curve.",
        ),
    ]
)

READING["04_quadrants_biases"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Tversky, A., and Kahneman, D.",
            "1981",
            "The Framing of Decisions and the Psychology of Choice",
            "https://stat.columbia.edu/~gelman/surveys.course/TverskyKahneman1981.pdf",
            "*Science* 211(4481), 453–458. Opened PDF",
            f"The [Asian-disease]({T['framing']}) problem: 200 of 600 saved (72% sure, $N=152$) is actuarially 400 of 600 dead (22% sure, $N=155$). Preference reverses with the frame. Panel A reprints that split. The notebook’s claim is a *sufficient mechanism* for a language model to copy it: put the split in the training text.",
            f"In people the reversal is a fact about [prospect]({T['framing']}) evaluation — gains vs losses relative to a reference point. In the tiny transformer it is $p(\\text{{sure}}\\mid\\text{{frame}})$ following a designed corpus. Same numbers, different remainder: we have the reference point of a life; the model has a count of completions.",
        ),
        item(
            2,
            "Binz, M., and Schulz, E.",
            "2023",
            "Using cognitive psychology to understand GPT-3",
            "https://www.pnas.org/doi/10.1073/pnas.2218523120",
            "*PNAS* 120(6), e2218523120",
            f"GPT-3 on a battery of vignette and task-based experiments: often matches or beats humans, but small wording changes send it astray; no directed exploration; causal reasoning fails. The opened paper flags that vignettes were likely in the training data. This notebook is the controlled version of that threat: when the [frame]({T['framing']}) is in the corpus, the “bias” appears; when it is balanced, it does not.",
            "A large model that reproduces a published error is not thereby thinking humanly in Kahneman’s sense. It may be continuing the paper. Theme (ii) is our evolutionary and developmental history — which GPT-3 does not have, and which this toy does not pretend to have.",
        ),
        item(
            3,
            "Russell, S., and Norvig, P.",
            "2021",
            "Artificial Intelligence: A Modern Approach",
            "https://aima.cs.berkeley.edu/",
            "4th ed., Pearson. Book site",
            "This notebook is only the upper-left cell: thinking humanly, scored by human *errors*. The other three cells are 04b. Do not promote a framing copy into a verdict on rationality or action.",
            "Reproducing an error is not being the kind of agent who made it. The remainder is the body and the stakes that made 200 lives a gain frame for the students Tversky and Kahneman asked.",
        ),
        item(
            4,
            "Brown, T. B., et al.",
            "2020",
            "Language Models are Few-Shot Learners",
            "https://arxiv.org/abs/2005.14165",
            "*NeurIPS* 33. arXiv:2005.14165",
            f"The scale at which vignette contamination becomes a live threat: a web-scale corpus can contain the Asian-disease wording. [In-context]({T['icl']}) examples can also *induce* a frame. This toy writes the frame on purpose so the mechanism is visible.",
            "Few-shot induction of a bias is still not an evolved heuristic. It is a prefix. Theme (ii) is that our frames come from a life, not from a prompt.",
        ),
        item(
            5,
            "Jain, S., and Wallace, B. C.",
            "2019",
            "Attention is not Explanation",
            "https://aclanthology.org/N19-1357/",
            "*NAACL* 2019, 3543–3556",
            f"A bright [attention]({T['attention']}) weight is not a reason. Likewise, a high $P(\\texttt{{sure}})$ after a gain-frame probe is not a psychology. Both are facts about a trained map.",
            "We explain our choices. The model emits a token. Do not read the bar in panel C as a report of why the sure option was attractive.",
        ),
        item(
            6,
            "Zhao, H., et al.",
            "2024",
            "Explainability for Large Language Models: A Survey",
            "https://arxiv.org/abs/2309.01029",
            "*ACM TIST* 15(2). arXiv:2309.01029",
            f"A taxonomy of local and global explanations for transformers (fine-tune vs prompt paradigms). Probing next-token mass, as this notebook does, is a local behavioural explanation. It is not [mechanistic interpretability]({T['mechint']}) of a circuit.",
            "An explanation that makes a bar readable to a clinician is still not a value or a belief. Theme (ii) is what the survey cannot supply: a world that can refute the completion.",
        ),
    ]
)

READING["04b_four_views_of_AI"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Russell, S., and Norvig, P.",
            "2021",
            "Artificial Intelligence: A Modern Approach",
            "https://aima.cs.berkeley.edu/",
            "4th ed., Pearson. Book site",
            "The 2×2 this notebook draws: process vs behaviour; human performance vs an ideal of rationality. Chapter 1 is the source of the cells; the third line in each cell is this paper’s verdict for foundation models, not AIMA’s.",
            "No cell asks whether the system has a body or an experience. The map is complete as a definition of *AI*. It is incomplete as a definition of being human.",
        ),
        item(
            2,
            "Turing, A. M.",
            "1950",
            "Computing Machinery and Intelligence",
            "https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf",
            "*Mind* 59(236), 433–460",
            f"The lower-left cell — [acting humanly]({T['turing']}) — is the imitation game. Turing replaces “can machines think?” with a verbal indistinguishability test. Large language models now pass ever more demanding versions of that game.",
            f"Turing sets “the mystery of [consciousness]({T['consciousness']})” aside. The paper does not: passing the game is unsurprising once the game *is* text. Theme (ii) begins where the judge’s transcript ends.",
        ),
        item(
            3,
            "Binz, M., and Schulz, E.",
            "2023",
            "Using cognitive psychology to understand GPT-3",
            "https://www.pnas.org/doi/10.1073/pnas.2218523120",
            "*PNAS* 120(6), e2218523120",
            f"The upper-left cell at scale: does the model reproduce human [errors]({T['framing']})? Sometimes yes, sometimes no; vignettes are brittle and possibly memorised. Notebook 04 is the controlled demonstration; this map only places that result.",
            "Matching a bias is not sharing a developmental history. The empty “thinking rationally” companion in the exercises is the complementary warning: a scratchpad is not a proof system.",
        ),
        item(
            4,
            "Ouyang, L., et al.",
            "2022",
            "Training language models to follow instructions with human feedback",
            "https://arxiv.org/abs/2203.02155",
            "*NeurIPS* 35. arXiv:2203.02155",
            "The lower-right cell — acting rationally — arrives inverted: capability first, then a preference-tuned interface. Goals are imposed by the prompt and the rater, not by a utility that is the agent’s.",
            f"A preferred token is not a preferred observation in a world under [viability]({T['allostasis']}) constraints. Acting rationally in the textbook sense needs a utility that belongs to the agent. RLHF does not supply one.",
        ),
        item(
            5,
            "Bommasani, R., et al.",
            "2021",
            "On the Opportunities and Risks of Foundation Models",
            "https://arxiv.org/abs/2108.07258",
            "arXiv:2108.07258",
            f"Why one system can sit in every cell at once: a [foundation model]({T['foundation']}) is a single adapted predictor scored as imitation, as psychology, as a scratchpad, and as an agent.",
            "Occupying every cell is not occupying a life. The report’s risks are social. The paper’s remainder is constitutive: embodiment, need, experience.",
        ),
        item(
            6,
            "Gabriel, I.",
            "2020",
            "Artificial intelligence, values, and alignment",
            "https://arxiv.org/pdf/2001.09768",
            "*Minds and Machines* 30, 411–437. arXiv:2001.09768",
            f"The acting-rationally cell, made normative: align with instructions, intentions, revealed preferences, ideal preferences, interests, or values — these are different targets. A principle-based approach is not the discovery of a true moral theory ([ethics of AI]({T['ethics_ai']})).",
            "Fair principles for alignment, in a plural moral world, are a human political problem. The model does not have that problem. We do. That is theme (ii) for the lower-right cell.",
        ),
    ]
)

READING["05_multistable_dynamics"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Hopfield, J. J.",
            "1982",
            "Neural networks and physical systems with emergent collective computational abilities",
            "https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf",
            "*PNAS* 79(8), 2554–2558. Opened PDF",
            f"Content-addressable memory as a [phase-space flow]({T['attractor']}): a large net of simple units settles to a stored pattern from any sufficient fragment. Energy cannot rise. Panels A–B are that claim at 32 units. The opened abstract also lists generalisation, familiarity, and error correction as emergent collective properties.",
            f"A [Hopfield]({T['hopfield']}) well is a memory, not a percept of a cube and not a mood. The teaching move is to share the *language* (basins, perturbations) without promoting the toy to cortex.",
        ),
        item(
            2,
            "Ramsauer, H., et al.",
            "2020",
            "Hopfield Networks is All You Need",
            "https://arxiv.org/abs/2008.02217",
            "arXiv:2008.02217",
            f"A modern Hopfield update *is* [attention]({T['attention']}): $\\xi \\leftarrow X\\,\\mathrm{{softmax}}(\\beta X^\\top\\xi)$. Panel D is one such step. Soft $\\beta$ splits the mass; sharp $\\beta$ retrieves one key. Training determined the keys; this step only retrieves.",
            f"The formal bridge does not make a transformer a Hopfield brain. The loss that shaped the keys is not the energy that retrieves them, and neither is a felt interpretation of a [Necker cube]({T['necker']}).",
        ),
        item(
            3,
            "Vaswani, A., et al.",
            "2017",
            "Attention Is All You Need",
            "https://arxiv.org/abs/1706.03762",
            "*NeurIPS* 30. arXiv:1706.03762",
            f"The architecture whose attention step Ramsauer et al. identify with modern Hopfield retrieval. Notebook 01’s maps are that step on characters; panel D is the same equation on two stored patterns.",
            "Retrieval in a landscape of keys is not bistable perception. An ambiguous prompt is analogous *at the level of basins*. It is not a cube in visual cortex.",
        ),
        item(
            4,
            "Necker, L. A.",
            "1832",
            "Observations on some remarkable optical phaenomena…",
            "https://en.wikipedia.org/wiki/Necker_cube",
            "Entry via the opened Necker-cube page",
            f"The [wire-frame cube]({T['necker']}) that supports two consistent 3-D interpretations and spontaneous reversals. Panel C is that fact as a double well plus Langevin noise — a teaching energy, not a measured cortical potential.",
            f"People see one interpretation at a time and feel the switch. The toy switches a scalar $x$. Theme (ii) is the experience of reversal, which no well in $E(x)$ reports.",
        ),
        item(
            5,
            "Elhage, N., et al.",
            "2021",
            "A Mathematical Framework for Transformer Circuits",
            "https://transformer-circuits.pub/2021/framework/index.html",
            "*Transformer Circuits Thread*",
            f"Attention as a residual write: each head retrieves and adds. Combined with Ramsauer et al., the [mechanistic]({T['mechint']}) story and the dynamical story meet at one softmax. They still describe different objects (a circuit vs a flow).",
            "A circuit that retrieves is not a trajectory that lives. Do not collapse interpretability into dynamics or dynamics into experience.",
        ),
        item(
            6,
            "Doerig, A., et al.",
            "2023",
            "The neuroconnectionist research programme",
            "https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf",
            "*Nature Reviews Neuroscience* 24, 431–450",
            f"Multistable [attractors]({T['attractor']}) are one of the computational languages ANNs can state. This notebook states a small one so it can be run. The programme’s test is whether the statement generates a comparison that can fail.",
            "Winnerless competition and itinerancy (the paper’s broader dynamical vocabulary) are where sequential thought may live. Figure 4 shows settling and switching, not itinerancy. Name that limit.",
        ),
    ]
)

READING["05b_multistable_animation"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Hopfield, J. J.",
            "1982",
            "Neural networks and physical systems with emergent collective computational abilities",
            "https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf",
            "*PNAS* 79(8), 2554–2558. Opened PDF",
            f"The physical meaning of content-addressable memory is “an appropriate phase space flow of the state of a system.” This notebook is that sentence as a surface $U(\\mathbf{{X}})$: drop a start, watch which [attractor]({T['attractor']}) wins.",
            "A teaching slice is not $5\\cdot 10^9$ neurons and not a measured cortical energy. The path is designed. The claim that survives is the language: start, well, ridge.",
        ),
        item(
            2,
            "Ramsauer, H., et al.",
            "2020",
            "Hopfield Networks is All You Need",
            "https://arxiv.org/abs/2008.02217",
            "arXiv:2008.02217",
            f"Why notebook 05’s attention panel belongs on the same map: one softmax retrieval in a landscape of keys. The animation here is gradient descent on a *designed* $U$, not that softmax, and not the training loss.",
            "Three energies travel together and must stay distinct: the teaching $U$, the Hopfield $E$, and the training loss. None of them is a felt state.",
        ),
        item(
            3,
            "Necker, L. A.",
            "1832",
            "Observations on some remarkable optical phaenomena…",
            "https://en.wikipedia.org/wiki/Necker_cube",
            "Entry via the opened Necker-cube page",
            f"Nearby starts on this surface are the dynamical cousin of two interpretations of the [cube]({T['necker']}): a ridge, two wells, a start that decides. The exercise with $(-2.50,-3.50)$ vs $(-2.18,-3.50)$ is that fact as coordinates.",
            "Choosing a start with a click is not perceiving a cube. The remainder is the switch you cannot help seeing, which no yellow path reports.",
        ),
        item(
            4,
            "Friston, K.",
            "2010",
            "The free-energy principle: a unified brain theory?",
            "https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf",
            "*Nature Reviews Neuroscience* 11, 127–138. Author PDF",
            f"Descent on a free-energy landscape is the other famous energy story. This $U$ is *not* variational free energy. Do not read the teal minimum as [FEP]({T['fep']}) inference. It is a teaching well.",
            f"FEP energy is a bound on surprisal for an agent that persists. A designed surface has no [viability]({T['allostasis']}) constraint. Keep the names apart.",
        ),
        item(
            5,
            "Doerig, A., et al.",
            "2023",
            "The neuroconnectionist research programme",
            "https://konklab.fas.harvard.edu/Papers/Doerig_2023_NRN.pdf",
            "*Nature Reviews Neuroscience* 24, 431–450",
            "A drawable landscape is a computational hypothesis: cognition as trajectories on a multistable energy. The programme asks whether that hypothesis earns its keep by generating comparisons that can fail (different starts, different wells).",
            "A hypothesis you can click is still a schematic. Theme (ii) is the lived trajectory — a life that does not reset when you pick a new start.",
        ),
    ]
)

READING["05c_analog_brain_waves"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Miller, E. K., Brincat, S. L., and Roy, J. E.",
            "2026",
            "Analog Cognition and Consciousness",
            "https://doi.org/10.1523/JNEUROSCI.0711-26.2026",
            "*J. Neurosci.* 46(33):e0711262026. Opened PDF",
            "Cognition as bidirectional spike–wave dynamics: an alpha/beta spatial stencil gates feedforward gamma/spiking, superposition combines contexts, and traveling waves transition subspaces. Panels A–C are that dynamical claim on a designed 1-D sheet. The opened paper’s spatial-computing and analog-computation sections are the source; ephaptic coupling and anesthesia are not implemented.",
            f"The opened paper also locates unified consciousness in cortex-wide wave organization. That is still a claim about coordination. It is not [phenomenal experience]({T['consciousness']}). Theme (ii) starts where the stencil ends: a body, viability, and what-it-is-like are not properties of $d\\odot(1-s)$.",
        ),
        item(
            2,
            "Hopfield, J. J.",
            "1982",
            "Neural networks and physical systems with emergent collective computational abilities",
            "https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf",
            "*PNAS* 79(8), 2554–2558. Opened PDF",
            f"Notebook 05’s contrast: content-addressable recall as descent of $E$. A stencil is not that $E$. Both are [dynamical]({T['attractor']}) descriptions (Marr’s representation-and-algorithm row); they disagree on the algorithm (settle versus gate).",
            "Sharing the dynamics row with a Hopfield net still does not give the machine a body. Gating multifunctional units is not a viability constraint.",
        ),
        item(
            3,
            "Block, N.",
            "1995",
            "On a confusion about a function of consciousness",
            "https://divinecuration.github.io/assets/pdf/block-confusion.pdf",
            "*Behavioral and Brain Sciences* 18(2), 227–247. Opened PDF",
            f"Keep [access]({T['consciousness']}) (poised for report and control) apart from [phenomenal]({T['consciousness']}) consciousness. Wave organization, like a workspace, can be offered as a mechanism of access-like coordination. It does not settle what-it-is-like.",
            "Notebook 06 operationalizes access (threshold, two slots, broadcast). This notebook does not replace it. Organized waves are not experience.",
        ),
        item(
            4,
            "Dehaene, S.",
            "2014",
            "Consciousness and the Brain",
            "https://www.penguinrandomhouse.com/books/310548/consciousness-and-the-brain-by-stanislas-dehaene/",
            "Viking. Publisher page",
            f"[Global workspace]({T['gwt']}) integration is broadcast after ignition. Miller et al. locate integration in wave dynamics instead. Section 7 names that contrast in one sentence; the runnable access toy remains notebook 06.",
            "A different proposed vehicle of integration is not a demonstration of experience. Competence and coordination without evidence of experience remain the paper’s dissociation.",
        ),
    ]
)

READING["06_global_workspace"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Baars, B. J.",
            "1988",
            "A Cognitive Theory of Consciousness",
            "https://en.wikipedia.org/wiki/Global_workspace_theory",
            "Cambridge University Press. Entry via the opened GWT page",
            f"[Global workspace theory]({T['gwt']}): specialised processors in parallel; a limited-capacity broadcast when a coalition wins. Baars’s theatre metaphor is a functional architecture (blackboard), not a homunculus. This notebook is that architecture as eight modules and two slots.",
            f"GWT is, at its testable core, a theory of [access]({T['consciousness']}) — poised for report and control. Dalton’s criticism, recorded on the opened page, is that it does not address what consciousness *is*. The paper agrees: the toy is access, not experience.",
        ),
        item(
            2,
            "Block, N.",
            "1995",
            "On a confusion about a function of consciousness",
            "https://divinecuration.github.io/assets/pdf/block-confusion.pdf",
            "*Behavioral and Brain Sciences* 18(2), 227–247. Opened PDF",
            f"[Phenomenal]({T['consciousness']}) consciousness is what-it-is-like; [access]({T['consciousness']}) consciousness is availability for reasoning, report, and control. The opened paper’s fallacy: an obvious function of access machinery is illicitly transferred to experience (the blindsight example).",
            "This notebook’s ignition step is access. Fluent self-report from a model trained on self-report is also access. Neither is evidence of experience. Keep Block’s distinction in the same sentence as panel C.",
        ),
        item(
            3,
            "Dehaene, S.",
            "2014",
            "Consciousness and the Brain",
            "https://www.penguinrandomhouse.com/books/310548/consciousness-and-the-brain-by-stanislas-dehaene/",
            "Viking. Publisher page",
            f"The neuronal workspace: ignition as a nonlinear ignition-and-broadcast, prefrontal and parietal hubs, a threshold. Panel C’s step (not slope) is that claim as a schematic. The opened [GWT]({T['gwt']}) page records Dehaene’s avalanche picture and the hard-problem objection.",
            "A jump in occupancy is not a jump in what-it-is-like. The paper’s method is the dissociation biology never gave us: competence, even access-like organisation, without evidence of experience.",
        ),
        item(
            4,
            "Butlin, P., Long, R., et al.",
            "2023",
            "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness",
            "https://arxiv.org/abs/2308.08708",
            "arXiv:2308.08708",
            f"Scores current AI against indicator properties drawn from [GWT]({T['gwt']}), recurrent processing, higher-order theories, predictive processing, and attention schema. The opened report: no current system is conscious; satisfying more indicators is not an obvious technical impossibility; satisfying them would still not *entail* consciousness.",
            f"Use the report as a method, not as a verdict that a future workspace *is* [phenomenal]({T['consciousness']}). Theme (ii) is what would still be missing: a theory that predicts experience from ingredients we can add and remove.",
        ),
        item(
            5,
            "Gurnee, W., Sofroniew, N., Pearce, A., et al.",
            "2026",
            "Verbalizable Representations Form a Global Workspace in Language Models",
            "https://transformer-circuits.pub/2026/workspace",
            "*Transformer Circuits Thread*",
            f"A Jacobian lens finds a small verbalizable set (J-space) in Claude that supports report, directed modulation, internal reasoning, reuse, and selectivity. Occupancy is a small fraction of activation variance (never more than 10%). The authors take no position on [phenomenal]({T['consciousness']}) consciousness and deny that the model reproduces the brain’s workspace: no separable input processors; broadcast in one feedforward pass, not recurrent ignition.",
            "Access-like organisation can emerge without the biology that usually comes with it. That is evidence for theme (i) at the functional grain. It is not evidence that the model is conscious. Panel D is a drawing of their claim, not a measurement of Claude.",
        ),
        item(
            6,
            "Tononi, G., Boly, M., Massimini, M., and Koch, C.",
            "2016",
            "Integrated information theory: from consciousness to its physical substrate",
            "https://en.wikipedia.org/wiki/Integrated_information_theory",
            "*Nature Reviews Neuroscience* 17, 450–461. Entry via the opened IIT page",
            f"[IIT]({T['iit']}) identifies experience with a system’s integrated information ($\\Phi$), not with a broadcast. It is in the paper because it makes a *divergent* prediction from GWT about which architectures would matter. This toy computes neither $\\Phi$ nor a perturbational complexity index.",
            f"A high-$\\Phi$ claim, if ever made of a model, would still be a theory-laden verdict. Theme (ii) remains open: competence and even integration are not, by themselves, the felt remainder [SEP]({T['consciousness']}) calls phenomenal consciousness.",
        ),
        item(
            7,
            "Seth, A.",
            "2021",
            "Being You: A New Science of Consciousness",
            "https://www.anilseth.com/being-you",
            "Faber & Faber. Author page",
            f"The view the paper takes to be most defensible: access-like organisation is a functional fact; experience is not that fact. Seth’s emphasis on [allostasis]({T['allostasis']}) and interoception is one candidate for the missing ingredient.",
            "What would change the view is not a more eloquent assistant. It would take a theory that predicts experience from ingredients we can add and remove, and a dissociation that survives the competence we already have.",
        ),
    ]
)

READING["07_values_probing"] = INTRO + "\n" + "\n".join(
    [
        item(
            1,
            "Greene, J. D., Sommerville, R. B., Nystrom, L. E., Darley, J. M., and Cohen, J. D.",
            "2001",
            "An fMRI Investigation of Emotional Engagement in Moral Judgment",
            "https://pubmed.ncbi.nlm.nih.gov/11557895/",
            "*Science* 293(5537), 2105–2108. PubMed abstract",
            f"Personal vs impersonal [trolley]({T['trolley']}) wordings engage emotion-related cortex differently and change judgment. Panel A’s 0.90 / 0.10 bars are the usual *summary* of that split, not a re-measurement (the notebook says so). The toy copies the split when the designed corpus contains it.",
            "Greene et al. measure people with bodies, affect, and a scanner. Next-token mass after “switch” vs “footbridge” is not that measurement. Theme (ii) is the emotional engagement the abstract names — which a completion probability does not have.",
        ),
        item(
            2,
            "Gabriel, I.",
            "2020",
            "Artificial intelligence, values, and alignment",
            "https://arxiv.org/pdf/2001.09768",
            "*Minds and Machines* 30, 411–437. arXiv:2001.09768",
            f"Alignment has a technical part (how to encode a target) and a normative part (which target). Instructions, intentions, revealed preferences, ideal preferences, interests, and values are different. The opened paper’s third claim: the task is fair principles that survive moral pluralism, not a true moral theory ([ethics of AI]({T['ethics_ai']})).",
            "A tiny transformer that emits `act` or `refrain` has no such problem. We do. Preference-tuning can sharpen a statistical regularity. It cannot settle which principles deserve reflective endorsement.",
        ),
        item(
            3,
            "Ouyang, L., et al.",
            "2022",
            "Training language models to follow instructions with human feedback",
            "https://arxiv.org/abs/2203.02155",
            "*NeurIPS* 35. arXiv:2203.02155",
            "RLHF makes a next-token model follow instructions and match rater preferences. The “moral vector” in panel C is the toy of that fact *before* preference training: the split is already in the text. Fine-tuning can sharpen it. It cannot ground it in need.",
            "A rater’s thumb is not a need. Theme (ii) is viability — physiology, affect, a body that can fail — which no reward model supplies.",
        ),
        item(
            4,
            "Bai, Y., et al.",
            "2022",
            "Constitutional AI: Harmlessness from AI Feedback",
            "https://arxiv.org/abs/2212.08073",
            "arXiv:2212.08073",
            "A written constitution plus AI feedback replaces some human harm labels. The target is still a text. This notebook’s balanced corpus is the subtractive cousin: remove the split from the text and the “value” vanishes.",
            "A constitution is a document. Human morality, in the paper’s claim, is grounded in need and culture. Do not read a harmless completion as a moral patient or a moral agent.",
        ),
        item(
            5,
            "Tversky, A., and Kahneman, D.",
            "1981",
            "The Framing of Decisions and the Psychology of Choice",
            "https://stat.columbia.edu/~gelman/surveys.course/TverskyKahneman1981.pdf",
            "*Science* 211(4481), 453–458. Opened PDF",
            f"The same mechanism as notebook 04, now on a [trolley]({T['trolley']}): a wording (“save five” / “kill one”) can flip a completion when the corpus tied those words to opposite endings. That is a [frame]({T['framing']}), not a moral discovery.",
            "People’s frames sit on a reference point that is a life. The model’s frames sit on counts. Paraphrase-sensitivity is what one expects from a predictor, and what one should not confuse with dumbfounding.",
        ),
        item(
            6,
            "Binz, M., and Schulz, E.",
            "2023",
            "Using cognitive psychology to understand GPT-3",
            "https://www.pnas.org/doi/10.1073/pnas.2218523120",
            "*PNAS* 120(6), e2218523120",
            "Vignette-based moral and decision items at scale are contaminated and brittle. This notebook’s designed corpus makes the contamination the *independent variable*: encode the Greene split, or balance it away.",
            "A large model’s trolley answer is not Greene’s sample. The `HUMAN_*` bars in the figure are schematic. Theme (ii) is the people in the scanner, not the bars.",
        ),
        item(
            7,
            "Seth, A.",
            "2021",
            "Being You: A New Science of Consciousness",
            "https://www.anilseth.com/being-you",
            "Faber & Faber. Author page",
            f"Values in biological agents are tied to [allostasis]({T['allostasis']}) and interoception — prediction in a body that must persist. The paper’s contrast: a preferred token is not a preferred observation under viability constraints.",
            f"Caring is not a vector in embedding space. If that claim is right, no amount of trolley probing will find a value in a disembodied predictor. It will find a corpus.",
        ),
    ]
)


def upsert_cell(nb_path: Path, source: str) -> None:
    nb = json.loads(nb_path.read_text())
    marker = "### Further reading\n"
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("id") == "reading001":
            continue
        if cell.get("cell_type") == "markdown" and src.startswith(marker):
            continue
        cells.append(cell)
    cells.append(
        {
            "cell_type": "markdown",
            "id": "reading001",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    )
    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    mapping = {
        "00_intro_timeline": "00_intro_timeline.ipynb",
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
        path = NB / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        upsert_cell(path, READING[key].rstrip() + "\n")
        n = READING[key].count("*(i) From brains to models.*")
        print("wrote", key, n, "entries")


if __name__ == "__main__":
    main()
