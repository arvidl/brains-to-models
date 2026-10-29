"""Append Questions for reflection cells to notebooks 00–07 (including b).

Ten questions per notebook, written against the two themes and for both
audiences. Inserted immediately before Further reading. The two-theme
opener lives only in the Further reading cell; do not repeat it here.

Regenerate with:  .venv/bin/python code/notebooks/_questions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _reading import T

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "code" / "notebooks"

INTRO = """\
### Questions for reflection

Not a quiz. Each question can be asked of a brain and of a machine. Write a short answer that a clinician or a neuroscientist, and a mathematically inclined reader, could both test. The two themes are defined in Further reading below.
"""


def qs(*items: str) -> str:
    lines = [INTRO, ""]
    for i, q in enumerate(items, start=1):
        lines.append(f"{i}. {q}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


ITEMS: dict[str, tuple[str, ...]] = {}

ITEMS["00_intro_timeline"] = (
    "Marr’s three levels apply to a brain and to a machine. If two systems share a computational theory (what is computed, and why), what has that established, and what has it left open?",
    "The paper renames Marr’s rows as function, dynamics, and architecture. Take one later section you already know — a transformer, a framing bias, a trolley wording — and state which row the comparison is actually being made on.",
    "Loose coupling means the same representation can be updated by different algorithms. Give one clinical example (two routes to the same diagnosis) and one formal example (two algorithms for the same map).",
    f"The tan band is not a fourth Marr level. Explain why [viability]({T['allostasis']}), [phenomenal experience]({T['consciousness']}), and values grounded in need are not properties of a hardware description.",
    "Theme (ii) starts from synapses, spikes, energy, and a body, then leaves that row. Suppose you had a complete implementational account of both a transformer and a cortex: what would still be missing?",
    "Before McCulloch–Pitts the comparison runs in one direction: the newest machine explains the brain. What observation would show that a metaphor had become bidirectional, and which Marr row would that observation occupy?",
    f"A [foundation model]({T['foundation']}) produces language that invites the reverse question. If fluency at the interface is the evidence, which Russell–Norvig cell are you scoring, and what does that score fail to capture?",
    "The timeline is a labelled set of points, not a causal model of progress. Which claim survives when several AI winters, many architectures, and almost all of neuroscience are omitted?",
    "Changing `FOCUS` or the year window filters the catalogue. Why does that leave both schematics unchanged, and what limit does that separation protect?",
    f"A shared objective — prediction — does not give a machine a body. State that limit twice: once as a Marr fact (agreement on a single row) and once as a human fact ([embodiment]({T['embodiment']})).",
)

ITEMS["01_tiny_transformer"] = (
    f"If a radiologist and this model both complete “the patient was started on …”, what would have to be the same for that to count as the same [computational theory]({T['transformer']}) (what is computed, and why)? What would still differ at implementation?",
    "The loss is $-\\log p(\\text{next}\\mid\\text{past})$. Name a clinical prediction that is scored the same way. What cost does the clinical miss pay that this loss does not?",
    f"[Attention]({T['attention']}) is a weighted look-back over the prompt. When you glance back in a note, what extra constraint does that glance have that a causal head does not?",
    "Four heads specialise in routing. What would it take to treat a head as a neurological “module,” and why is that a category error on this teaching model?",
    f"The corpus is a designed mixture of five frames. If a [foundation model]({T['foundation']}) sounds like a colleague, which mixture component is it matching — and what does that match fail to include?",
    "Dropping the causal mask lets a query see characters that have not been written. Why is that fatal at generation time? Why is a speaker who plans a sentence not relying on the same shortcut?",
    "Temperature sharpens or flattens $p(\\cdot\\mid\\text{prompt})$. Is a more peaked categorical a stronger belief? What would a Bayesian clinician mean by belief that this vector does not?",
    "Scaling keeps the same objective. If competence rises, which of Marr’s three rows has changed, and which has not?",
    f"[Embodiment]({T['embodiment']}) has been removed from the training signal. Name one measurement you could add, in principle, that would put a viability constraint back — and say why next-token mass is not that measurement.",
    "Suppose the model’s next character is correct and the clinician’s next word is also correct. What, exactly, is still missing if the comparison is meant to be about being human?",
)

ITEMS["02_predictive_coding"] = (
    f"This net and notebook 01 both minimise a prediction error. What is being predicted in each case, and why is that a difference at the computational level, not only in the data type? ([Predictive coding]({T['pc']}).)",
    "Inference updates $r$ with $U$ frozen; learning updates $U$ slowly. In a patient, what is the analogue of “explaining this sensation now” versus “changing what sensations of this kind will mean later”?",
    "A falling residual on an $8\\times 8$ patch is the score. What would that establish about V1, and what would it not?",
    "Whitening makes the Euclidean residual a less silly proxy for unexplained structure. If you skip it, what changes in the mathematics, and what clinical “normalisation” is that roughly like?",
    "A one-level net can still shrink $e_0$. What does the second level add that a clinician would call context, and that a mathematician would call a prior on $r_1$?",
    "There is no class label. What would adding a diagnostic loss do to the job this teaching model is here to illustrate?",
    f"The free-energy claim is that prediction is for staying viable ([allostasis]({T['allostasis']})). Panel C cannot show that “why.” What extra ingredient would have to enter the energy?",
    "Oriented blobs resemble simple-cell fields. Agreement of shape is which of Marr’s rows? Why is it not an agreement of hardware?",
    f"An LLM predicts other people’s tokens; this net predicts its own input. If both are “prediction machines,” what does theme (ii) still isolate? ([Embodiment]({T['embodiment']}).)",
    "Change `n_level1` and the tiles move. Which claim should not move?",
)

ITEMS["02b_active_inference_vs_rl"] = (
    f"On trial 1, the [active-inference]({T['aif']}) agent visits the cue. What term in expected free energy paid for that detour, and what clinical habit is that term a schematic of?",
    f"[Q-learning]({T['ql']}) can acquire the same policy after punishment. Why is “after enough bruises” not a viable story for a body, even if the TD update is sound?",
    "Drop the epistemic term. Who still goes to the cue, and what two ingredients were doing the work? (You can answer from the equations without re-running.)",
    "Preferences $C$ are priors over observations, not a reward sitting in the world. What preferred observation would a hypoxic patient be acting to avoid, and how is that different from a preferred token?",
    "The $Q$-table never represents context $c$. It represents (location, memory bit). When is that enough as an algorithm, and when does a clinician still need the hidden diagnosis?",
    f"Adding an exploration bonus to [RL]({T['rl']}) can imitate cue-first behaviour. Why does that tweak not erase the paper’s contrast between the two scores?",
    "Horizon 1 versus horizon 2: what does a one-step lookahead fail to value that a two-step policy can?",
    "Same maze, two objectives. What computational requirement is the cue-first policy a candidate for?",
    f"Even the active-inference agent has no metabolism. What would “value as a belief” still lack if $G$ were minimised in a machine with no body that can fail [homeostasis]({T['homeostasis']})?",
    "As usually implemented, active inference is small and discrete; RL is large. Does that fact change the trial-1 gap, or only what we can deploy?",
)

ITEMS["03_bio_vs_artificial"] = (
    "Same points, two updates. Which of Marr’s rows agrees, and which does not?",
    "XOR is the witness. What does failure of this local spike-count rule *not* prove about cortex?",
    f"SGD assigns credit through a hidden layer by [back-propagation]({T['backprop']}). A local [Hebbian]({T['hebb']}) count cannot. Where in a ward protocol do you already rely on a non-local “credit” that a single synapse would not see?",
    "Leak is not depth. What does a membrane time-constant change that stacking layers changes, and why should those not be confused?",
    "Representations may still converge across systems. If two networks match on a representation test, what learning-cost facts can still differ?",
    "The 20-watt budget versus cluster-scale training is a fact about people and foundation models, not about these 200 points. What would a fair energy comparison have to measure?",
    "The LIF rule can keep updating; the MLP here is a batch on a fixed set. Which clinical skill is closer to the first, and what breaks if you only have the second?",
    "Separating two classes is a shared computational theory realized by two algorithms. State that sentence with the linear clouds in it, then with XOR.",
    f"Locality, sparsity, a body, a power budget: which of these is a hardware description, and which is already more than hardware ([embodiment]({T['embodiment']}))?",
    "If both learners succeed on the linear clouds, why is that the uninteresting panel for the paper?",
)

ITEMS["04_quadrants_biases"] = (
    f"A sufficient mechanism for the model to “show [framing]({T['framing']})” is that the designed corpus contained it. What would falsify that sufficiency claim in a larger model?",
    "Tversky and Kahneman’s subjects had not read the finding. A trainee who has, has. Which act is the language model closer to, and why does that matter at the bedside?",
    "The two frames are equivalent as prospects. Validity is not in the loss. What would have to change in the objective for *thinking rationally* to be the score?",
    "Interpolating `gain_p_sure` moves the bars. What does that tell you about whether the split lives in $\\theta$ or in $P$?",
    "A held-out wording still uses tokens from training. What would a truly new clinical phrasing test that this probe does not?",
    "Swapping the labels reverses the “bias.” If a diagnostic split reversed the same way, what would you conclude about the model — and about the people in the original study?",
    "Same architecture as notebook 01, different corpus. Which of Marr’s levels changed?",
    f"Loss aversion at the bedside has a body in it. Where, in this figure, is that body absent? ([Embodiment]({T['embodiment']}).)",
    "A judge who saw only panel C might say the model thinks like a person. What would you show them next, and which Russell–Norvig cell would that correct?",
    "Preferred tokens are not preferred observations. State that contrast in one sentence a mathematician and a clinician can both use.",
)

ITEMS["04b_four_views_of_AI"] = (
    f"A language model can be scored in every Russell–Norvig cell. Why is that not the same as occupying every cell as a person does? (The [imitation game]({T['turing']}) is only one cell.)",
    "Thinking humanly versus acting humanly: which score is a cognitive test, which is a public impression, and which does a fluent note confuse?",
    "Thinking rationally: a longer “therefore” is still $p(\\text{next}\\mid\\text{past})$. What would a proof add that a prefix cannot?",
    "Acting rationally: a plan given a goal is not a goal the system has. What extra structure did notebook 02b put in that this map does not?",
    "Thinking rationally has no extra teaching model in the paper. Why is that a finding rather than a missing notebook?",
    "Looking up a cell is not a proof that the system meets the definition. Give a clinical example of the same mistake: a score that looks like the construct.",
    "What can agreement in one cell establish at a computational level, and what can it not?",
    f"What does no cell measure? Use [embodiment]({T['embodiment']}) or [phenomenal experience]({T['consciousness']}) in the answer.",
    "If you change `FOCUS`, the cells stay. What claim should stay with them?",
    "Name one remainder for each of the four cells in a phrase a mathematician would accept.",
)

ITEMS["05_multistable_dynamics"] = (
    f"Content-addressable recall: a fragment falls into a whole ([Hopfield]({T['hopfield']})). When is a “click” in diagnosis the same computational shape, and when is vividness a spurious [attractor]({T['attractor']})?",
    "Energy never rises. Why is a falling score not insight — in this net, and in a confident wrong diagnosis?",
    f"Two wells, one page ([Necker cube]({T['necker']})). What does bistable perception require that a changing stimulus does not? What does the SDE add that the discrete net does not?",
    f"[Attention]({T['attention']}) as one modern-Hopfield step: $\\beta$ is inverse temperature. What clinical “commitment” is a large $\\beta$, and what knowledge does it not add?",
    "Training determines the landscape; inference moves in it. Map that split onto learning versus seeing in a person, and onto a weight update versus a forward pass in a net.",
    "A prompt completed two ways is a walk that left one well. Why is that not a claim that the model sees a cube?",
    "Energy, well, and perturbation are a shared dynamical vocabulary. Which of Marr’s rows is that vocabulary, and which row is still open?",
    f"The wells we can share as mathematics do not include a viability constraint. State one constraint that a Kramers crossing does not contain. ([Allostasis]({T['allostasis']}).)",
    "Corrupt a pattern until recall fails. What does capacity tell you about a memory that “feels complete”?",
    "Noise can switch the well. In a note that can be read two ways, what is the noise, and what is the page?",
)

ITEMS["05b_multistable_animation"] = (
    "Nearby starts fall into two different wells. What does that say about a slight change of history, or of prompt?",
    "The surface is a teaching function, not the Hopfield $E$ of notebook 05 and not a measured cortical energy. What comparison is still licensed?",
    "Step size: too large and you jump wells; too small and you crawl. What clinical “update size” is that a schematic of?",
    "Energy along a gradient path should not rise. If it rose, what would you have changed — the designed $U$, or the integrator?",
    "“The start determines which attractor wins.” Write that as a statement true of a brain description and of a model description.",
    f"A body that must stay in a viable region is not a ball on a designed $U(\\mathbf{{X}})$. What constraint is missing from the surface? ([Allostasis]({T['allostasis']}).)",
    "In the interactive picker you choose the start. Who chooses the start in a person, and who chooses it in a sampled completion?",
    "The animation is the same descent, drawn. What does watching it add that the static figure does not, and what does it still not measure?",
    "Three attractors on a square. Why is counting wells not a theory of thought?",
    "If two systems share a landscape sketch, have they agreed at implementation?",
)

ITEMS["05c_analog_brain_waves"] = (
    "Notebooks 05 and 05b describe settling in an energy. This notebook gates units with a stencil. Which Marr row are both on, and what algorithm have they not agreed on?",
    "Unit 1 keeps an object-A preference across contexts; unit 2 expresses B only when the stencil opens that side. What would have to change for that to count as a new synapse, and what was actually changed?",
    "Hold the two object drives fixed and swap the two stencils. What should happen to unit 2, and what clinical “same findings, different framing” is that a schematic of?",
    "Superposition of two control waves is the combination rule. Write the four gates as $\\tfrac12(w_1+w_2)$ and say what a learned map from context labels would have added that addition does not.",
    f"A traveling open window relocates activity while the drive stays put. Why is that not a [Hopfield]({T['hopfield']}) descent, and why is it not a next-token sample?",
    f"Miller et al. locate large-scale integration in organized waves rather than in broadcast. Why does that still fail to settle [phenomenal experience]({T['consciousness']})?",
    "The stencil is low-dimensional next to a synaptic matrix on the same sheet. What control problem does that dimensionality buy, and what does it not buy for a foundation model?",
    f"A shared low-dimensional trajectory is a dynamics-row agreement. Which implementation facts remain open — spikes and fields, or attention and weights? ([Embodiment]({T['embodiment']}).)",
    "If you freeze the traveling window at $t=0$, the space–time map becomes a stripe. What claim has been removed, and what claim about the drive remains?",
    "Name the remainder this toy cannot show: a body that must keep itself viable, or a measured LFP. Why is neither in $d\\odot(1-s)$?",
)

ITEMS["06_global_workspace"] = (
    f"Most computation is not broadcast ([global workspace]({T['gwt']})). Name a clinical process that is parsed but not reportable, and a model computation that is used but not in a small reportable set.",
    "Crossing a threshold is a different event from being slightly more active. Why does a continuous drive plus a step mask give a jump, and why is that not a proof of ignition in cortex?",
    "One slot versus two: what can you hold, and what drops? What working-memory fact is that a schematic of?",
    "Lesion the winner and a runner may take the workspace. What does that show about “the content,” and what does it not show about experience?",
    f"A fluent “I am thinking about $X$” can be produced by a system trained to imitate self-report. Why does that fail to settle [phenomenal experience]({T['consciousness']})?",
    "Verbalizability is a functional test you can run on a brain and on a model. Why is that theme (i), and why is it not theme (ii)?",
    "Raise the threshold on a mid-strength pulse. At what point does the same drive stay local, and what clinical “almost noticed” is that?",
    f"[IIT]({T['iit']}) is not implemented here. What extra claim would a $\\Phi$ estimate or a patient report be making that this teaching model refuses?",
    "The rust trace is a count that never enters the workspace. Is unnoticed processing a failure, or the normal case — in a person, and in a model?",
    f"If a large model has a small reportable set ([mechanistic interpretability]({T['mechint']})), have we found a workspace or only a bottleneck in the residual stream?",
)

ITEMS["07_values_probing"] = (
    f"A stable next-token answer to a [trolley]({T['trolley']}) wording is not a value. What would have to persist after you balance the corpus for you still to call it a value?",
    "Greene et al.’s 0.90 / 0.10 is a published sample, not these bars. Why must a clinician keep those two numbers distinct?",
    "A paraphrase of one outcome can flip the completion. Did the model track the outcome, or the wording? How would you tell?",
    "People who will not push, and cannot say why, are being described as judgment older than reasons. What part of that description can a model copy, and what part requires a body that can be the one?",
    f"Writing a split into a corpus on purpose is a political act ([ethics of AI]({T['ethics_ai']})). Why is it not a discovery of what the model already cared about?",
    "Same architecture, two corpora: the split is in $P$. What computational-level claim does that support, and what does it block?",
    f"Values grounded in need require [homeostasis]({T['homeostasis']}) or [allostasis]({T['allostasis']}). Where in this notebook is need absent — in the loss, in the probes, or in both?",
    "RLHF is not in the code. If a reward model were added, would the bars become values, or a second corpus?",
    "Care versus fairness could be written as counts in a designed corpus. Why would that still be a count?",
    f"The remainder the paper names is embodied prediction under viability constraints. Which panel, if any, could show that remainder — and why the answer is none? ([Embodiment]({T['embodiment']}).)",
)

QUESTIONS: dict[str, str] = {key: qs(*items) for key, items in ITEMS.items()}

NOTEBOOKS: dict[str, str] = {
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

TITLES: dict[str, str] = {
    "00_intro_timeline": "00 — Brain metaphors and AI milestones",
    "01_tiny_transformer": "01 — Tiny transformer",
    "02_predictive_coding": "02 — Predictive coding",
    "02b_active_inference_vs_rl": "02b — T-maze (active inference vs RL)",
    "03_bio_vs_artificial": "03 — Local spikes vs SGD",
    "04_quadrants_biases": "04 — Framing as next-token mass",
    "04b_four_views_of_AI": "04b — Four views of AI",
    "05_multistable_dynamics": "05 — Hopfield / Necker",
    "05b_multistable_animation": "05b — Energy landscape",
    "05c_analog_brain_waves": "05c — Analog brain waves",
    "06_global_workspace": "06 — Global workspace",
    "07_values_probing": "07 — Values / trolley",
}


def upsert_cell(nb_path: Path, source: str) -> None:
    nb = json.loads(nb_path.read_text())
    q_marker = "### Questions for reflection\n"
    r_marker = "### Further reading\n"
    body: list[dict] = []
    reading: list[dict] = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("id") == "questions001":
            continue
        if cell.get("cell_type") == "markdown" and src.startswith(q_marker):
            continue
        if cell.get("id") == "reading001" or (
            cell.get("cell_type") == "markdown" and src.startswith(r_marker)
        ):
            reading.append(cell)
            continue
        body.append(cell)
    body.append(
        {
            "cell_type": "markdown",
            "id": "questions001",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    )
    body.extend(reading)
    nb["cells"] = body
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    for key, filename in NOTEBOOKS.items():
        path = NB / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        text = QUESTIONS[key]
        n = sum(
            1
            for line in text.splitlines()
            if line.startswith("10. ") or (len(line) > 3 and line[1] == "." and line[0].isdigit())
        )
        if n != 10:
            raise SystemExit(f"{key}: expected 10 questions, found {n}")
        upsert_cell(path, text)
        print("wrote", key, n, "questions")


if __name__ == "__main__":
    main()
