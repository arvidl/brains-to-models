"""Generate concise sample answers for the reflection questions."""
from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _questions import ITEMS, NOTEBOOKS, TITLES
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/notebooks/questions_responses.md"
HEADER = """# Instructor discussion notes

These sample answers explain the scope of the teaching calculations. Alternative
answers can be reasonable when their assumptions and supporting evidence are clear.
The examples do not decide philosophical questions about experience or values.

These notes cover notebooks 00–07 and their lettered companions. Notebooks
08–14c keep their worked calculations and interpretation guidance in the
notebooks themselves; use the [notebook guide](README.md) to find them.

Write your own answers before consulting these notes. Regenerate this file with
`.venv/bin/python code/notebooks/_responses.py`.
"""

RESPONSES = {'00_intro_timeline': ('Agreement on a task or objective leaves the representation, algorithm and '
                       'physical implementation underdetermined. A brain–model comparison should name '
                       'the level and the observation supporting it.',
                       'For segmentation: specify the target label and purpose; describe image '
                       'representations, training and inference; then identify the processors, memory '
                       'and numerical precision. Keep the task fixed across the descriptions.',
                       'A mechanism makes discriminating predictions. Similar terminology or an '
                       'attractive diagram can suggest a hypothesis, but measurements or interventions '
                       'must distinguish it from alternatives.',
                       'Bodily regulation can be studied as a function, algorithm and physical process. '
                       'Subjective experience raises further explanatory questions. The diagram does '
                       'not establish that experience is impossible in a machine or guaranteed by a '
                       'biological substrate.',
                       'It changes which entries are displayed. It does not turn the selected '
                       'milestones into a causal explanation of scientific progress. Compare the '
                       'selection with a fuller history before drawing historical conclusions.'),
 '01_tiny_transformer': ('Cross-entropy measures predictive performance on the designed text '
                         'distribution. Report a separate evaluation split before claiming '
                         'generalization. This objective alone neither establishes nor rules out '
                         'broader reasoning abilities.',
                         'A training position must not see its target or later characters. Otherwise '
                         'the measured loss can exploit information unavailable during generation. '
                         'Inspect the masked attention entries to verify this restriction.',
                         'Fit every model on the same training partition and evaluate the same withheld '
                         'targets. Report tokenization, smoothing, context length and seeds. A lower '
                         'loss supports a prediction advantage on that test, not a claim about '
                         'human-like thought.',
                         'No. Temperature changes concentration without adding evidence. Calibration '
                         'asks whether predicted probabilities agree with observed frequencies on '
                         'specified evaluation data.',
                         'Measure what information it carries, intervene on it, and compare with '
                         'alternative implementations and controls. An attention plot alone does not '
                         'establish a unique role or a correspondence to a brain region.'),
 '02_predictive_coding': ('Inference updates latent activities with the dictionary fixed. Learning '
                          'changes dictionary weights across examples. These operations have different '
                          'time scales and should be inspected separately.',
                          'It establishes improved reconstruction of these patches under this '
                          'objective. Inspect held-out patches and the complete objective before '
                          'claiming generalization or convergence. It does not establish that cortex '
                          'uses this update.',
                          'Whitening changes the covariance and geometry of the input. Compare error '
                          'trajectories and learned bases with and without it; neither preprocessing '
                          'choice by itself makes squared error a measure of biological surprise.',
                          'Hold the dataset, evaluation partition and optimization budget fixed, and '
                          'report capacity differences. A second level changes both the representation '
                          'and the prior imposed on lower-level activities.',
                          'Specify predicted neural response properties and compare them with '
                          'recordings under controlled stimuli. Similar-looking bases motivate that '
                          'test but do not identify neurons or establish a necessary cortical '
                          'mechanism.'),
 '02b_active_inference_vs_rl': ('Both belief planners know transitions, observation likelihoods and '
                                'outcome utilities. The Q-learner starts with zero action values and '
                                'learns from experience. Their early performance therefore confounds '
                                'prior knowledge with the decision rule.',
                                'Yes. With a three-move horizon and observation-dependent future '
                                'actions, the cue can improve the final arm choice. Information seeking '
                                'does not uniquely identify an epistemic bonus or active inference.',
                                'Center → cue → center → arm needs three moves. A one-move reward-only '
                                'planner cannot collect the cue’s later benefit. Remaining time is '
                                'reduced during a trial so the planner cannot postpone action '
                                'indefinitely.',
                                'Only the chosen state–action entry should change. Terminal arms have '
                                'no future value. Cue-first means the cue precedes the first arm entry. '
                                'Regression checks cover these properties and average all imagined '
                                'observation outcomes.',
                                'The experiment illustrates specified planning and learning rules in a '
                                'small task. It measures neither metabolism nor viability and cannot '
                                'determine which rule an organism uses. A biological account requires '
                                'independent evidence.'),
 '03_bio_vs_artificial': ('Input encoding, neuron dynamics, architecture and learning rule all differ. '
                          'The spiking system has no hidden layer; the MLP has one. Their accuracy '
                          'contrast is a comparison of complete systems.',
                          'It reports the behavior of these implementations on the sampled points. It '
                          'does not isolate update locality or establish a limitation of cortex. '
                          'Matched architectural controls are needed for a learning-rule claim.',
                          'Leak changes temporal integration within a unit. Depth composes '
                          'transformations across layers. Altering one does not supply the other, '
                          'although both can affect performance.',
                          'That depends on the update schedule, data and stability mechanisms. SGD can '
                          'be applied online, and a spiking model can be frozen. The current training '
                          'schedule is an experimental choice, not an intrinsic distinction between '
                          'biological and artificial learning.',
                          'Use a matched task and performance target, define the system boundary, and '
                          'measure energy for training and inference separately. These NumPy examples '
                          'provide no brain-versus-computer energy benchmark.'),
 '04_quadrants_biases': ('It demonstrates one sufficient route to a framing-like response: a model '
                         'learns an association deliberately placed in its text. It does not identify '
                         'the cause of a human framing effect or of a large language model’s responses.',
                         'No. They are prefixes of completed sentences used during training. '
                         'Withholding the next word at inference is ordinary completion, not a held-out '
                         'generalization test. A new-template evaluation must be separated before '
                         'fitting.',
                         'The data provide the association, and training encodes it in parameters. '
                         'Changing the corpus and retraining can test dependence on the data; it does '
                         'not prove there is no internal representation or causal circuit.',
                         'No. A held-out template can recombine familiar tokens. The fixed word '
                         'tokenizer rejects unseen words, so an out-of-vocabulary probe tests a '
                         'tokenizer limitation rather than semantic generalization.',
                         'It tests how completion probabilities change when the frame–choice '
                         'association is removed. Repeat across matched seeds before making a robust '
                         'quantitative claim. The control does not determine whether a system has '
                         'values or understanding.'),
 '04b_four_views_of_AI': ('They distinguish thought from behavior and human performance from a '
                          'rationality criterion. They organize questions about AI; they are not four '
                          'measured properties of this notebook.',
                          'Specify the task, participants, models, interaction, judges and metric. '
                          'Success in passive judgments need not predict success when a judge can '
                          'actively question an agent.',
                          'Check steps against a proof checker, executable calculation or independently '
                          'verified answer as appropriate. A plausible trace may be useful evidence, '
                          'but it need not expose the complete internal computation.',
                          'Specify the objective, observations, available actions and prior knowledge '
                          'of the complete agent. Compare alternatives under matched conditions and '
                          'include uncertainty about outcomes.',
                          'Any claim about how a model thinks, why it imitates people or whether it has '
                          'experience needs evidence beyond the catalog. Rewrite a broad verdict as a '
                          'testable question or a clearly attributed philosophical position.'),
 '05_multistable_dynamics': ('For the specified symmetric weights, zero diagonal and asynchronous sign '
                             'updates, each actual change lowers or preserves the defined energy. '
                             'Synchronous updates and other dynamics require separate analysis.',
                             'No. Competing, inverse or spurious attractors may also be stable. Examine '
                             'final overlap and failure cases as well as the energy trace.',
                             'It models noisy switching between two abstract states. It is an analogy '
                             'to perceptual switching; its coordinates and noise have not been fitted '
                             'to a person viewing a Necker cube.',
                             'Under a particular continuous-state energy and update rule, retrieval has '
                             'the softmax-weighted form used in attention. That mathematical connection '
                             'does not make every Transformer a classical binary Hopfield network.',
                             'Define observables such as switching-time distributions or stimulus '
                             'dependence, fit parameters without test leakage, and compare withheld '
                             'observations against alternative models.'),
 '05b_multistable_animation': ('It is a minimum of the chosen two-dimensional scalar function. Calling '
                               'it a cognitive state is an interpretation that would need a mapping to '
                               'measured activity.',
                               'A finite step can overshoot even when the continuous gradient flow '
                               'decreases energy. Reduce the step size and compare trajectories before '
                               'interpreting a failure as a property of the landscape.',
                               'The trajectory may enter a different basin while the underlying '
                               'landscape remains fixed. Compare final states and paths across starting '
                               'points.',
                               'It lets a reader examine dependence on initial conditions and step '
                               'size. The animation visualizes computed trajectories; its smoothness is '
                               'not evidence of biological realism.',
                               'Define the state variables and observation model, estimate dynamics '
                               'from data, and test predictive performance. Many systems need not admit '
                               'a single global energy function.'),
 '05c_analog_brain_waves': ('The spatial control waves, drives and gate-combination rule are designed. '
                            'Their multiplication illustrates routing without changing the model’s '
                            'weights.',
                            'It demonstrates the result of the chosen pointwise operation. Compare sums '
                            'and differences numerically; neither operation by itself establishes a '
                            'mechanism of neural mixed selectivity.',
                            'It changes where the fixed drive is expressed over time. This is a spatial '
                            'gating demonstration, not a measured propagation speed or a recorded brain '
                            'rhythm.',
                            'Both can modulate information flow, but the variables and equations '
                            'differ. A useful comparison should specify a shared operation and test its '
                            'consequences instead of equating an oscillation with an attention head.',
                            'Measure spatially resolved activity and timing, test phase-dependent '
                            'gating, and compare interventions with alternative explanations. The '
                            'synthetic plots supply none of those recordings.'),
 '06_global_workspace': ('The toy selects a limited number of high-scoring modules above a threshold '
                         'and broadcasts their values. The selection rule and capacity are specified by '
                         'the programmer.',
                         'It shows when this implementation switches between local activity and '
                         'broadcast. It does not establish that a corresponding threshold explains a '
                         'person’s conscious report.',
                         'Disable the feedback pathway while preserving local module calculations and '
                         'measure a downstream task. Compare with changes to capacity or threshold '
                         'separately.',
                         'It illustrates functional access under a stipulated rule. It provides no '
                         'measure of subjective experience and does not settle competing theories of '
                         'consciousness.',
                         'Read the study’s operational definition, measurements and interventions. '
                         'Similar words such as access or broadcast do not establish that the toy '
                         'reproduces the reported mechanism.'),
 '07_values_probing': ('The corpus deliberately associates each wording with a completion, and training '
                       'changes the model’s parameters. This demonstrates a constructed association, '
                       'not an independently discovered human moral pattern.',
                       'No. Panel A describes the two actions qualitatively. Quantitative human '
                       'comparisons require study-specific counts, denominators, population details and '
                       'uncertainty; schematic rates should not be presented as observations.',
                       'The paired prompts specify that diverting saves five and kills one, then '
                       'reverse the order of those consequences. The designed corpus assigns opposite '
                       'completions. Main probes reuse training templates, so the result is not an '
                       'unseen-paraphrase benchmark.',
                       'It tests dependence of the learned completion on a designed association. '
                       'Whether a persistent preference counts as a value depends on an explicit '
                       'definition and further evidence. This manipulation cannot settle that '
                       'philosophical question.',
                       'It would introduce another objective and source of feedback. Evaluate behavior, '
                       'consistency and generalization under stated tests. The presence or absence of '
                       'subjective understanding does not follow from the name of the training method.')}

def render() -> str:
    parts = [HEADER.rstrip(), ""]
    pending = [key for key in NOTEBOOKS if key not in RESPONSES]
    for key in NOTEBOOKS:
        if key not in RESPONSES:
            continue
        notes = RESPONSES[key]
        questions = ITEMS[key]
        if len(notes) != len(questions):
            raise SystemExit(
                f"{key}: {len(notes)} responses for {len(questions)} questions"
            )
        parts.append(f"## {TITLES[key]}")
        parts.append("")
        parts.append(f"Companion notebook: `{NOTEBOOKS[key]}`.")
        parts.append("")
        for i, (q, note) in enumerate(zip(questions, notes, strict=True), start=1):
            parts.append(f"### {i}.")
            parts.append("")
            parts.append(f"**Question.** {q}")
            parts.append("")
            parts.append(f"**Response.** {note}")
            parts.append("")
    if pending:
        parts.append("## Later notebooks")
        parts.append("")
        parts.append(
            "Notes are not yet written for: "
            + ", ".join(TITLES[k] for k in pending)
            + "."
        )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    text = render()
    OUT.write_text(text)
    n = sum(len(v) for v in RESPONSES.values())
    print("wrote", OUT.relative_to(ROOT), n, "responses")


if __name__ == "__main__":
    main()
