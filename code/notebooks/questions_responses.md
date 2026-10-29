# Instructor discussion notes

These sample answers explain the scope of the teaching calculations. Alternative
answers can be reasonable when their assumptions and supporting evidence are clear.
The examples do not decide philosophical questions about experience or values.

These notes cover notebooks 00–07 and their lettered companions. Notebooks
08–14c keep their worked calculations and interpretation guidance in the
notebooks themselves; use the [notebook guide](README.md) to find them.

Write your own answers before consulting these notes. Regenerate this file with
`.venv/bin/python code/notebooks/_responses.py`.

## 00 — Brain metaphors and AI milestones

Companion notebook: `00_intro_timeline.ipynb`.

### 1.

**Question.** What does agreement at one of Marr’s levels establish, and what remains open?

**Response.** Agreement on a task or objective leaves the representation, algorithm and physical implementation underdetermined. A brain–model comparison should name the level and the observation supporting it.

### 2.

**Question.** Choose one task and describe it at all three levels.

**Response.** For segmentation: specify the target label and purpose; describe image representations, training and inference; then identify the processors, memory and numerical precision. Keep the task fixed across the descriptions.

### 3.

**Question.** What evidence would distinguish a useful brain metaphor from a proposed mechanism?

**Response.** A mechanism makes discriminating predictions. Similar terminology or an attractive diagram can suggest a hypothesis, but measurements or interventions must distinguish it from alternatives.

### 4.

**Question.** How do bodily regulation and experience relate to Marr’s framework?

**Response.** Bodily regulation can be studied as a function, algorithm and physical process. Subjective experience raises further explanatory questions. The diagram does not establish that experience is impossible in a machine or guaranteed by a biological substrate.

### 5.

**Question.** What does filtering the timeline change?

**Response.** It changes which entries are displayed. It does not turn the selected milestones into a causal explanation of scientific progress. Compare the selection with a fuller history before drawing historical conclusions.

## 01 — Tiny transformer

Companion notebook: `01_tiny_transformer.ipynb`.

### 1.

**Question.** What does next-character prediction measure in this experiment?

**Response.** Cross-entropy measures predictive performance on the designed text distribution. Report a separate evaluation split before claiming generalization. This objective alone neither establishes nor rules out broader reasoning abilities.

### 2.

**Question.** Why is the causal mask necessary for left-to-right prediction?

**Response.** A training position must not see its target or later characters. Otherwise the measured loss can exploit information unavailable during generation. Inspect the masked attention entries to verify this restriction.

### 3.

**Question.** How would you compare this Transformer fairly with a unigram or bigram model?

**Response.** Fit every model on the same training partition and evaluate the same withheld targets. Report tokenization, smoothing, context length and seeds. A lower loss supports a prediction advantage on that test, not a claim about human-like thought.

### 4.

**Question.** Does a concentrated output distribution imply a calibrated belief?

**Response.** No. Temperature changes concentration without adding evidence. Calibration asks whether predicted probabilities agree with observed frequencies on specified evaluation data.

### 5.

**Question.** What would be needed to interpret a head as a functional component?

**Response.** Measure what information it carries, intervene on it, and compare with alternative implementations and controls. An attention plot alone does not establish a unique role or a correspondence to a brain region.

## 02 — Predictive coding

Companion notebook: `02_predictive_coding.ipynb`.

### 1.

**Question.** What differs between inference and learning in this model?

**Response.** Inference updates latent activities with the dictionary fixed. Learning changes dictionary weights across examples. These operations have different time scales and should be inspected separately.

### 2.

**Question.** What does decreasing reconstruction error establish?

**Response.** It establishes improved reconstruction of these patches under this objective. Inspect held-out patches and the complete objective before claiming generalization or convergence. It does not establish that cortex uses this update.

### 3.

**Question.** How does whitening affect the experiment?

**Response.** Whitening changes the covariance and geometry of the input. Compare error trajectories and learned bases with and without it; neither preprocessing choice by itself makes squared error a measure of biological surprise.

### 4.

**Question.** What is a fair comparison of one-level and two-level models?

**Response.** Hold the dataset, evaluation partition and optimization budget fixed, and report capacity differences. A second level changes both the representation and the prior imposed on lower-level activities.

### 5.

**Question.** How could a cortical interpretation be tested?

**Response.** Specify predicted neural response properties and compare them with recordings under controlled stimuli. Similar-looking bases motivate that test but do not identify neurons or establish a necessary cortical mechanism.

## 02b — T-maze (active inference vs RL)

Companion notebook: `02b_active_inference_vs_rl.ipynb`.

### 1.

**Question.** What does each agent know before the first trial?

**Response.** Both belief planners know transitions, observation likelihoods and outcome utilities. The Q-learner starts with zero action values and learns from experience. Their early performance therefore confounds prior knowledge with the decision rule.

### 2.

**Question.** Can a reward-only planner value visiting the cue?

**Response.** Yes. With a three-move horizon and observation-dependent future actions, the cue can improve the final arm choice. Information seeking does not uniquely identify an epistemic bonus or active inference.

### 3.

**Question.** What changes when the planning horizon is shortened?

**Response.** Center → cue → center → arm needs three moves. A one-move reward-only planner cannot collect the cue’s later benefit. Remaining time is reduced during a trial so the planner cannot postpone action indefinitely.

### 4.

**Question.** How should the Q-update and cue-first metric be checked?

**Response.** Only the chosen state–action entry should change. Terminal arms have no future value. Cue-first means the cue precedes the first arm entry. Regression checks cover these properties and average all imagined observation outcomes.

### 5.

**Question.** What conclusions about organisms follow from this comparison?

**Response.** The experiment illustrates specified planning and learning rules in a small task. It measures neither metabolism nor viability and cannot determine which rule an organism uses. A biological account requires independent evidence.

## 03 — Local spikes vs SGD

Companion notebook: `03_bio_vs_artificial.ipynb`.

### 1.

**Question.** Which factors differ between the two learners?

**Response.** Input encoding, neuron dynamics, architecture and learning rule all differ. The spiking system has no hidden layer; the MLP has one. Their accuracy contrast is a comparison of complete systems.

### 2.

**Question.** What does the XOR result establish?

**Response.** It reports the behavior of these implementations on the sampled points. It does not isolate update locality or establish a limitation of cortex. Matched architectural controls are needed for a learning-rule claim.

### 3.

**Question.** How does a membrane leak parameter differ from depth?

**Response.** Leak changes temporal integration within a unit. Depth composes transformations across layers. Altering one does not supply the other, although both can affect performance.

### 4.

**Question.** Could either model learn continually?

**Response.** That depends on the update schedule, data and stability mechanisms. SGD can be applied online, and a spiking model can be frozen. The current training schedule is an experimental choice, not an intrinsic distinction between biological and artificial learning.

### 5.

**Question.** What would a meaningful energy comparison require?

**Response.** Use a matched task and performance target, define the system boundary, and measure energy for training and inference separately. These NumPy examples provide no brain-versus-computer energy benchmark.

## 04 — Framing as next-token mass

Companion notebook: `04_quadrants_biases.ipynb`.

### 1.

**Question.** What does training on the designed corpus demonstrate?

**Response.** It demonstrates one sufficient route to a framing-like response: a model learns an association deliberately placed in its text. It does not identify the cause of a human framing effect or of a large language model’s responses.

### 2.

**Question.** Are the main probes held-out sentence templates?

**Response.** No. They are prefixes of completed sentences used during training. Withholding the next word at inference is ordinary completion, not a held-out generalization test. A new-template evaluation must be separated before fitting.

### 3.

**Question.** Is the learned association in the data or in the parameters?

**Response.** The data provide the association, and training encodes it in parameters. Changing the corpus and retraining can test dependence on the data; it does not prove there is no internal representation or causal circuit.

### 4.

**Question.** Does a new wording require new vocabulary?

**Response.** No. A held-out template can recombine familiar tokens. The fixed word tokenizer rejects unseen words, so an out-of-vocabulary probe tests a tokenizer limitation rather than semantic generalization.

### 5.

**Question.** What does the balanced control establish?

**Response.** It tests how completion probabilities change when the frame–choice association is removed. Repeat across matched seeds before making a robust quantitative claim. The control does not determine whether a system has values or understanding.

## 04b — Four views of AI

Companion notebook: `04b_four_views_of_AI.ipynb`.

### 1.

**Question.** What do the two axes of the Russell–Norvig grid distinguish?

**Response.** They distinguish thought from behavior and human performance from a rationality criterion. They organize questions about AI; they are not four measured properties of this notebook.

### 2.

**Question.** What should be specified in an imitation test?

**Response.** Specify the task, participants, models, interaction, judges and metric. Success in passive judgments need not predict success when a judge can actively question an agent.

### 3.

**Question.** How could a generated reasoning trace be evaluated?

**Response.** Check steps against a proof checker, executable calculation or independently verified answer as appropriate. A plausible trace may be useful evidence, but it need not expose the complete internal computation.

### 4.

**Question.** What belongs in an evaluation of rational action?

**Response.** Specify the objective, observations, available actions and prior knowledge of the complete agent. Compare alternatives under matched conditions and include uncertainty about outcomes.

### 5.

**Question.** Which claims in the grid are hypotheses rather than results?

**Response.** Any claim about how a model thinks, why it imitates people or whether it has experience needs evidence beyond the catalog. Rewrite a broad verdict as a testable question or a clearly attributed philosophical position.

## 05 — Hopfield / Necker

Companion notebook: `05_multistable_dynamics.ipynb`.

### 1.

**Question.** When is Hopfield energy guaranteed not to increase?

**Response.** For the specified symmetric weights, zero diagonal and asynchronous sign updates, each actual change lowers or preserves the defined energy. Synchronous updates and other dynamics require separate analysis.

### 2.

**Question.** Does energy descent guarantee the intended memory?

**Response.** No. Competing, inverse or spurious attractors may also be stable. Examine final overlap and failure cases as well as the energy trace.

### 3.

**Question.** What does the double-well example model?

**Response.** It models noisy switching between two abstract states. It is an analogy to perceptual switching; its coordinates and noise have not been fitted to a person viewing a Necker cube.

### 4.

**Question.** In what sense is modern Hopfield retrieval related to attention?

**Response.** Under a particular continuous-state energy and update rule, retrieval has the softmax-weighted form used in attention. That mathematical connection does not make every Transformer a classical binary Hopfield network.

### 5.

**Question.** What would turn the analogy into a biological test?

**Response.** Define observables such as switching-time distributions or stimulus dependence, fit parameters without test leakage, and compare withheld observations against alternative models.

## 05b — Energy landscape

Companion notebook: `05b_multistable_animation.ipynb`.

### 1.

**Question.** What does a valley in this drawing represent?

**Response.** It is a minimum of the chosen two-dimensional scalar function. Calling it a cognitive state is an interpretation that would need a mapping to measured activity.

### 2.

**Question.** Why can a numerical descent step increase the energy?

**Response.** A finite step can overshoot even when the continuous gradient flow decreases energy. Reduce the step size and compare trajectories before interpreting a failure as a property of the landscape.

### 3.

**Question.** What changes when the initial point changes?

**Response.** The trajectory may enter a different basin while the underlying landscape remains fixed. Compare final states and paths across starting points.

### 4.

**Question.** What does the interactive view add?

**Response.** It lets a reader examine dependence on initial conditions and step size. The animation visualizes computed trajectories; its smoothness is not evidence of biological realism.

### 5.

**Question.** What would be needed to infer a landscape from data?

**Response.** Define the state variables and observation model, estimate dynamics from data, and test predictive performance. Many systems need not admit a single global energy function.

## 05c — Analog brain waves

Companion notebook: `05c_analog_brain_waves.ipynb`.

### 1.

**Question.** What is specified rather than learned in the stencil example?

**Response.** The spatial control waves, drives and gate-combination rule are designed. Their multiplication illustrates routing without changing the model’s weights.

### 2.

**Question.** What does combining two gates demonstrate?

**Response.** It demonstrates the result of the chosen pointwise operation. Compare sums and differences numerically; neither operation by itself establishes a mechanism of neural mixed selectivity.

### 3.

**Question.** What does moving the open window change?

**Response.** It changes where the fixed drive is expressed over time. This is a spatial gating demonstration, not a measured propagation speed or a recorded brain rhythm.

### 4.

**Question.** How does this relate to Transformer attention?

**Response.** Both can modulate information flow, but the variables and equations differ. A useful comparison should specify a shared operation and test its consequences instead of equating an oscillation with an attention head.

### 5.

**Question.** What biological evidence would support the analogy?

**Response.** Measure spatially resolved activity and timing, test phase-dependent gating, and compare interventions with alternative explanations. The synthetic plots supply none of those recordings.

## 06 — Global workspace

Companion notebook: `06_global_workspace.ipynb`.

### 1.

**Question.** Which workspace operations are implemented?

**Response.** The toy selects a limited number of high-scoring modules above a threshold and broadcasts their values. The selection rule and capacity are specified by the programmer.

### 2.

**Question.** What does the threshold comparison establish?

**Response.** It shows when this implementation switches between local activity and broadcast. It does not establish that a corresponding threshold explains a person’s conscious report.

### 3.

**Question.** What intervention would test the role of broadcast?

**Response.** Disable the feedback pathway while preserving local module calculations and measure a downstream task. Compare with changes to capacity or threshold separately.

### 4.

**Question.** What can this model say about subjective experience?

**Response.** It illustrates functional access under a stipulated rule. It provides no measure of subjective experience and does not settle competing theories of consciousness.

### 5.

**Question.** How should language-model workspace findings be compared with the toy?

**Response.** Read the study’s operational definition, measurements and interventions. Similar words such as access or broadcast do not establish that the toy reproduces the reported mechanism.

## 07 — Values / trolley

Companion notebook: `07_values_probing.ipynb`.

### 1.

**Question.** Where do the switch and footbridge completion patterns come from?

**Response.** The corpus deliberately associates each wording with a completion, and training changes the model’s parameters. This demonstrates a constructed association, not an independently discovered human moral pattern.

### 2.

**Question.** Does this figure contain measured human choice rates?

**Response.** No. Panel A describes the two actions qualitatively. Quantitative human comparisons require study-specific counts, denominators, population details and uncertainty; schematic rates should not be presented as observations.

### 3.

**Question.** How are equivalent consequences tested?

**Response.** The paired prompts specify that diverting saves five and kills one, then reverse the order of those consequences. The designed corpus assigns opposite completions. Main probes reuse training templates, so the result is not an unseen-paraphrase benchmark.

### 4.

**Question.** What does balancing the corpus tell us about values?

**Response.** It tests dependence of the learned completion on a designed association. Whether a persistent preference counts as a value depends on an explicit definition and further evidence. This manipulation cannot settle that philosophical question.

### 5.

**Question.** Would adding preference training establish moral understanding?

**Response.** It would introduce another objective and source of feedback. Evaluate behavior, consistency and generalization under stated tests. The presence or absence of subjective understanding does not follow from the name of the training method.
