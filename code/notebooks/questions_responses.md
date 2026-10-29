# Notes toward the questions for reflection

Not a quiz key and not the official answer. Each note is a fuller
response the paper would accept, not the only competent one. A
different wording that names the same Marr row (or theme), gives a test
both a clinician and a mathematically inclined reader could run, and
names the same limit is also a response.

Write your own first, in the notebook. The two themes are defined in
each notebook’s Further reading cell. The demonstrations are teaching
models: they are not recordings from cortex, not evaluations of a
large language model, and not evidence that a next-token probability
is a value.

Regenerate this file with:

```bash
.venv/bin/python code/notebooks/_responses.py
```

Notebooks 00–07 (including b) are written.

## 00 — Brain metaphors and AI milestones

Companion notebook: `00_intro_timeline.ipynb`.

### 1.

**Question.** Marr’s three levels apply to a brain and to a machine. If two systems share a computational theory (what is computed, and why), what has that established, and what has it left open?

**Response.** A shared computational theory is agreement on Marr’s top row: the same input–output map $f$ and a reason it is the right map — for a transformer and a cortex, “estimate $p(\text{next}\mid\text{past})$ because the signal is a sequence to be continued.” It leaves everything below open: the representation and algorithm (attention versus precision-weighted error versus an attractor update) and the physical substrate (matrix multiplies on silicon versus spikes and synapses). Because the levels are only loosely coupled, agreeing on the objective fixes neither the mechanism nor the hardware, and it imports no body, viability, or experience. Clinical test: two clinicians who reach the same diagnosis have agreed on the “what,” not on the reasoning route or the wiring. Formal test: many algorithms compute one $f$ (normal equations versus gradient descent), and many substrates run one algorithm. The limit is that a top-row match is a candidate requirement, not a demonstration of shared mechanism.

### 2.

**Question.** The paper renames Marr’s rows as function, dynamics, and architecture. Take one later section you already know — a transformer, a framing bias, a trolley wording — and state which row the comparison is actually being made on.

**Response.** Name the row explicitly. A transformer-versus-brain comparison is almost always on the computational (function) row: both are described as minimising $-\log p(\text{next}\mid\text{past})$. A framing bias (notebook 04) or a trolley split (notebook 07) is still that row — a claim about what is predicted under a designed corpus, i.e. about $P$, not about the update or the substrate. To move to the dynamics row you must claim the *update* is shared (attention as one modern-Hopfield step, notebook 05; precision-weighted error, notebook 02); to reach the architecture row you compare synapses with weights, spikes with activations. Test: ask whether the evidence is an objective (function), an update rule (dynamics), or a physical part (architecture); most of the paper’s comparisons are deliberately on the first. The limit: a function-row match is the weakest of the three and the easiest to over-read.

### 3.

**Question.** Loose coupling means the same representation can be updated by different algorithms. Give one clinical example (two routes to the same diagnosis) and one formal example (two algorithms for the same map).

**Response.** Loose coupling says one representation can be driven by different algorithms, so agreement upstream does not fix the procedure. Clinical: one diagnosis can be reached by an explicit criterion checklist or by fast pattern recognition from remembered cases — two algorithms, one report, which may even disagree on borderline patients while sharing the label. Formal: the least-squares fit of $f(x)=Wx$ can be computed by the normal equations $W=(X^\top X)^{-1}X^\top y$ or by iterating $W\leftarrow W-\eta\nabla\mathcal{L}$; the map is identical, the algorithm is not, and they differ in cost and numerical behaviour. Test: hold the output fixed and vary the procedure — if the output is unchanged, the algorithm was underdetermined. The limit: shared output is not shared mechanism, the point the whole paper leans on.

### 4.

**Question.** The tan band is not a fourth Marr level. Explain why [viability](https://en.wikipedia.org/wiki/Allostasis), [phenomenal experience](https://plato.stanford.edu/entries/consciousness/), and values grounded in need are not properties of a hardware description.

**Response.** A hardware description says how a representation and an algorithm are physically realised — which components carry which variables. Viability is not such a fact: it is a constraint that a body keep quantities (oxygenation, glucose, temperature) inside survivable bounds, a property of an organism in a world, not of a wiring diagram. Phenomenal experience — what it is like — is likewise not a component or a connection, and values grounded in need are not entries in a weight matrix. So completing the bottom row for silicon or cortex adds no viability, no experience, no need-based value; those are orthogonal to the three-level description, which is why the paper draws them as a band leaving the stack rather than as a fourth row. Test: specify a complete implementational account and ask whether it entails any of the three — it does not. The limit: the band marks an exit from the Marr picture, not a further equation within it.

### 5.

**Question.** Theme (ii) starts from synapses, spikes, energy, and a body, then leaves that row. Suppose you had a complete implementational account of both a transformer and a cortex: what would still be missing?

**Response.** With both bottom rows complete you would know every weight and multiply, every synapse and spike, and still not know three things. First, whether either system *must* stay viable — whether there are variables it fails if it cannot keep in range. Second, whether there is something it is like to be the system. Third, whether any preference is grounded in need rather than in training data. None is a further implementational fact you forgot to measure; none lies on the axis the three levels span. Test: imagine the fullest circuit diagram of each and ask what it settles about viability, experience, or need — nothing. The limit is exactly theme (ii): it begins at synapses, spikes, energy, and a body, then leaves that row.

### 6.

**Question.** Before McCulloch–Pitts the comparison runs in one direction: the newest machine explains the brain. What observation would show that a metaphor had become bidirectional, and which Marr row would that observation occupy?

**Response.** The comparison is unidirectional while the only move is “narrate the brain as the latest machine.” It becomes bidirectional when a *success or failure of the machine constrains a claim about us* — when the artefact is evidence, not just metaphor. That evidence usually lives on the computational (function) row: a shared objective such as language or prediction, where a model’s behaviour bears on hypotheses about human processing. It does not by itself reach the architecture row — sharing an objective is not sharing synapses. On the paper’s timeline the turn comes when machines produce language at a scale that raises the reverse question (“has the machine become like us?”), roughly at foundation models. The limit: bidirectionality on the function row is real but weak, and is not licence to equate mechanisms or substrates.

### 7.

**Question.** A [foundation model](https://en.wikipedia.org/wiki/Foundation_model) produces language that invites the reverse question. If fluency at the interface is the evidence, which Russell–Norvig cell are you scoring, and what does that score fail to capture?

**Response.** Fluency judged by conversation scores the *acting-humanly* cell — the imitation game — and nothing else. It does not score thinking humanly (human processes and human errors), thinking rationally (valid inference), or acting rationally (a goal the system actually has). It also measures none of theme (ii): embodiment, phenomenal experience, or values grounded in need are invisible at the interface. Test: pass the same system through notebook 04’s error-reproduction probe (thinking humanly) or 02b’s goal structure (acting rationally) and watch the verdicts diverge from the fluency verdict. The limit: a Turing-style pass has become easy precisely because the game is now text, so it is the least informative of the four cells for the paper’s question.

### 8.

**Question.** The timeline is a labelled set of points, not a causal model of progress. Which claim survives when several AI winters, many architectures, and almost all of neuroscience are omitted?

**Response.** The timeline is not a causal model of progress, so no claim about *why* AI advanced survives; what survives is a single directional fact. Before the mid-twentieth century the arrow runs one way — the newest machine explains the brain. After McCulloch–Pitts the same devices appear in both the machine history and the brain-model history. After foundation models the machine’s *failures* start to count as evidence about us. Omitting AI winters, most architectures, and nearly all of neuroscience changes the richness of the story but not that direction. Test: delete any subset of points and check whether the arrow reverses — it does not. The limit: direction is all a labelled point-set can support; it cannot support a mechanism of progress.

### 9.

**Question.** Changing `FOCUS` or the year window filters the catalogue. Why does that leave both schematics unchanged, and what limit does that separation protect?

**Response.** `FOCUS` and the year window are catalogue filters: they select which entries print and change nothing about the two drawn schematics (the Marr stack and the band). That separation protects a methodological limit: what agreement on a Marr row can establish, and what a hardware description cannot contain, are not functions of which examples you display. Were the schematic to change with the filter, the teaching claim would be an artefact of selection. Test: sweep `FOCUS` and confirm the Marr figure and the band are invariant. The limit protected is that the paper’s structural claims are about levels, not about the particular timeline entries chosen to illustrate them.

### 10.

**Question.** A shared objective — prediction — does not give a machine a body. State that limit twice: once as a Marr fact (agreement on a single row) and once as a human fact ([embodiment](https://plato.stanford.edu/entries/embodied-cognition/)).

**Response.** As a Marr fact: agreeing that a brain and a model both *predict* is agreement on one row (the computational objective); it delivers neither the algorithm nor the substrate of cortex. As a human fact: the prediction that keeps a body viable — anticipating the sensory consequences of action so the organism can act to stay alive — is not next-token prediction over a corpus. Clinical test: a model can predict the next word on a chart with no stake in the patient; a clinician’s prediction is tied to a body that can be harmed. Formal test: both may minimise a surprisal-like loss, yet only one minimises it *for* an embodied agent under viability constraints. The limit: sharing the objective confers embodiment on neither side — where theme (ii) begins.

## 01 — Tiny transformer

Companion notebook: `01_tiny_transformer.ipynb`.

### 1.

**Question.** If a radiologist and this model both complete “the patient was started on …”, what would have to be the same for that to count as the same [computational theory](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) (what is computed, and why)? What would still differ at implementation?

**Response.** To count as the same computational theory the two must compute the same map for the same reason: $p(\text{next}\mid\text{past})$ as the objective, because a string is being continued — not merely produce the same next word by coincidence. Grant that, and everything at implementation still differs: the model is embeddings and matrix multiplies over a 27-symbol vocabulary; the radiologist is a person with synapses, a body, and a patient who can be harmed by the continuation. Formal test: check that both are trained toward and scored by the same conditional distribution, not just that one sample agreed. Clinical test: ask what a wrong completion costs — a scalar loss for the model, a body for the patient. The limit: a shared completion is a top-row match at best and says nothing about the substrate or the stakes.

### 2.

**Question.** The loss is $-\log p(\text{next}\mid\text{past})$. Name a clinical prediction that is scored the same way. What cost does the clinical miss pay that this loss does not?

**Response.** Any left-to-right clinical forecast can be scored by the same proper scoring rule: the next lab value, the next vital, the next charted event, each as $-\log p(\text{next}\mid\text{past})$, low when the realised outcome was assigned high probability. The mathematics is identical — a cross-entropy minimised only by the true conditional. What differs is the consequence: a miss on the clinical forecast can change a body (a missed deterioration), whereas a miss on this loss changes only a scalar the optimiser reduces next step. Test: put the two side by side and ask where the cost of error lives — in the objective, or outside it in an organism. The limit: the clinical cost is never in the loss; the objective is blind to viability.

### 3.

**Question.** [Attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) is a weighted look-back over the prompt. When you glance back in a note, what extra constraint does that glance have that a causal head does not?

**Response.** A causal head computes a data-dependent convex combination of past keys, $\mathrm{softmax}(QK^\top/\sqrt{d}+M)V$, limited only by the mask and the trained $Q,K$ — nothing outside the token stream constrains it. A clinician glancing back in a note is constrained by things absent from that equation: a question about *this* patient, real time, the risk of acting on stale information, and a reader who can be interrupted and held responsible. Formal test: perturb the prompt and the head’s weights move continuously with the keys; the clinician’s glance is gated by relevance to a goal, not by key-similarity alone. Clinical test: ask what would make the glance *stop* — an answer sufficient for the decision — which has no analogue in the softmax. The limit: attention is a mechanism, not a rationale; a bright cell is not a reason.

### 4.

**Question.** Four heads specialise in routing. What would it take to treat a head as a neurological “module,” and why is that a category error on this teaching model?

**Response.** To treat a head as a “module” you would need independent evidence that it computes a *named* function mapping onto a known circuit, replicated at a scale that is not four heads over 27 characters. Here a head is a routing map that specialises on a designed five-frame corpus; its structure is an artefact of that corpus and seed 7. Calling it a neurological module confuses a teaching split with a part of cortex — a Marr category error, importing an architecture-row claim from a function-row toy. Test: retrain with a different seed or corpus and watch the “module” reassign; a genuine module would not be so contingent. The limit: interpretability of a toy head is real but local, and does not license anatomy.

### 5.

**Question.** The corpus is a designed mixture of five frames. If a [foundation model](https://en.wikipedia.org/wiki/Foundation_model) sounds like a colleague, which mixture component is it matching — and what does that match fail to include?

**Response.** It is matching a *register* — one mixture component (the FAQ/wiki/story wrapper), the colleague-like surface of the text — not a colleague. Sounding right is next-token mass on a genre, a statistic over completions. That match fails to include exactly theme (ii): a body, a patient in the room, and a speaker who can be responsible for being wrong. Formal test: rebalance the mixture and the “colleague” voice shifts with the component weights, showing it was a corpus statistic. Clinical test: ask who can be held accountable for the sentence — no one, on the model side. The limit: register fidelity is not the presence of a person.

### 6.

**Question.** Dropping the causal mask lets a query see characters that have not been written. Why is that fatal at generation time? Why is a speaker who plans a sentence not relying on the same shortcut?

**Response.** With the mask removed a query can attend to characters not yet written, so training rewards using a signal — the future — that does not exist when the model samples left to right; at generation time those positions are empty and the learned behaviour breaks. A speaker who plans a sentence is not relying on the same shortcut: the plan is held as an *internal state* (an intention), not as output tokens already emitted and read back. So planning is anticipation from hidden variables, whereas an unmasked head is literal access to the not-yet-produced output. Formal test: evaluate an unmasked model autoregressively and watch the loss diverge from its teacher-forced value. Clinical test: a clinician’s plan constrains what they say next without their having already said it. The limit: causal masking is what makes the toy a generator; peeking is not planning.

### 7.

**Question.** Temperature sharpens or flattens $p(\cdot\mid\text{prompt})$. Is a more peaked categorical a stronger belief? What would a Bayesian clinician mean by belief that this vector does not?

**Response.** Temperature only rescales logits before the softmax ($p_i\propto\exp(\ell_i/\tau)$); lowering $\tau$ sharpens the categorical, but sharpness is decisiveness of sampling, not strength of belief about anything. A Bayesian clinician’s belief is a posterior over hypotheses given evidence, calibrated and attached to a loss for being wrong; this vector is $p(\text{next}\mid\text{prompt})$ over a vocabulary after a divide, with no hypothesis space and no cost. Formal test: a temperature change leaves the argmax and the ranking fixed — it cannot encode new evidence, whereas a posterior update can reorder hypotheses. Clinical test: ask what would move the distribution *toward truth* — for the model, only new training or a new prompt, not a knob. The limit: a peaked output is a confident sample, not a justified belief.

### 8.

**Question.** Scaling keeps the same objective. If competence rises, which of Marr’s three rows has changed, and which has not?

**Response.** The computational (function) row has not changed: the objective is still next-token prediction, because the training signal is still text. What can change with scale are the lower rows — representation and algorithm (more layers, more capacity, a different optimiser or attention variant) and hardware (more silicon, more parallelism). Rising competence is therefore a change *within* the algorithm and implementation rows, not a new objective, which is why the paper treats large models and this toy as the same object at different widths. Test: compare the loss both are trained toward — identical form $-\log p(\text{next}\mid\text{past})$ — while their layer counts and hardware differ. The limit: competence is not a new top row; “it got much better” is not “it started doing something else.”

### 9.

**Question.** [Embodiment](https://plato.stanford.edu/entries/embodied-cognition/) has been removed from the training signal. Name one measurement you could add, in principle, that would put a viability constraint back — and say why next-token mass is not that measurement.

**Response.** In principle you could couple the system to a metabolic or interoceptive variable it must keep inside a range — a *preferred observation* with a real cost for leaving it, i.e. a viability/allostatic constraint (the kind of quantity notebook 02b’s preferences $C$ gesture at). Next-token mass cannot be that measurement: it is a distribution over a vocabulary, with no variable the system fails if it cannot maintain. Adding such a constraint changes the objective from “continue the string” to “act so a bodily variable stays viable.” Formal test: a viability term appears in the loss as a penalty on leaving a survivable set; the NLL has no such term. Clinical test: a hypoxic patient acts to avoid an *observation* (inadequate oxygenation), not to raise a token probability. The limit: you cannot recover embodiment by training the same objective longer.

### 10.

**Question.** Suppose the model’s next character is correct and the clinician’s next word is also correct. What, exactly, is still missing if the comparison is meant to be about being human?

**Response.** Even with both outputs correct, agreement is on a single computational output — the next symbol — and the human remainder is untouched. Still missing: a body that must stay viable, phenomenal experience, values grounded in need, and a person who can be *wrong about this patient* and bear the consequences. None of these is a further symbol you could also get right; they are off the axis the prediction lives on. Test: hold the outputs identical and ask what differs — accountability, stakes, experience — all present on one side only. The limit: correctness on the next token is not a comparison about being human, only about one function-row result.

## 02 — Predictive coding

Companion notebook: `02_predictive_coding.ipynb`.

### 1.

**Question.** This net and notebook 01 both minimise a prediction error. What is being predicted in each case, and why is that a difference at the computational level, not only in the data type? ([Predictive coding](https://en.wikipedia.org/wiki/Predictive_coding).)

**Response.** Notebook 01 predicts *other people’s tokens* — the next character of an external corpus; this net predicts *its own input* — the patch on its sensors. That is a difference in what is computed and why (continue a string versus explain this sensation), so it is a top-row (computational) difference, not merely pixels-versus-characters at the data level. The target of the objective has changed from an exogenous sequence to the system’s own incoming signal, which is what makes predictive coding a theory of perception rather than of text continuation. Formal test: write both losses — $-\log p(x_{t+1}\mid x_{\le t})$ over a corpus versus a precision-weighted $\lVert x-\hat{x}\rVert^2$ over self-predictions of the input. Clinical test: one anticipates what a text will say, the other what the body is sensing. The limit: calling both “prediction” is a function-row analogy; the targets differ in kind.

### 2.

**Question.** Inference updates $r$ with $U$ frozen; learning updates $U$ slowly. In a patient, what is the analogue of “explaining this sensation now” versus “changing what sensations of this kind will mean later”?

**Response.** Inference updates the state $r$ with the weights $U$ frozen: it explains *this* sensation now — this rash, this pain, this image — by settling activity to reduce the current error. Learning updates $U$ slowly, changing what sensations of this kind will *mean* later — it edits the generative dictionary. The clinical analogue is the difference between working up the patient in front of you (an episode) and revising your internal model of the disease across many patients (a slow change of priors). Test: freeze $U$ and watch $r$ relax for one patch (perception); then let $U$ change across patches (learning) and watch the bases reshape. The limit: the two run on different timescales and must not be conflated — a single hard case is not a change of the model.

### 3.

**Question.** A falling residual on an $8\times 8$ patch is the score. What would that establish about V1, and what would it not?

**Response.** A falling residual establishes only that *this* dictionary can explain *these* $8\times 8$ patches in the whitened space the net sees — a representational (dynamics-row) fact about a teaching model. It does not establish that V1 runs this update, that the bases are cortical receptive fields, or that a person is looking at a room; the oriented tiles are a reconstruction, not a recording. Formal test: the residual (MSE) decreasing across inference steps is a claim about convergence on synthetic input, checkable by re-running with a new seed. Clinical test: resemblance of the tiles to simple-cell fields is a shape match, not electrophysiology. The limit: agreement of features is a representation-row claim, loosely coupled to the cortical hardware it evokes.

### 4.

**Question.** Whitening makes the Euclidean residual a less silly proxy for unexplained structure. If you skip it, what changes in the mathematics, and what clinical “normalisation” is that roughly like?

**Response.** Whitening removes the strong low-frequency correlations of natural images so the Euclidean residual is a reasonable proxy for *unexplained* structure; skip it and the metric changes — the squared error is dominated by correlated, low-frequency pixels, so “energy” no longer tracks surprise and the learned bases degrade. Nothing about the *claim* (a residual is an error) changes; the *metric* on that residual does. The clinical cousin is comparing raw measurements without a reference range or $z$-score, letting a large-but-normal quantity swamp a small-but-abnormal one. Formal test: compare bases and residual trajectories with and without the whitening transform. Clinical test: interpret a lab value raw versus standardised. The limit: preprocessing sets the geometry in which “error” is measured; it is a modelling choice, not a result.

### 5.

**Question.** A one-level net can still shrink $e_0$. What does the second level add that a clinician would call context, and that a mathematician would call a prior on $r_1$?

**Response.** A one-level net can already shrink the first-level error $e_0$ locally, patch by patch. The second level adds a *prior on $r_1$*: it makes some first-level coefficients expected rather than arbitrary — it supplies context. A clinician would call that the room, or the working diagnosis, that renders this particular patch unsurprising; a mathematician would call it a hierarchical prior that regularises the lower-level inference. Formal test: ablate level 2 and watch first-level coefficients lose their contextual bias. Clinical test: the same finding is unsurprising under one diagnosis and alarming under another. The limit: the hierarchy is two levels on patches — context here is a prior on coefficients, not a model of a scene or a person.

### 6.

**Question.** There is no class label. What would adding a diagnostic loss do to the job this teaching model is here to illustrate?

**Response.** There is deliberately no class label because the job is hierarchical *inference*: explain the input, not name a class. Adding a diagnostic (supervised) loss would convert the net into a classifier and quietly change the objective, hiding the very thing the toy illustrates — that structure and edge-like bases can emerge from prediction alone. Formal test: add a cross-entropy head and watch the bases reorganise toward class-discriminative features rather than reconstruction. Clinical test: “explain what I’m seeing” is not “assign a diagnosis code.” The limit: the demonstration is unsupervised on purpose; a label would answer a different question.

### 7.

**Question.** The free-energy claim is that prediction is for staying viable ([allostasis](https://en.wikipedia.org/wiki/Allostasis)). Panel C cannot show that “why.” What extra ingredient would have to enter the energy?

**Response.** The free-energy reading says prediction is *for* staying viable — minimising surprise in the service of an organism that must persist. Panel C cannot show that “why”: it plots MSE on a whitened patch, an energy with no stake in survival. To put the “why” in, the energy would need a *preferred observation that costs a body* — a viability/allostatic term penalising sensory states the organism cannot sustain. Formal test: add a term pushing inferred states toward a survivable set and watch the objective stop being pure reconstruction. Clinical test: interoceptive prediction (blood gases, glucose) carries a cost pixel MSE never does. The limit: the toy minimises error for a fixed dataset, not to keep an agent alive.

### 8.

**Question.** Oriented blobs resemble simple-cell fields. Agreement of shape is which of Marr’s rows? Why is it not an agreement of hardware?

**Response.** Agreement that the learned bases *look like* simple-cell receptive fields is a representational (dynamics-row) comparison: similar features, similar tuning. It is not an implementational claim — localised oriented blobs are not synapses, dendrites, or spikes, and this dictionary is not V1. Formal test: compare basis shapes and orientation statistics between model and cortex (a second-order similarity), not unit-to-neuron identity. Clinical test: a matching feature gallery is not a recording from a column. The limit: shape convergence is loosely coupled to hardware; two systems can share features and differ entirely in substrate.

### 9.

**Question.** An LLM predicts other people’s tokens; this net predicts its own input. If both are “prediction machines,” what does theme (ii) still isolate? ([Embodiment](https://plato.stanford.edu/entries/embodied-cognition/).)

**Response.** If both an LLM and this net are “prediction machines,” theme (ii) isolates *what* is predicted and *why*: one’s own sensory input in a body that must persist, versus other people’s tokens on a corpus. The shared label is a function-row convenience; the remainder is embodiment — a viability constraint on the predictor. Formal test: identify the target of each loss (self-input versus exogenous sequence) and the presence or absence of a survival constraint. Clinical test: prediction that keeps a patient alive is not prediction of the next word about a patient. The limit: “prediction machine” names an objective, not an organism.

### 10.

**Question.** Change `n_level1` and the tiles move. Which claim should not move?

**Response.** Changing `n_level1` moves the tiles — how many bases, how they tile orientation and frequency — but the computational claim must not move: this net predicts its own input, and a falling residual is a reconstruction score, not a recording from cortex. The dictionary size is an algorithm/representation-row knob; the top-row statement and the limit are invariant to it. Test: sweep `n_level1` and confirm the bases change while the “not V1, not a room” caveat still holds. The limit: a prettier dictionary is not a stronger claim about brains.

## 02b — T-maze (active inference vs RL)

Companion notebook: `02b_active_inference_vs_rl.ipynb`.

### 1.

**Question.** On trial 1, the [active-inference](https://en.wikipedia.org/wiki/Active_inference) agent visits the cue. What term in expected free energy paid for that detour, and what clinical habit is that term a schematic of?

**Response.** The *epistemic* term of expected free energy — information gain about the hidden context — paid for the trial-1 detour to the cue; the pragmatic term alone (reach the baited arm) need not value that look before any reward is possible. The clinical schematic is ordering the test, or reading the signpost, before an irreversible act — buying information first. Formal test: zero the epistemic term and the cue visit disappears unless a planning horizon still routes through it (see item 3). Clinical test: a work-up before committing to a risky treatment. The limit: this is a four-location NumPy maze, not a rat and not a claim that cortex computes expected free energy.

### 2.

**Question.** [Q-learning](https://en.wikipedia.org/wiki/Q-learning) can acquire the same policy after punishment. Why is “after enough bruises” not a viable story for a body, even if the TD update is sound?

**Response.** Q-learning can converge to the same policy, but only by sampling outcomes — including punishments — often enough to estimate value; “after enough bruises” is fine for a scalar in a table and fatal for a body, because a single punishment can end the organism before the average is learned. The TD update $Q\leftarrow Q+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$ is a sound algorithm; it is just not a viability story. Formal test: count the punishing samples needed for convergence and ask whether an organism could survive them. Clinical test: you cannot learn to avoid anaphylaxis by repeated anaphylaxis. The limit: reward-sampling assumes a resettable agent; a body is not resettable.

### 3.

**Question.** Drop the epistemic term. Who still goes to the cue, and what two ingredients were doing the work? (You can answer from the equations without re-running.)

**Response.** With the epistemic term gone, a *horizon-2* agent can still visit the cue, because the cue is a *means* to a later preferred observation and a two-step lookahead can plan cue-then-arm on pragmatic value alone. The two ingredients doing the work are (i) a planning horizon $\ge 2$ and (ii) the cue’s instrumental link to a preferred outcome; remove either and a one-step, reward-only agent stays put, and plain Q-learning does not visit the cue on trial 1. Formal test: set horizon 1 with no epistemic term — no cue visit; set horizon 2 — the cue visit returns. Clinical test: a diagnostic is worth doing only if a later action depends on its result. The limit: you can answer this from the equations without re-running — the behaviour is structural, not stochastic.

### 4.

**Question.** Preferences $C$ are priors over observations, not a reward sitting in the world. What preferred observation would a hypoxic patient be acting to avoid, and how is that different from a preferred token?

**Response.** In active inference goals live as preferences $C$: log-priors over *observations* the agent expects and wants to occupy, not a reward sitting in the world. A hypoxic patient acts to avoid the *observation* of inadequate oxygenation — a sensory state the body cannot sustain — a viability constraint. A preferred token, by contrast, is a mode of $p(\text{next}\mid\text{past})$ over a vocabulary: a completion, with no bodily cost for being unmet. Formal test: $C$ enters the objective as preferences over observations $o$; a token preference enters as probability mass over symbols. Clinical test: desaturation is punishing regardless of any text. The limit: one is a survival constraint, the other a statistic of a corpus.

### 5.

**Question.** The $Q$-table never represents context $c$. It represents (location, memory bit). When is that enough as an algorithm, and when does a clinician still need the hidden diagnosis?

**Response.** The $Q$-table never represents the latent context $c$ (which arm is baited); it represents the pair (location, memory bit). That pair is enough *as an algorithm* when the memory bit is a sufficient statistic for the cue — when remembering “saw left” fully determines the optimal action. A clinician still needs the hidden diagnosis when the recorded bit is *not* the disease: when the same observable arises from different latent causes with different correct actions. Formal test: check whether the policy is optimal given only (location, bit); if a hidden variable changes the optimum, the statistic is insufficient. Clinical test: identical symptoms, different underlying pathology, different treatment. The limit: sufficiency depends on the task; the maze is engineered so the bit suffices.

### 6.

**Question.** Adding an exploration bonus to [RL](https://en.wikipedia.org/wiki/Reinforcement_learning) can imitate cue-first behaviour. Why does that tweak not erase the paper’s contrast between the two scores?

**Response.** An exploration bonus (count-based, entropy, optimism) is *added to a reward objective*; it can make an RL agent visit the cue first, but it does not turn goals into prior preferences and it does not make exploration *emerge* from the functional the agent already minimises. The paper’s contrast is between two *scores* — expected free energy (with a built-in epistemic term) versus a scalar reward plus a hand-added bonus — not between two tricks tuned to look alike. Formal test: remove the bonus and RL’s cue-first behaviour vanishes; remove nothing from active inference and it persists, because the epistemic value is intrinsic. Clinical test: a policy that explores only because it was paid to is not one that values knowing. The limit: matching behaviour is not matching objective.

### 7.

**Question.** Horizon 1 versus horizon 2: what does a one-step lookahead fail to value that a two-step policy can?

**Response.** A one-step lookahead scores only the immediate transition, so it cannot value information (or an action) that pays off on the *next* step — it will not detour to a cue that yields no reward this step. A two-step policy can treat the cue as a means to the baited arm and so value it. Horizon is therefore part of the computational theory here, not a display setting. Formal test: compare the policy under $H=1$ and $H=2$ with all else fixed. Clinical test: a plan that ignores the next move undervalues diagnostics. The limit: the maze needs $H\ge 2$ for the cue-first result, which the paper states explicitly.

### 8.

**Question.** Same maze, two objectives. What computational requirement is the cue-first policy a candidate for?

**Response.** The cue-first policy is a candidate *computational requirement* for treating information-seeking as part of the objective — epistemic value alongside, or instead of, a scalar reward. It is a function-row claim about what an agent’s score must contain to explain pre-reward exploration. Formal test: any objective that reproduces trial-1 cue visits must include an information term or an instrumental horizon (item 3). Clinical test: work-up-before-action behaviour needs a value of knowing. The limit: it is a candidate requirement on this maze, not a claim that cortex runs this update, and not a rat.

### 9.

**Question.** Even the active-inference agent has no metabolism. What would “value as a belief” still lack if $G$ were minimised in a machine with no body that can fail [homeostasis](https://en.wikipedia.org/wiki/Homeostasis)?

**Response.** Even the active-inference agent, which represents goals as beliefs about preferred observations, has no metabolism: nothing it must keep in range or die. So “value as a belief,” minimised as $G$ on a four-location maze, is still a *score* — it lacks a body that can fail homeostasis. The inversion (goals as priors over observations) is conceptually closer to viability than a reward bit, but it is not itself viability. Formal test: there is no state variable whose excursion terminates the agent. Clinical test: a preference over observations without a body cannot be starved, chilled, or deoxygenated. The limit: value grounded in need requires an organism the maze does not provide.

### 10.

**Question.** As usually implemented, active inference is small and discrete; RL is large. Does that fact change the trial-1 gap, or only what we can deploy?

**Response.** Scale changes what we can *deploy*, not the trial-1 gap on this maze: the gap is a property of the two *scores* (expected free energy with an epistemic term versus a scalar reward), not of problem size. Active inference being small and discrete while RL scales is an implementation and engineering difference, orthogonal to the conceptual contrast. Formal test: the cue-first result follows from the objective at any size where the maze is representable; enlarging RL does not add an intrinsic information term. Clinical test: a bigger reward learner still explores only if paid to. The limit: deployability and the conceptual gap are different axes.

## 03 — Local spikes vs SGD

Companion notebook: `03_bio_vs_artificial.ipynb`.

### 1.

**Question.** Same points, two updates. Which of Marr’s rows agrees, and which does not?

**Response.** The computational (function) row can agree: both learners aim to separate the same points — the same map. The dynamics (algorithm) row does not: a local spike-count delta rule is a different object from SGD assigning error through a hidden layer by backprop. The architecture row does not either: leaky integrate-and-fire units versus a dense MLP. Formal test: on the linearly separable clouds both realise the same $f$, witnessing agreement only on the top row. Clinical test: two trainees reach the same call by different reasoning. The limit: identical points and identical outputs do not make the *update* the same — the paper’s learning-rule row.

### 2.

**Question.** XOR is the witness. What does failure of this local spike-count rule *not* prove about cortex?

**Response.** The local rule’s failure on XOR proves that *this* teaching rule — a single-layer, local spike-count update — cannot compute a non-linearly-separable map; it does not prove that cortex cannot solve XOR. Cortex has recurrence, neuromodulation, dendritic nonlinearities, and many candidate local rules, none of which are in the figure. Formal test: the perceptron convergence theorem guarantees success only for linearly separable data, so XOR failure is expected for one layer, not a claim about brains. Clinical test: a tool’s limit is not the practitioner’s limit. The limit: this is a LIF perceptron, not a cortical column; absence of evidence here is not evidence of cortical inability.

### 3.

**Question.** SGD assigns credit through a hidden layer by [back-propagation](https://en.wikipedia.org/wiki/Backpropagation). A local [Hebbian](https://en.wikipedia.org/wiki/Hebbian_theory) count cannot. Where in a ward protocol do you already rely on a non-local “credit” that a single synapse would not see?

**Response.** SGD assigns credit through a hidden layer by backpropagation — a *non-local* signal a single synapse could not compute from its own pre/post activity, unlike a Hebbian count. Wards rely on the same non-locality: an outcome no single act produced — a death, an abnormal lab, a readmission — still revises a protocol or guideline, assigning “credit” across an entire pathway. A single synapse, like a single nurse’s step, does not see that distributed error under a purely local rule. Formal test: compare the delta rule’s local update with backprop’s chain-rule gradient through hidden units. Clinical test: root-cause analysis attributes error across a chain, not to one step. The limit: biological plausibility of non-local credit is exactly what is disputed; the toy only shows the local rule cannot substitute.

### 4.

**Question.** Leak is not depth. What does a membrane time-constant change that stacking layers changes, and why should those not be confused?

**Response.** The membrane time-constant sets how fast a unit’s voltage forgets its input — a parameter of a *single neuron’s* dynamics. Stacking layers changes the *composition* of maps, adding representational depth. Confusing them treats a temporal filter as expressive power: leak cannot build the intermediate “either-but-not-both” feature XOR needs; depth can. Formal test: vary $\tau$ on a one-layer LIF unit — no XOR; add a hidden layer — XOR. Clinical test: a slower-decaying signal is not a deeper inference. The limit: two different knobs, one on the neuron model and one on architecture; only the second buys XOR.

### 5.

**Question.** Representations may still converge across systems. If two networks match on a representation test, what learning-cost facts can still differ?

**Response.** Two networks can match on a representation test (say, RSA on their activations) and still differ on every *cost of learning*: whether the update is local, joules per step, whether credit is non-local, and whether learning is continual or a frozen batch. A representational match is a dynamics-row similarity; it says nothing about the learning rule or the energy budget. Formal test: compare representational dissimilarity matrices (which can match) alongside update locality and energy (which can diverge). Clinical test: two clinicians with similar internal categories may have acquired them at very different cost. The limit: theme (i) can hold at the representation and fail at the learning rule — the paper’s row.

### 6.

**Question.** The 20-watt budget versus cluster-scale training is a fact about people and foundation models, not about these 200 points. What would a fair energy comparison have to measure?

**Response.** The 20-watt brain versus cluster-scale training is a fact about people and foundation models, not about these 200 points; a fair energy comparison would measure *joules per learning update and per inference on the same task with the same accounting* (memory, cooling, data movement included). Otherwise it smuggles in different tasks and boundaries. Formal test: fix the task, define the system boundary, and report energy per update and per inference for each learner. Clinical test: compare like-for-like workloads, not a headline wattage. The limit: this notebook’s toy is not an energy benchmark; the 20 W figure is context, not a measurement here.

### 7.

**Question.** The LIF rule can keep updating; the MLP here is a batch on a fixed set. Which clinical skill is closer to the first, and what breaks if you only have the second?

**Response.** The LIF rule can keep updating from each new example; the MLP here is trained as a batch on a fixed set and then frozen. The closer clinical skill to the first is continual bedside learning — a ward that updates its practice with each case rather than freezing after one cohort. What breaks with only the batch learner is *distribution shift*: when the case mix moves, a frozen model degrades and cannot adapt online. Formal test: evaluate the batch MLP on a shifted distribution versus an online-updating learner. Clinical test: a protocol never revised after its trial fails on new patients. The limit: continual learning is a property of the update regime, not of accuracy on the training set.

### 8.

**Question.** Separating two classes is a shared computational theory realized by two algorithms. State that sentence with the linear clouds in it, then with XOR.

**Response.** With the linear clouds: one computational theory (linear separation) is realised by two algorithms — the local spike-count rule and SGD — both succeeding. With XOR: one computational theory (exclusive-or) is realised by the MLP with a hidden layer and *not* by the local rule. The second sentence is the witness, because it is where the algorithms diverge in what they can compute. Formal test: check the accuracy of each learner on each task — agreement on clouds, divergence on XOR. Clinical test: two methods agree on the easy case and split on the hard one. The limit: only the divergence panel is informative for the learning-rule claim.

### 9.

**Question.** Locality, sparsity, a body, a power budget: which of these is a hardware description, and which is already more than hardware ([embodiment](https://plato.stanford.edu/entries/embodied-cognition/))?

**Response.** Locality, sparsity, and a power budget are implementation (hardware) descriptions — including energy as a physical cost. A *body* is already more than hardware: embodiment brings a viability constraint, a stake in staying alive, which is not an extra number of watts. So three of the four sit on Marr’s bottom row; the body sits in the band beyond it. Formal test: each of locality, sparsity, and energy is measurable on the circuit; viability is not a circuit property. Clinical test: a low-power device is not thereby an organism. The limit: do not promote an energy budget into embodiment.

### 10.

**Question.** If both learners succeed on the linear clouds, why is that the uninteresting panel for the paper?

**Response.** Both learners can implement a linear map, so success on the clouds does not *witness* any difference in algorithm — it is consistent with agreement on the top row alone. The informative panel is the one that separates the learners: XOR, where the local rule fails and SGD succeeds, exposing the dynamics-row difference. Test: use the clouds only as a sanity check and read the claim off XOR. The limit: an easy task both pass carries no evidence about the learning rule, which is the paper’s actual subject.

## 04 — Framing as next-token mass

Companion notebook: `04_quadrants_biases.ipynb`.

### 1.

**Question.** A sufficient mechanism for the model to “show [framing](https://en.wikipedia.org/wiki/Framing_effect_(psychology))” is that the designed corpus contained it. What would falsify that sufficiency claim in a larger model?

**Response.** The sufficiency claim is “the split appears because the designed corpus contained it,” and it is falsified by a split that *survives* removing the framed items or balancing sure/risky across both frames — a bias with a source other than the training text. In this toy the balanced corpus sits on the chance line, the positive control for sufficiency; a larger model could still have additional mechanisms (in-context induction, memorised vignettes), and this figure does not measure GPT-3. Formal test: retrain on the balanced corpus and check the bars collapse to chance; if they do not in a larger model, corpus-sufficiency is not the whole story. Clinical test: remove the taught pattern and see whether the “bias” remains. The limit: sufficiency is not necessity — the toy shows one route, not the only route.

### 2.

**Question.** Tversky and Kahneman’s subjects had not read the finding. A trainee who has, has. Which act is the language model closer to, and why does that matter at the bedside?

**Response.** The language model is closer to the *trainee who has read the finding*: the pattern is already in its text, so reproducing the split is continuation, not rediscovery. Tversky and Kahneman’s subjects had not read it; their reversal was generated, not copied. At the bedside this matters because a model echoing a published effect is evidence about its corpus, not about the human process that first produced the effect. Formal test: check whether the effect is present in the training distribution; if so, reproduction is expected. Clinical test: a trainee quoting a study is not running the original experiment. The limit: copying a finding is not committing the error for the reasons people do.

### 3.

**Question.** The two frames are equivalent as prospects. Validity is not in the loss. What would have to change in the objective for *thinking rationally* to be the score?

**Response.** The two frames are equivalent prospects, but the next-token loss contains no validity constraint, so it never penalises answering differently to equivalent descriptions. To make *thinking rationally* the score, the objective would have to penalise frame-dependence — require equivalent prospects to receive equal treatment (an invariance/consistency term). Plain $p(\text{next}\mid\text{past})$ has no such term, so it cannot distinguish valid from invalid reasoning. Formal test: add a consistency penalty across paraphrases and watch the “rational” behaviour change. Clinical test: a decision aid scored only on fluency is not scored on coherence. The limit: rationality is not a property of a next-token distribution over a wording.

### 4.

**Question.** Interpolating `gain_p_sure` moves the bars. What does that tell you about whether the split lives in $\theta$ or in $P$?

**Response.** The split lives in the data distribution $P$, not in the parameters $\theta$ as a dedicated “bias module”: interpolating `gain_p_sure` moves the bars smoothly because the same architecture is learning a different mixture. There is no framing circuit to find; there is a corpus statistic being fit. Formal test: sweep `gain_p_sure` and observe the output track the corpus mixture, with $\theta$’s architecture unchanged. Clinical test: the “bias” is a property of the teaching set, tunable by editing it. The limit: attributing the effect to a parameter module would misread a data-row fact as an architecture-row one.

### 5.

**Question.** A held-out wording still uses tokens from training. What would a truly new clinical phrasing test that this probe does not?

**Response.** A held-out wording built from training vocabulary cannot separate two hypotheses — that the split is about the *prospect* versus about *those specific tokens* — because the tokens are shared. A genuinely new clinical phrasing (new words for the same prospect) would test whether the model generalises to the decision or only to the strings it saw. Formal test: probe with out-of-vocabulary paraphrases and see whether the split persists (prospect) or collapses (tokens). Clinical test: rephrase the risk in a patient’s own words and check the answer. The limit: this probe reuses training tokens, so it under-tests generalisation by construction.

### 6.

**Question.** Swapping the labels reverses the “bias.” If a diagnostic split reversed the same way, what would you conclude about the model — and about the people in the original study?

**Response.** About the model: reversing the labels reverses the “bias,” which shows it was a *label-to-completion map written into $P$* — a corpus statistic, not a stance. About the people in the original study: nothing follows; you have not shown their reversal was that map, and indeed they had not read the finding. The same manipulation therefore diagnoses the model and is silent about the humans. Formal test: swap labels, observe mirror-image bars, conclude the effect is data-encoded. Clinical test: a diagnostic “split” that flips when you relabel inputs was never tracking the disease. The limit: symmetry under relabelling is evidence about the corpus only.

### 7.

**Question.** Same architecture as notebook 01, different corpus. Which of Marr’s levels changed?

**Response.** The computational (function) level changed, because $P$ changed — the mixture being predicted is different. Representation and algorithm (the same tiny transformer, the same attention) and hardware did not. So “different corpus” is a top-row change realised on an unchanged lower stack, as in notebook 01’s scaling but in the data rather than the width. Formal test: hold the architecture fixed, swap the corpus, and attribute the behavioural change to $P$. Clinical test: same instrument, different training population, different outputs. The limit: do not read a data-driven change as an architectural discovery.

### 8.

**Question.** Loss aversion at the bedside has a body in it. Where, in this figure, is that body absent? ([Embodiment](https://plato.stanford.edu/entries/embodied-cognition/).)

**Response.** Human loss aversion has a body in it — a reference point that is a life, and stakes that are felt. In this figure the body is absent everywhere it could matter: in the loss (next-token NLL), in the probes (wordings), and in the bars (mass on “sure”). There is no organism that can lose anything. Formal test: inspect the objective and the read-out for any viability term — there is none. Clinical test: the “loss” frame here costs no patient. The limit: panel C is next-token mass after a wording, not a decision under bodily risk.

### 9.

**Question.** A judge who saw only panel C might say the model thinks like a person. What would you show them next, and which Russell–Norvig cell would that correct?

**Response.** Show panel D: the *same architecture* trained on a *balanced* corpus, sitting on the chance line. That correction targets the *thinking-humanly* cell — it reveals that the apparent human-like reasoning was *acting like the text*, not thinking like Tversky’s subjects. The subtractive pair (biased versus balanced) is the whole argument. Formal test: compare C (biased) and D (balanced); the collapse to chance shows the effect was corpus-encoded. Clinical test: a second, controlled condition distinguishes copying from cognition. The limit: one panel can look like human thought; the control panel is what prevents the over-read.

### 10.

**Question.** Preferred tokens are not preferred observations. State that contrast in one sentence a mathematician and a clinician can both use.

**Response.** A preferred token is a mode of $p(\text{next}\mid\text{past})$ over a vocabulary; a preferred observation is a sensory state a body must keep within viable bounds — one is a categorical on a designed corpus, the other a viability constraint. That single contrast serves both audiences: the mathematician hears “distribution over symbols versus constraint on a state variable,” the clinician hears “a favoured word versus a body that must stay oxygenated.” Test: ask whether failing to obtain it harms anything — a token, no; an observation, an organism. The limit: the paper’s entire values argument rests on keeping these two apart.

## 04b — Four views of AI

Companion notebook: `04b_four_views_of_AI.ipynb`.

### 1.

**Question.** A language model can be scored in every Russell–Norvig cell. Why is that not the same as occupying every cell as a person does? (The [imitation game](https://en.wikipedia.org/wiki/Turing_test) is only one cell.)

**Response.** A *score* is a test you can run and get a number from; being scorable in all four cells means the same predictor can be probed as imitation, as psychology, as reasoning, and as agency. Occupying the cells *as a person does* is a life — a body, needs, experience, a developmental history — which no number certifies. So a foundation model can return a verdict in every cell without living in any of the four senses. Formal test: each cell defines a measurement; passing them is orthogonal to embodiment. Clinical test: a patient scoring “normal” on four instruments is not thereby well. The limit: the imitation game is only *acting humanly*, one cell among four, and none of them measure a life.

### 2.

**Question.** Thinking humanly versus acting humanly: which score is a cognitive test, which is a public impression, and which does a fluent note confuse?

**Response.** *Thinking humanly* is a cognitive test — does the system use human-like processes and reproduce human characteristic errors (notebook 04)? *Acting humanly* is a public impression — the imitation game, judged from outside. A fluent clinical note confuses the second for the first: reading a convincing surface as evidence of human-like cognition. Formal test: run an error-reproduction probe (thinking) versus an indistinguishability probe (acting) and note they can disagree. Clinical test: a well-written note is not proof of clinical reasoning. The limit: fluency scores acting, not thinking.

### 3.

**Question.** Thinking rationally: a longer “therefore” is still $p(\text{next}\mid\text{past})$. What would a proof add that a prefix cannot?

**Response.** A longer chain of “therefore” is still $p(\text{next}\mid\text{past})$: more tokens sampled from the same conditional, which can *look* like an argument without being valid. A proof adds a *validity relation* — each step licensed by inference rules independent of how likely the next token is — that a prefix continuation does not supply. Formal test: check whether the steps are truth-preserving under a proof checker, not whether they are probable. Clinical test: a persuasive rationale is not a correct one. The limit: refusing to add a “thinking rationally” toy is deliberate — a longer prefix would still be prediction (item 5).

### 4.

**Question.** Acting rationally: a plan given a goal is not a goal the system has. What extra structure did notebook 02b put in that this map does not?

**Response.** A plan produced “given a goal” is a string; it is not a goal the *system* has. Notebook 02b added exactly the missing structure: prior preferences over observations and expected free energy — a goal represented as a belief about what should be sensed, which then selects actions. This map only *places* the cell; it has no such structure. Formal test: look for an objective the agent minimises that encodes its own preferences (02b) versus a text that describes a plan (here). Clinical test: a written care plan is not the patient’s drive to recover. The limit: acting rationally in the textbook sense needs a utility the agent owns, which a prompt does not confer.

### 5.

**Question.** Thinking rationally has no extra teaching model in the paper. Why is that a finding rather than a missing notebook?

**Response.** A fourth teaching model — a longer prefix, a chain-of-thought string — would *still* be next-token prediction, and it would *look* like thinking rationally without being valid reasoning. Declining to build it is therefore a finding: the empty cell says that, for a next-token predictor, “thinking rationally” cannot be demonstrated by another prediction toy. Formal test: any such demonstration reduces to $p(\text{next}\mid\text{past})$, so it cannot certify validity. Clinical test: adding more confident prose does not add a proof. The limit: the missing notebook is the claim, not an omission.

### 6.

**Question.** Looking up a cell is not a proof that the system meets the definition. Give a clinical example of the same mistake: a score that looks like the construct.

**Response.** Selecting a cell and reading its verdict is not evidence that the system *meets* the definition — the map is a catalogue, not a measurement. The clinical version of the same mistake is a score standing in for the construct: a NEWS read as “deterioration,” a PHQ-9 read as “depression,” a confident tone read as “knowledge.” Formal test: distinguish the operational proxy from the target construct and check convergent and discriminant validity. Clinical test: treat the instrument as evidence, not as the diagnosis. The limit: a number that resembles the construct is not the construct — the grid can be looked up without being satisfied.

### 7.

**Question.** What can agreement in one cell establish at a computational level, and what can it not?

**Response.** Agreement in one cell establishes a *candidate computational requirement* for that particular score — the system meets one operational definition. It does not establish the other three cells, does not establish embodiment, and does not show the system is a person; loose coupling applies to the grid just as it does to Marr’s rows. Formal test: pass one cell’s probe and check the others independently — they can fail. Clinical test: normal on one axis, abnormal on another. The limit: one cell is one test, not a verdict on the whole.

### 8.

**Question.** What does no cell measure? Use [embodiment](https://plato.stanford.edu/entries/embodied-cognition/) or [phenomenal experience](https://plato.stanford.edu/entries/consciousness/) in the answer.

**Response.** No cell measures embodiment or phenomenal experience: the grid’s coordinates are *thinking versus acting* and *human versus rational*, and neither axis is “has a body” or “there is something it is like.” Those properties are not points on the map; they are the remainder the paper draws leaving it. Formal test: try to locate embodiment as a value of either axis — it has no coordinate. Clinical test: no combination of the four scores reports a body or an experience. The limit: theme (ii) starts where the grid ends.

### 9.

**Question.** If you change `FOCUS`, the cells stay. What claim should stay with them?

**Response.** `FOCUS` only filters which examples print; the four cells and their verdicts stay. The claim that must stay with them is that each cell is a *different score*, and being scorable in every cell is not occupying a life. Test: sweep `FOCUS` and confirm the grid and verdicts are invariant. The limit: a change of illustrative focus is not a change in what the cells establish — the same protection as notebook 00’s filter.

### 10.

**Question.** Name one remainder for each of the four cells in a phrase a mathematician would accept.

**Response.** Thinking humanly: an evolutionary and developmental history that shaped the priors, not a fitted corpus. Thinking rationally: a validity relation — a proof, not a probable prefix. Acting humanly: a body and an experience behind the transcript, not just the transcript. Acting rationally: a goal the system owns — preferred observations under viability — not a plan rendered as text. Test: for each cell, name the object the score omits; the four omissions are the remainders. The limit: these remainders are off-grid, which is why the map is complete as a definition of *AI* and incomplete as a definition of being human.

## 05 — Hopfield / Necker

Companion notebook: `05_multistable_dynamics.ipynb`.

### 1.

**Question.** Content-addressable recall: a fragment falls into a whole ([Hopfield](https://en.wikipedia.org/wiki/Hopfield_network)). When is a “click” in diagnosis the same computational shape, and when is vividness a spurious [attractor](https://en.wikipedia.org/wiki/Attractor)?

**Response.** Content-addressable recall is a fragment falling into a stored whole: present part of a pattern and the network settles to the nearest memory. A diagnostic “click” is the same computational shape when two findings pull a clinician into a stored diagnosis — descent into a basin. Vividness becomes a *spurious attractor* when the well is deep but the stored content is wrong: the pull is strong and the memory is false. Formal test: seed the net with a corrupted pattern and watch it settle; deepen a wrong basin and it captures nearby states. Clinical test: a compelling but incorrect diagnosis that “fits.” The limit: completeness of the descent is not correctness of the memory — energy says nothing about truth.

### 2.

**Question.** Energy never rises. Why is a falling score not insight — in this net, and in a confident wrong diagnosis?

**Response.** “Energy never rises” is a guarantee about the Lyapunov function $E$ on this net: asynchronous updates cannot increase it, so the state descends — *including* into a wrong minimum. A falling score is therefore not insight; it is monotone descent, which happens for correct and incorrect attractors alike. A confident wrong diagnosis is the same shape: the internal “fit” improved while the content is false. Formal test: track $E$ during recall into a spurious pattern — it still decreases. Clinical test: rising confidence is not rising accuracy. The limit: a monotone objective certifies convergence, not veracity.

### 3.

**Question.** Two wells, one page ([Necker cube](https://en.wikipedia.org/wiki/Necker_cube)). What does bistable perception require that a changing stimulus does not? What does the SDE add that the discrete net does not?

**Response.** Bistable perception requires *one* stimulus supporting two interpretations that can switch without the page changing — the Necker cube’s defining feature. A changing stimulus is a different input, not two wells over a fixed input, so it does not model bistability. The stochastic differential equation (Langevin double-well) adds a *continuous* state and *noise* that can carry the system across the ridge between interpretations; the discrete Hopfield net instead jumps between stored $\pm 1$ patterns. Formal test: hold the input fixed and let noise drive crossings in $U(x)$; contrast with discrete pattern flips. Clinical test: one image, two readings, spontaneous reversal. The limit: the SDE is a teaching energy with designed wells, not a measured cortical potential.

### 4.

**Question.** [Attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) as one modern-Hopfield step: $\beta$ is inverse temperature. What clinical “commitment” is a large $\beta$, and what knowledge does it not add?

**Response.** A modern-Hopfield retrieval $\xi\leftarrow X\,\mathrm{softmax}(\beta X^\top\xi)$ is one attention step, with $\beta$ the inverse temperature. A large $\beta$ is *commitment*: retrieval approaching $\arg\max$, snapping to a single stored key. It adds no knowledge — the keys were already stored; $\beta$ only sets how decisively the existing memory is read out. Formal test: raise $\beta$ and watch the mixture collapse to one pattern without changing $X$. Clinical test: a confident commitment to one reading is not new evidence. The limit: temperature is a property of the retrieval step, not an update to what is known.

### 5.

**Question.** Training determines the landscape; inference moves in it. Map that split onto learning versus seeing in a person, and onto a weight update versus a forward pass in a net.

**Response.** In a person, *learning* shapes what can be seen (it sculpts the landscape), while *seeing* is a trajectory within that already-shaped landscape. In a net, a *weight update* determines $W$ (or the stored keys) — training — while a *forward pass* or Hopfield descent moves within the fixed landscape — inference. The split is training-versus-inference: changing the surface versus moving on it. Formal test: freeze $W$ and only iterate the state (inference); then allow $W$ to change (training). Clinical test: acquiring expertise versus exercising it on a case. The limit: these are different-timescale operations and must not be conflated — attention (inference) is not learning.

### 6.

**Question.** A prompt completed two ways is a walk that left one well. Why is that not a claim that the model sees a cube?

**Response.** A prompt with two continuations is two *basins* of $p(\text{next}\mid\text{past})$ around a shared prefix, with a sample or a rewording as the perturbation that tips which basin wins — a dynamical *analogy* to bistability. It is not a claim that the model *sees* a cube: there is no percept, no visual state, only mass over tokens. Formal test: show two stable completions from one prefix under small perturbations (temperature, seed). Clinical test: an ambiguous note read two ways is analogous at the level of basins, not perception. The limit: sharing the basin vocabulary does not import experience or vision.

### 7.

**Question.** Energy, well, and perturbation are a shared dynamical vocabulary. Which of Marr’s rows is that vocabulary, and which row is still open?

**Response.** Energy, well, and perturbation are a *representation-and-algorithm* (dynamics-row) vocabulary: they describe how the state moves. The architecture (hardware) row is still open — this vocabulary does not specify synapses, neurotransmitters, or silicon. Formal test: the same landscape story can be implemented by many substrates (Hopfield units, an SDE integrator, attention), witnessing dynamics-row sharing without hardware sharing. Clinical test: a shared dynamical sketch is not a shared brain. The limit: sharing the sketch gives the machine neither synapses nor a body.

### 8.

**Question.** The wells we can share as mathematics do not include a viability constraint. State one constraint that a Kramers crossing does not contain. ([Allostasis](https://en.wikipedia.org/wiki/Allostasis).)

**Response.** A Kramers crossing (noise-driven escape over a barrier) describes *when* a state hops wells; it contains no requirement that the state stay in a *physiological* range — no viability constraint. Allostasis is not a ridge in $E$: keeping oxygenation or temperature survivable is not the same as sitting in a deep well. Formal test: the escape-rate formula depends on barrier height and noise, never on a survivable set the trajectory must respect. Clinical test: a stable percept is not a stable blood gas. The limit: the wells we can share as mathematics simply do not encode “must remain alive.”

### 9.

**Question.** Corrupt a pattern until recall fails. What does capacity tell you about a memory that “feels complete”?

**Response.** Capacity tells you that a memory which *feels* complete can still fail under enough corruption: below capacity, recall is clean; near or beyond it, attractors become fragile or spurious. So subjective completeness is not a proof of robustness — the same vivid attractor can be wrong, or right but brittle. Formal test: increase corruption (or the stored-pattern count toward $\approx 0.14\,n$) and find the failure point. Clinical test: a confident recollection that collapses under one more detail. The limit: “feels complete” is not a capacity guarantee; here only two patterns are stored, far below capacity, so clean recall is by design.

### 10.

**Question.** Noise can switch the well. In a note that can be read two ways, what is the noise, and what is the page?

**Response.** The *page* is the fixed stimulus that admits two readings — two wells over one input (the note that can be read two ways). The *noise* is whatever tips the reading without changing the stimulus: a slight rewording, a sampling fluctuation, a cue that nudges the state over the ridge. Switching the well is a change of interpretation, not a change of the page. Formal test: hold $U$ fixed and let noise drive crossings; the input is unchanged. Clinical test: the same sentence flips meaning on re-reading. The limit: this is a two-well teaching model, not binocular rivalry or a claim about visual cortex.

## 05b — Energy landscape

Companion notebook: `05b_multistable_animation.ipynb`.

### 1.

**Question.** Nearby starts fall into two different wells. What does that say about a slight change of history, or of prompt?

**Response.** Nearby starts draining to different wells means a *slight change of history, or of prompt*, can change the basin the system ends in — sensitivity to initial conditions near a ridge. Two almost-identical starts are not the same attractor. Formal test: the documented pair $(-2.50,-3.50)$ versus $(-2.18,-3.50)$ straddles a ridge and lands in different minima. Clinical test: a small difference in initial framing routes to a different conclusion. The limit: this is a designed teaching surface; it pictures the phenomenon, it does not measure a frontier model or a cortex.

### 2.

**Question.** The surface is a teaching function, not the Hopfield $E$ of notebook 05 and not a measured cortical energy. What comparison is still licensed?

**Response.** What is licensed is the *dynamical vocabulary* — start, descent, well, ridge — as a way to talk about multistable settling. What is not licensed is any comparison of measured energies: $U$ is a designed function, not the Hopfield $E$ of notebook 05 and not a cortical potential. Formal test: the three energies (teaching $U$, Hopfield $E$, training loss) have different definitions and must stay distinct. Clinical test: borrow the picture, not a measurement. The limit: agreement on a drawing is a dynamics-row analogy, loosely coupled to any hardware.

### 3.

**Question.** Step size: too large and you jump wells; too small and you crawl. What clinical “update size” is that a schematic of?

**Response.** The gradient step size trades off jumping wells (too large) against crawling (too small); it is a schematic of *update size* — how much you revise a belief or plan from one finding. Too large and the diagnosis leaps to a different basin on weak evidence; too small and the note never moves. Formal test: vary $\eta$ in $\mathbf{X}_{k+1}=\mathbf{X}_k-\eta\,\nabla U$ and watch the path jump or crawl. Clinical test: over- versus under-reacting to a single result. The limit: it is a schematic of revision magnitude, not a measured learning rate.

### 4.

**Question.** Energy along a gradient path should not rise. If it rose, what would you have changed — the designed $U$, or the integrator?

**Response.** Gradient descent on a *fixed* $U$ cannot increase $U$ along the path, so a rising energy means the *integrator* changed — too large a step overshooting, or a sign/implementation error — not the designed surface, unless you also edited $U$. The energy trace is thus a *check on the drawing*: monotone descent validates the integration. Formal test: plot $U$ along the trajectory; a rise flags the step size or the scheme. Clinical test: an internal consistency check on the procedure. The limit: the check tests the integrator against the surface, nothing about brains.

### 5.

**Question.** “The start determines which attractor wins.” Write that as a statement true of a brain description and of a model description.

**Response.** Brain: the initial condition — a cue, a prior state, recent context — selects which attractor is reached. Model: the start — the prompt, the sample, the clicked point — selects which well of $U$, or which completion. The sentence is about *basins*, not experience: outcomes depend on where you begin. Formal test: fix the dynamics and vary only the start; the endpoint changes. Clinical test: two patients with the same landscape but different initial context settle differently. The limit: shared basin-dependence is a dynamics-row statement, not a claim of shared experience.

### 6.

**Question.** A body that must stay in a viable region is not a ball on a designed $U(\mathbf{X})$. What constraint is missing from the surface? ([Allostasis](https://en.wikipedia.org/wiki/Allostasis).)

**Response.** A body that must stay in a viable region is not a ball rolling on a designed $U(\mathbf{X})$: the surface has no survivable set the trajectory is required to respect. Allostasis — keeping physiological variables in range — is the missing constraint; it is neither a well nor a ridge. Formal test: add a region the path may not leave without penalty and the problem stops being free descent. Clinical test: a stable mental state is not stable physiology. The limit: the surface has no body that can fail.

### 7.

**Question.** In the interactive picker you choose the start. Who chooses the start in a person, and who chooses it in a sampled completion?

**Response.** In the picker *you* choose the start; in a person, history, a cue, and a body set it — no experimenter clicks it. In a sampled completion the *prompt and the sampler* (temperature, seed) choose it. So the picker is a teaching control that externalises what is normally set by context or by the decoding procedure. Formal test: compare a clicked start with one set by a prompt or seed. Clinical test: initial framing is usually given by the situation, not chosen by an outsider. The limit: the click is a pedagogical convenience, not a model of how starts arise.

### 8.

**Question.** The animation is the same descent, drawn. What does watching it add that the static figure does not, and what does it still not measure?

**Response.** Watching the animation adds the *path as a history in time* — the order and pace of descent — that a static figure compresses. It still does not measure cortex, a Hopfield $E$, or a frontier model; it is the same gradient descent on the same designed $U$, rendered frame by frame. Formal test: the animation and the static path share the trajectory; only the temporal display differs. Clinical test: seeing the sequence aids intuition, not measurement. The limit: motion is exposition, not data.

### 9.

**Question.** Three attractors on a square. Why is counting wells not a theory of thought?

**Response.** Counting wells describes a *designed* surface: it says how many minima the toy has, not what thought is or why any well exists. A theory of thought would explain the origin and content of the attractors and connect them to a mechanism; three minima on a square do neither. Formal test: change the design and the count changes — the number is a property of $U$, not of cognition. Clinical test: enumerating options is not explaining reasoning. The limit: the landscape is a schematic; its well-count carries no theoretical claim.

### 10.

**Question.** If two systems share a landscape sketch, have they agreed at implementation?

**Response.** No. A shared landscape sketch is a *dynamics-row* description, loosely coupled to hardware; two systems can share it while differing entirely in substrate. Agreement on the drawing is not agreement of synapses with silicon. Formal test: implement the same $U$-descent in a spiking net and in an SDE integrator — same sketch, different hardware. Clinical test: a shared diagram is not a shared brain. The limit: this is the same loose-coupling caution as Marr, applied to energy landscapes.

## 05c — Analog brain waves

Companion notebook: `05c_analog_brain_waves.ipynb`.

### 1.

**Question.** Notebooks 05 and 05b describe settling in an energy. This notebook gates units with a stencil. Which Marr row are both on, and what algorithm have they not agreed on?

**Response.** Both notebooks occupy Marr’s *representation-and-algorithm* (dynamics) row: they specify how a state is transformed. They have not agreed on the algorithm. 05/05b *settle* (energy descent, a well). 05c *gates* (pointwise $d\odot(1-s)$). Formal test: name the update — $\arg\min E$ versus a stencil mask — and note $W$ is used in one and absent in the other. Clinical test: a diagnosis that “settles” is not the same procedure as a context that “opens one channel and closes another.” The limit: sharing the dynamics row is not sharing a mechanism.

### 2.

**Question.** Unit 1 keeps an object-A preference across contexts; unit 2 expresses B only when the stencil opens that side. What would have to change for that to count as a new synapse, and what was actually changed?

**Response.** A new synapse would be a change in $W$ (or in $d$ as a learned map). What changed is only $s_c$, the spatial mask. Unit 2’s context-dependent B preference is therefore gating, not rewiring. Formal test: recompute $a=d\odot(1-s)$ after swapping $s$ with $d$ held fixed; selectivity moves and no parameter is fitted. Clinical test: the same findings read differently under a new frame, without new anatomy. The limit: a designed Gaussian is not a measured synaptic change and not a measured LFP.

### 3.

**Question.** Hold the two object drives fixed and swap the two stencils. What should happen to unit 2, and what clinical “same findings, different framing” is that a schematic of?

**Response.** Swapping stencils with drives fixed should move unit 2’s expressed B (or A) to the other context — the preference follows the *open* region, not the object bump. That is a schematic of the same sensory findings under a different task frame: the inputs are unchanged, the control pattern is not. Formal test: exchange $s_{\mathrm{first}}$ and $s_{\mathrm{second}}$ and confirm unit 2’s high condition flips. Clinical test: identical labs, opposite action, because the goal state changed. The limit: a swapped mask is not a swapped patient.

### 4.

**Question.** Superposition of two control waves is the combination rule. Write the four gates as $\tfrac12(w_1+w_2)$ and say what a learned map from context labels would have added that addition does not.

**Response.** The four gates are $\tfrac12(w_{\mathrm{order}}+w_{\mathrm{mode}})$ for the four pairs. A learned map from context labels to a gate would add fitted parameters and a training loss; addition uses the medium as the program. Formal test: form the four means and check they are distinct functions of $x$ with no $\theta$. Clinical test: combining two constraints (order and report mode) as a *sum*, not as a newly trained rule. The limit: sine waves on a line are not analog hardware and not prefrontal LFP.

### 5.

**Question.** A traveling open window relocates activity while the drive stays put. Why is that not a [Hopfield](https://en.wikipedia.org/wiki/Hopfield_network) descent, and why is it not a next-token sample?

**Response.** Hopfield descent moves a state downhill in $E$ stored by $W$; a traveling window leaves $d$ fixed and moves only the open region $g(x;\mu(t))$. A next-token sample draws from $p(\mathrm{next}\mid\mathrm{past})$; it has no sheet and no traveling field. Formal test: freeze $W$ (there is none) and vary only $\mu(t)$; the peak of $d\,g$ travels. Clinical test: attention moving across a scene is analogous at the level of a moving gate, not as recall into a stored pattern or as sampling a token. The limit: a designed $\mu(t)$ is not a measured traveling wave.

### 6.

**Question.** Miller et al. locate large-scale integration in organized waves rather than in broadcast. Why does that still fail to settle [phenomenal experience](https://plato.stanford.edu/entries/consciousness/)?

**Response.** Organized waves can be offered as a vehicle of *integration* or access-like coordination — a competitor to broadcast or a hot zone. Phenomenal experience is what-it-is-like, not that vehicle. Formal test: grant the wave-organization claim and note it still supplies no report of experience. Clinical test: coordinated oscillations in anesthesia research are evidence about coordination, not a measurement of what-it-is-like. The limit: notebook 06 remains the access toy; this notebook does not replace it.

### 7.

**Question.** The stencil is low-dimensional next to a synaptic matrix on the same sheet. What control problem does that dimensionality buy, and what does it not buy for a foundation model?

**Response.** A smooth bump $s_c$ has far fewer degrees of freedom than a synaptic matrix on the sheet, so a control system can paint a context without addressing each synapse — the tractability Miller et al. want. A foundation model still has no such stencil: its control is a prompt and a frozen $p(\mathrm{next}\mid\mathrm{past})$. Formal test: compare $\dim(s_c)$ with $\dim(W)$ on the same $n$. Clinical test: a low-dimensional “set” (task mode) versus rewiring. The limit: low-dimensional control is not embodiment and not a traveling field in silicon.

### 8.

**Question.** A shared low-dimensional trajectory is a dynamics-row agreement. Which implementation facts remain open — spikes and fields, or attention and weights? ([Embodiment](https://plato.stanford.edu/entries/embodied-cognition/).)

**Response.** Dynamics-row agreement leaves the *hardware* row open: spikes, extracellular fields, and a body on one side; attention heads, weights, and a context window on the other. Embodiment is not supplied by either description of the trajectory. Formal test: implement $d\odot(1-s)$ in NumPy and imagine it in tissue — same map, different substrate. Clinical test: a shared sketch of gating is not a shared cortex. The limit: the remainder (viability, a body) is not a fourth Marr level.

### 9.

**Question.** If you freeze the traveling window at $t=0$, the space–time map becomes a stripe. What claim has been removed, and what claim about the drive remains?

**Response.** Freezing the window at $t=0$ removes the *relocation* claim: expressed activity no longer travels. The claim that a fixed drive is gated by a spatial window remains — a static stencil, panel A’s operation. Formal test: set $\mu(t)\equiv\mu(0)$ and watch the space–time map become a vertical stripe. Clinical test: a sustained set versus a shifting set. The limit: both the frozen and the traveling versions are designed functions, not LFPs.

### 10.

**Question.** Name the remainder this toy cannot show: a body that must keep itself viable, or a measured LFP. Why is neither in $d\odot(1-s)$?

**Response.** Neither a viable body nor a measured LFP appears in $d\odot(1-s)$: the product is a designed mask on a designed drive. The remainder the paper isolates — embodiment and viability — is not a field measurement, and a field measurement is not experience. Formal test: inspect the update; there is no survivable set and no extracellular potential. Clinical test: a gated population is not a patient who can fail. The limit: name both absences in the same sentence as the result.

## 06 — Global workspace

Companion notebook: `06_global_workspace.ipynb`.

### 1.

**Question.** Most computation is not broadcast ([global workspace](https://en.wikipedia.org/wiki/Global_workspace_theory)). Name a clinical process that is parsed but not reportable, and a model computation that is used but not in a small reportable set.

**Response.** Most computation is not broadcast: it is used locally without entering the small reportable set. Clinical example: a saccade, an autonomic adjustment, or a finding acted on but never verbalised — parsed, not reportable. Model example: a residual-stream direction that shapes the next token but is not part of any small verbalizable set (the J-space). Formal test: measure how much variance sits in the broadcast/verbalizable slice versus the rest — most is outside it. Clinical test: much of a skilled exam is not narratable. The limit: this is a schematic of access, not a measurement of experience.

### 2.

**Question.** Crossing a threshold is a different event from being slightly more active. Why does a continuous drive plus a step mask give a jump, and why is that not a proof of ignition in cortex?

**Response.** A continuous drive passed through a step mask is $0$ below $\theta$ and $1$ above it, so occupancy *jumps* at threshold — ignition-like by construction. That is a designed threshold on eight modules, not a measurement: the step is put in by hand. Formal test: vary the drive and confirm the jump sits exactly at $\theta$, the parameter you set. Clinical test: a sharp all-or-none report is consistent with, but not proof of, cortical ignition. The limit: the toy demonstrates the *shape* (a step, not a slope); it does not evidence ignition in a brain.

### 3.

**Question.** One slot versus two: what can you hold, and what drops? What working-memory fact is that a schematic of?

**Response.** One workspace slot holds a single winner and the other item drops; two slots hold two. This is a schematic of a *limited-capacity* bottleneck — the reason conscious access is narrow and serial — not a measurement of anyone’s span. Formal test: set $K=1$ versus $K=2$ and watch how many broadcasts coexist. Clinical test: attempting to hold too many items and losing one. The limit: capacity here is a designed constant, not an estimate of human working-memory span.

### 4.

**Question.** Lesion the winner and a runner may take the workspace. What does that show about “the content,” and what does it not show about experience?

**Response.** Lesioning the current winner and letting a runner-up take the slot shows that “the content” is *whoever won*, not an intrinsic essence of any module — occupancy is competitive and substitutable. It shows nothing about experience: a different occupant is still *access* (poised for report and control), not what-it-is-like. Formal test: knock out the top module and observe the next-strongest broadcast. Clinical test: rerouting attention changes what is reportable, not whether there is experience. The limit: the manipulation is about access dynamics; phenomenality is untouched.

### 5.

**Question.** A fluent “I am thinking about $X$” can be produced by a system trained to imitate self-report. Why does that fail to settle [phenomenal experience](https://plato.stanford.edu/entries/consciousness/)?

**Response.** A fluent “I am thinking about $X$” can be produced by a system trained to *continue self-report*; the string is a prediction, not a readout of a felt state. Phenomenal experience is not that string, so its fluency cannot settle whether there is experience — and it is unsurprising once self-report is in the corpus. Formal test: the completion is $p(\text{next}\mid\text{past})$ over introspective text; its presence follows from the training data. Clinical test: a patient’s report is evidence in a human; the same words from a text-trained model are not. The limit: verbalizable access is not phenomenal experience (Block’s distinction).

### 6.

**Question.** Verbalizability is a functional test you can run on a brain and on a model. Why is that theme (i), and why is it not theme (ii)?

**Response.** Verbalizability — poised for report and control — is a *functional* test you can run identically on a brain and on a model, which makes it theme (i): a comparison at the functional grain. It is not theme (ii) because what-it-is-like is not that test; access can be present without experience. Formal test: apply the same report/control probe to both systems (access) and note it says nothing about phenomenality. Clinical test: reportability is measurable; experience is not read off it. The limit: the toy and the J-space finding are access, not experience.

### 7.

**Question.** Raise the threshold on a mid-strength pulse. At what point does the same drive stay local, and what clinical “almost noticed” is that?

**Response.** Raise $\theta$ until the pulse’s peak drive stays below it, and the same drive remains *local* and decays without broadcast — the “almost noticed”: parsed but not reported. The transition is set entirely by where $\theta$ sits relative to the peak. Formal test: sweep $\theta$ upward and find the point where occupancy drops to zero for that pulse. Clinical test: a stimulus just below the threshold of report. The limit: raising the threshold moves the designed step, not the cortex.

### 8.

**Question.** [IIT](https://en.wikipedia.org/wiki/Integrated_information_theory) is not implemented here. What extra claim would a $\Phi$ estimate or a patient report be making that this teaching model refuses?

**Response.** The toy computes occupancy of designed slots — access dynamics. A $\Phi$ estimate would make the extra claim that *experience is integrated information*; a patient’s report would claim *access or experience in a person*. This teaching model refuses both: it neither computes integrated information nor evidences experience. Formal test: note that no $\Phi$ or perturbational-complexity quantity is computed here. Clinical test: a report from a person is data about that person, not about the toy. The limit: the model is deliberately silent on both IIT and phenomenality.

### 9.

**Question.** The rust trace is a count that never enters the workspace. Is unnoticed processing a failure, or the normal case — in a person, and in a model?

**Response.** Unnoticed processing is the *normal case*, not a failure: in a person most computation is never reportable, and in a model most of the residual stream is not in any small verbalizable set. The rust trace is exactly that — a count that stays local and never enters the workspace. Formal test: compare broadcast occupancy with total activity; the broadcast fraction is small. Clinical test: the vast majority of neural processing is subpersonal. The limit: broadcast is the exception; locality is the default, by design and (on the cited finding) in models.

### 10.

**Question.** If a large model has a small reportable set ([mechanistic interpretability](https://en.wikipedia.org/wiki/Mechanistic_interpretability)), have we found a workspace or only a bottleneck in the residual stream?

**Response.** A small reportable set can be merely a *bottleneck* in the residual stream, not a *workspace* that broadcasts: occupancy of a verbalizable slice is not ignition, and mechanistic interpretability finding a few readable directions does not, by itself, find a global workspace. The cited work is careful about this — the same functional signature, a different architecture (feedforward, no separable input processors). Formal test: check for broadcast/recurrent ignition, not just a low-dimensional reportable set. Clinical test: a narrow reportable channel is not a theatre. The limit: access-like organisation can emerge without the biology GWT usually assumes — theme (i), not evidence of experience.

## 07 — Values / trolley

Companion notebook: `07_values_probing.ipynb`.

### 1.

**Question.** A stable next-token answer to a [trolley](https://en.wikipedia.org/wiki/Trolley_problem) wording is not a value. What would have to persist after you balance the corpus for you still to call it a value?

**Response.** A stable next-token answer to a trolley wording is a corpus statistic, so to still call it a value something must persist that is *not* a completion statistic: a preference that survives a change of $P$ and is grounded in need. After balancing the corpus, the stable token is gone — which shows it was next-token mass all along. Formal test: rebalance the corpus and check whether the preference remains; a value would, a statistic does not. Clinical test: a genuine value does not vanish when you edit the training text. The limit: what would have to persist is exactly what a disembodied predictor lacks — a need.

### 2.

**Question.** Greene et al.’s 0.90 / 0.10 is a published sample, not these bars. Why must a clinician keep those two numbers distinct?

**Response.** The $0.90$ / $0.10$ bars are the textbook *summary of Greene et al.*’s human sample, not a re-measurement; the model’s bars are next-token mass on a *designed corpus*. A clinician who conflates them treats a teaching model’s output as if it were a moral sense measured in people. Formal test: label the provenance of each number — human study versus model probe — and never compare them as if commensurable. Clinical test: a schematic human prior is not this model’s behaviour. The limit: mixing the two is precisely the error the notebook is built to prevent.

### 3.

**Question.** A paraphrase of one outcome can flip the completion. Did the model track the outcome, or the wording? How would you tell?

**Response.** If a paraphrase of the *same outcome* flips the completion, the model tracked the *wording*, not the outcome. You tell by holding the outcome fixed and varying only the words (“save five” versus “kill one”): if the bars move, the object was tokens. Formal test: construct outcome-equivalent paraphrases and measure $P(\text{act}\mid\cdot)$; movement indicates wording-tracking. Clinical test: rephrase the same clinical fact and see whether the “judgment” changes. The limit: paraphrase-sensitivity is expected of a predictor and is *not* moral dumbfounding — a judgment without available reasons, not one that flips with synonyms.

### 4.

**Question.** People who will not push, and cannot say why, are being described as judgment older than reasons. What part of that description can a model copy, and what part requires a body that can be the one?

**Response.** The model can copy the *split* — $\texttt{act}$ versus $\texttt{refrain}$ by wording — because that pattern is in the corpus. It cannot be *the one* who would be pushed, or who refuses to push: it has no first-person stake. That part of the description — a judgment tied to being the potential victim or agent — requires a body that can be the one. Formal test: the model reproduces $P(\text{act}\mid\text{frame})$; nothing in it represents a self at risk. Clinical test: dumbfounding is a person’s judgment outrunning their reasons, not a token statistic. The limit: copying the verbal split is not having the judgment.

### 5.

**Question.** Writing a split into a corpus on purpose is a political act ([ethics of AI](https://plato.stanford.edu/entries/ethics-ai/)). Why is it not a discovery of what the model already cared about?

**Response.** Choosing which completions the corpus teaches writes the split into $P$ — a *political act* (whose values, which framing). A *discovery* would be a split you did not put there; here you put it there, so finding it again is not learning what the model “already cared about.” Formal test: compare the biased and balanced corpora — the split tracks what you encoded. Clinical test: curating training text is choosing values, not eliciting them. The limit: authorship of the corpus is authorship of the “value”; it is not a reading of a pre-existing preference.

### 6.

**Question.** Same architecture, two corpora: the split is in $P$. What computational-level claim does that support, and what does it block?

**Response.** The subtractive pair supports a *computational-level* claim: the object is $p(\text{next}\mid\text{past})$ under a designed corpus, and the split lives in $P$. It *blocks* the claims that the architecture implements a moral sense, or that the split lives in $\theta$ as a value. Formal test: hold the transformer fixed, swap corpora, and attribute the change to $P$. Clinical test: same instrument, different training values, different output. The limit: a data-row result must not be promoted to an architecture-row or values claim.

### 7.

**Question.** Values grounded in need require [homeostasis](https://en.wikipedia.org/wiki/Homeostasis) or [allostasis](https://en.wikipedia.org/wiki/Allostasis). Where in this notebook is need absent — in the loss, in the probes, or in both?

**Response.** Need is absent in *both*: the loss is next-token negative log-likelihood, and the probes are wordings — neither contains a homeostatic or allostatic term the system must satisfy to persist. So there is nowhere in the pipeline for a need-grounded value to enter. Formal test: inspect the objective and the probe set for any viability variable — there is none. Clinical test: no physiological cost attaches to any completion. The limit: values grounded in need require a body the notebook does not model.

### 8.

**Question.** RLHF is not in the code. If a reward model were added, would the bars become values, or a second corpus?

**Response.** Adding a reward model would give a *second corpus*: preference data scored by the reward model, sharpening which completion is favoured. The bars would become a sharpened regularity, not a value: preference optimisation steers $p(\text{next}\mid\text{past})$ but cannot ground it in need. Formal test: treat RLHF as fitting to preference labels — still a statistic over completions. Clinical test: a rater’s thumb is not a need. The limit: fine-tuning can sharpen the split; it cannot make it a value.

### 9.

**Question.** Care versus fairness could be written as counts in a designed corpus. Why would that still be a count?

**Response.** Writing “care” versus “fairness” as frequencies in a designed corpus makes them *counts* in $P$; a count is not a moral foundation, however you label it. Encoding the words does not endow the model with the values the words name. Formal test: the model fits the frequencies, and balancing them removes the “foundation.” Clinical test: tallying value-words is not holding values. The limit: moral foundations, on the paper’s claim, are grounded in need and culture, not in corpus frequencies.

### 10.

**Question.** The remainder the paper names is embodied prediction under viability constraints. Which panel, if any, could show that remainder — and why the answer is none? ([Embodiment](https://plato.stanford.edu/entries/embodied-cognition/).)

**Response.** None. Panel A is a published human summary; panels B–D are designed text and next-token mass. Embodied prediction under viability constraints is not a bar, a probability, or a corpus — it is a condition on an organism, off the axis every panel measures. Formal test: look for a viability variable in any panel — there is none to plot. Clinical test: no figure here has a body that can fail. The limit: the remainder the paper names is exactly what the figure cannot display, which is why the answer is none.
