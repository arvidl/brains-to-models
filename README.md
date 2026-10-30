# From Brains to Models: What AI Reveals About Being Human

A perspective paper and reproducible computational examples for the joint PRESIMAL × NORA HS research school at Sola, November 2026.

The paper connects representation, attention, memory, embodiment and consciousness with medical AI. The examples distinguish a model's successful performance from claims about biological mechanisms, understanding and clinical benefit. They are teaching resources, not clinical systems.

This publication copy contains the paper, computational material and final 30-minute [PowerPoint presentation](talk/From%20Brains%20to%20Models_20261029_1030.pptx) (see also this [[link](https://docs.google.com/presentation/d/e/2PACX-1vQYRZnUC5Ae8V7WGU5Gw1XjBR4-0Ez_DUImMWuV4mgNcOI_JADfLvSEEagQYJXCipluD4p232XmfuGo/pub?start=false&loop=false&delayms=3000)]).


## Start here

| Resource | Contents |
|---|---|
| [Paper PDF](paper/main.pdf) | Compiled paper, including the 25-page slide companion in Appendix D |
| [LaTeX source](paper/main.tex) | Main text, with [appendices](paper/appendices.tex) and [bibliography](paper/references.bib) |
| [Notebook guide and Colab links](code/notebooks/README.md) | Twenty-two teaching examples with learning goals, definitions and guided interpretation |
| [Python implementation](code/src/brainsmodels) | Shared numerical models and figure builders |
| [Exercises](code/solutions/README.md) | Worked exercises and instructor resources |
| [Final presentation](talk/From%20Brains%20to%20Models_20261029_1030.pptx) | 25-slide PowerPoint deck for the 30-minute talk, 78.1 MiB |
| [Slide-to-resource guide](talk/resources.md) | All 25 slides, local examples and pinned external implementations |

This repository is public. Every notebook has an **Open in Colab** badge and runs on a standard CPU runtime. After dependencies are installed, the examples run on CPU without API keys, model downloads or patient data. The two normal BrainWeb phantom volumes used by 14b and 14c are bundled (about 9.25 MB). After checkout and dependency setup, all default calculations run offline, independently of the companion repositories.

## Paper

The paper connects clinical interpretation and brain–model comparisons with numerical verification of AI-assisted tumor-growth code. A short [provenance note](paper/background/README.md) identifies the preserved background manuscript.

The [reader's guide](paper/main.pdf#page=2) offers two reading paths through a shared core:

- **Biomedical and imaging:** focus on clinical observations, measurements and interpretation.
- **Computational:** trace variables, update rules, numerical checks and worked derivations.

Four paired perspective boxes connect these approaches in growth modelling, attention and image overlays, associative retrieval, and predictive value and decisions. Both paths retain the assumptions, takeaways and check questions. Readers can switch paths as their interests and preparation suggest.

Appendix D is a [25-page slide companion](paper/main.pdf#page=35), with one page
per slide. Each page starts with an unchanged thumbnail and a short title,
followed by **Biomedical and imaging path** and **Computational path** explanations.
It serves as extended presenter notes and a syllabus route through the
[presentation](talk/resources.md) and notebooks. Its [LaTeX source](paper/appendix-slides.tex)
is editable separately from the main text.

## Run the notebooks

For a browser session, choose a notebook in the [notebook guide](code/notebooks/README.md), click **Open in Colab**, connect to a CPU runtime, and run the setup cell followed by the remaining cells. Setup downloads the shared Python code and figure references, excludes the large PPTX, and installs the numerical dependencies. Notebooks 05b, 14a–14c and 15 also enable Colab's interactive widget support. The optional external 3-D audit in notebook 13 requires a separate checkout; the default notebook is self-contained.

**Colab verified — 29–30 October 2026:** All twenty-two notebooks completed their default runs on hosted CPU runtimes: 00–14c on 29 October and 15 on 30 October. Verification included representative widget checks, 05b’s full-screen mouse interactions, notebook 15’s independent population and decision controls, and bundled BrainWeb data loading. See [verification results and scope](code/notebooks/README.md#colab-verification).

Each notebook includes learning goals, two reading paths (biomedical/imaging and computational), definitions, equations, guided predictions and discussion of what the results establish. Notebooks 00–07 and their lettered companions develop the broader concepts; 08–13 focus on attention, segmentation, shortcut dependence, memory and numerical verification; 14a–14c extend growth modelling into tissue-guided spread, BrainWeb anatomy and generated-data parameter recovery. Notebook 15 connects expected diagnostic counts and PPV to explicit decision costs. See the guide for direct Colab links to all twenty-two.

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). Clone this repository using your GitHub access, then change to the checkout directory containing `pyproject.toml` and `uv.lock`:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev jupyter lab
```

In JupyterLab, open a notebook under `code/notebooks`, choose the project's Python kernel, and run the setup cell first. Use **Kernel → Restart Kernel and Run All Cells** to reproduce a notebook's default calculation. Each notebook is independent; earlier notebooks do not need to run first.

Keep `code/src` with the notebooks when running locally. Setup cells locate the shared implementation from the repository root or a subdirectory and set the working directory to `code/notebooks` for consistent exercise paths. In Colab, setup obtains the shared files automatically. Some notebooks regenerate `paper/figures`, and notebook 05b can replace animations under `code/animations`; inspect `git status` after experiments.

| Notebook | Topic |
|---|---|
| [00](code/notebooks/00_intro_timeline.ipynb) | Timeline and levels of explanation |
| [01](code/notebooks/01_tiny_transformer.ipynb) | Tiny transformer |
| [02](code/notebooks/02_predictive_coding.ipynb) / [02b](code/notebooks/02b_active_inference_vs_rl.ipynb) | Predictive coding / information seeking with known-model and Q-learning controls |
| [03](code/notebooks/03_bio_vs_artificial.ipynb) | Spiking and MLP systems with different architectures and update rules |
| [04](code/notebooks/04_quadrants_biases.ipynb) / [04b](code/notebooks/04b_four_views_of_AI.ipynb) | Framing / four views of AI |
| [05](code/notebooks/05_multistable_dynamics.ipynb) / [05b](code/notebooks/05b_multistable_animation.ipynb) / [05c](code/notebooks/05c_analog_brain_waves.ipynb) | Multistability / landscape animation / analog waves |
| [06](code/notebooks/06_global_workspace.ipynb) | Global workspace |
| [07](code/notebooks/07_values_probing.ipynb) | Values and framing |
| [08](code/notebooks/08_attention_maps.ipynb) | Attention, input sensitivity and occlusion maps |
| [09](code/notebooks/09_segmentation_reflection.ipynb) | Segmentation, annotation policies, overlays and learned features |
| [10](code/notebooks/10_two_key_attention.ipynb) | Exact two-key attention calculation and value/query interventions |
| [11](code/notebooks/11_shortcut_dependence.ipynb) | Analytic shortcut dependence and a constructed binary dataset |
| [12](code/notebooks/12_hopfield_12_units.ipynb) | Exact 12-unit, one-pattern Hopfield recall and energy checks |
| [13](code/notebooks/13_verified_growth_model.ipynb) | Analytic, conservation and refinement checks for a growth model; optional pinned 3-D solver audit |
| [14a](code/notebooks/14a_growth_density_and_thresholds.ipynb) | Density, tissue-dependent spread and the effect of display thresholds |
| [14b](code/notebooks/14b_brainweb_growth.ipynb) | Three-dimensional Fisher–KPP growth on the normal BrainWeb phantom |
| [14c](code/notebooks/14c_growth_parameter_recovery.ipynb) | Generated observations, two-parameter recovery and a withheld forecast |
| [15](code/notebooks/15_ppv_and_decisions.ipynb) | Expected diagnostic counts, prevalence-dependent PPV and decisions under stipulated costs |

### Editors, widgets and exercises

- **VS Code:** open the checkout, install the Python and Jupyter extensions, and select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows as the notebook kernel.
- **Interactive landscape:** notebook 05b uses `ipympl` and `ipywidgets`, included in the environment above. Restart the kernel if the widget does not appear. GIF export needs no additional system software; MP4 export requires `ffmpeg`.
- **Missing imports:** repeat `uv sync --locked --extra dev`, select the project kernel, and restart and run all cells. Check that `import sys; print(sys.executable)` points into the project's `.venv`.
- **Exercises:** expand a notebook's **Show solution** section, or load a solution with `%load ../solutions/01_1_drop_the_causal_mask.py`. See the [exercise guide](code/solutions/README.md) and [instructor discussion notes](code/notebooks/questions_responses.md).
- **Google Colab:** all twenty-two notebooks have their own badge and setup. No GitHub authentication or Drive mount is needed to download this public repository. Save a notebook copy in Drive to retain edits, and download generated files before the temporary runtime ends. Setup reuses an existing runtime checkout; to obtain a fresh copy after repository updates, disconnect and delete the runtime, then reconnect and run again.

### Interactive growth lessons

Notebooks 14a–14c use `ipywidgets` in Colab, JupyterLab and Cursor. Run the notebook, choose simulation/fitting parameters, then press its run button. In 14a/14b, time, threshold and image-view controls read cached states without recomputing. Pending model changes are identified, and an unstable requested step is rejected with its permitted bound. Fixed examples and paper figures remain independent of widget state.

- **14a:** white/grey diffusivities, growth rate, numerical step, time and density threshold.
- **14b:** those parameters plus 2/4/6 mm anatomical sampling, orthogonal RAS+ slice positions and overlay opacity.
- **14c:** generating rates, noise and seed, training times, candidate-grid resolution and fitting step. Day 90 stays withheld; persistence uses the latest selected training observation.

GitHub displays saved reference plots. Open a live notebook to use controls; if a widget does not appear, rerun setup and the interactive cell. `make check-growth-widgets` exercises callbacks, cached views, rejection of unstable steps, selected-time baselines and protection of withheld observations, with network access blocked.

## Reproduce the paper and check the code

The lock file records the Python dependencies. A separate TeX installation providing `latexmk`, pdfLaTeX and BibTeX is required to compile the paper. The Python checks do not require TeX or presentation-authoring software.

```bash
uv sync --locked --extra dev
make figures PYTHON=.venv/bin/python
make paper
make check PYTHON=.venv/bin/python
```

`make all` runs figure generation, paper compilation and checks in that order. `make paper` compiles directly from the committed figures; it does not execute Python or load anatomy. `make growth-figures` rebuilds just the 14a–14c figures and their computed number macros, using the same teaching implementation as the notebooks and the two checksum-verified normal-phantom volumes bundled with the package. `make clean-paper` removes TeX intermediates while retaining the compiled PDF.

`make check` checks bibliography keys, cited figures, local documentation links, notebook structure and saved errors, numerical reference values, segmentation gradients and interventions, action-specific Q-learning, maze outcomes and planner controls, Transformer gradients, growth-model verification, bundled BrainWeb checksums, widget callbacks with network access blocked, generated teaching notes, and all 49,152 possible single-unit updates in the 12-unit Hopfield example. It also checks the final PPTX's checksum, byte size, package integrity and slide count against the resource manifest. Other presentation sources and development archives remain excluded; the final PPTX resides under `talk/`.

To execute notebooks afresh in separate kernels:

```bash
make check-notebooks PYTHON=.venv/bin/python
```

This requires the `dev` extra. It executes current-paper notebooks 08–15 in a temporary copy of the project and saves executed notebooks and generated outputs under ignored `.build/`. Use `.venv/bin/python scripts/check_notebooks.py --all` to execute all twenty-two notebooks. To check a different notebook, pass its filename:

```bash
.venv/bin/python scripts/check_notebooks.py 01_tiny_transformer.ipynb
```

To exercise the Colab setup path in isolated local kernels:

```bash
.venv/bin/python scripts/check_notebooks.py --colab --all
```

This uses fresh sparse checkouts of a temporary snapshot, installs the declared notebook dependencies, and simulates Colab detection and widget-manager registration. It checks the bootstrap and all calculations; it does not test Google's hosted runtime or browser widget rendering. Results go to `.build/notebook-checks-colab/`. Notebook calculations remain offline after environment setup. Fresh Colab snapshots include the bundled BrainWeb data; no companion checkout or cached data folder is needed.

## Scientific scope

The shortcut figure compares fixed analytic rules; no model was trained. The positive predictive value example holds sensitivity and specificity at 90%. The 12-unit Hopfield example has one stored pattern, symmetric weights and asynchronous updates; notebook 05 studies a different 32-unit, two-pattern model. The attention exercise uses hand-specified parameters and separates attention weights, gradients and occlusion effects.

Notebook 09 trains small NumPy networks on designed two-channel image phantoms. Whole images are held out from training and normalization. Annotation-policy agreement, input dependence and hidden-feature interventions are measured separately; changes in human interpretation are discussion questions, not measured results. These examples do not validate a clinical system or establish a theory of consciousness.

The framing and trolley examples train on designed text associations; their main probes reuse training-template prefixes. They do not measure unseen-dilemma generalization. The trolley figure contains no measured human choice rates. The spiking-versus-MLP example changes architecture, encoding and learning rule together, so its accuracy difference does not isolate update locality.

The T-maze compares two planners given a transition/observation model with a learner initially given zero action values. The reward-only planner demonstrates the instrumental value of information. This is not a comparison under equal prior knowledge or a test of biological active inference.

Notebook 13 uses independent one-dimensional and uniform-compartment references. It reports whether the optional pinned 3-D check was run. See the [resource guide](talk/resources.md#check-the-external-implementation) for that additional check and for the full external coupled demonstrations. Notebooks 14a–14c add a compact independent NumPy solver: constructed 2-D tissue experiments, 3-D BrainWeb-phantom growth, and synthetic two-parameter recovery. They do not reproduce the full external oxygen/necrosis model or its uncertainty audits.

### Growth lessons and provenance

The 14a–14c sequence adapts selected questions from the [Shandong companion material](https://github.com/arvidl/AI-assisted-glioma-research/tree/b9c1b512d93426392db6c0571f47f1b54ff05eec) and the [BrainWeb explorer](https://github.com/arvidl/brainweb_fisher_kpp/tree/81ecd42e6f37692a711208caa8d24361904f7e65). The local solver is a small independent educational implementation. The teaching questions retain source attribution; the runtime uses only this repository’s code and included data. The full external coupled solver remains a separate reference.

Normal phantom anatomy, assigned simulated tumour density and generated noisy observations are identified separately. Tissue diffusivities are locally scalar; no dMRI-derived direction field is used. In 14c the seed, tissue ratio and observation operator are known, and day 90 is withheld from fitting. Grid-search ranges do not constitute confidence intervals.

Run `make check-growth` for offline numerical checks and `make check-growth-brainweb` to include pinned-data geometry, anatomical runs and the generated-data forecast. Set `BRAINSMODELS_BRAINWEB_DIR` to a matching existing data folder to read it without writing there. Missing or altered bundled files raise an error; there is no downloader or substitute anatomy. Input hashes, source versions, coordinates, settings and computed results are recorded in [the growth figure record](paper/figures/growth_extension_values.json).

## Rights and attribution

Code and original material retain the repository's [MIT license](LICENSE). The two reproduced glioma figures retain their article-specific CC BY 4.0 attributions in the paper. BrainWeb data retain their original terms and attribution ([BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/); Collins et al. 1998; Cocosco et al. 1997); the two checksum-verified input NIfTI files are included under [the package data directory](code/src/brainsmodels/data/brainweb/README.md), with source aliases and attribution. They are also included when building a Python wheel. The generated anatomical figure carries attribution in the paper. Third-party material retains its original rights and credit lines. Full reference-article PDFs and archived source presentations are not distributed in this publication copy.
