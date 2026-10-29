# From Brains to Models: What AI Reveals About Being Human

A perspective paper and reproducible computational examples for the joint PRESIMAL × NORA HS research school at Sola, November 2026.

The paper connects representation, attention, memory, embodiment and consciousness with medical AI. The examples distinguish a model's successful performance from claims about biological mechanisms, understanding and clinical benefit. They are teaching resources, not clinical systems.

This publication copy contains the paper, computational material and final 30-minute [PowerPoint presentation](talk/From%20Brains%20to%20Models_20261029_1030.pptx) (see also this [[link](https://docs.google.com/presentation/d/e/2PACX-1vQYRZnUC5Ae8V7WGU5Gw1XjBR4-0Ez_DUImMWuV4mgNcOI_JADfLvSEEagQYJXCipluD4p232XmfuGo/pub?start=false&loop=false&delayms=3000)]).


## Start here

| Resource | Contents |
|---|---|
| [Paper PDF](paper/main.pdf) | Compiled perspective paper |
| [LaTeX source](paper/main.tex) | Main text, with [appendices](paper/appendices.tex) and [bibliography](paper/references.bib) |
| [Notebook guide and Colab links](code/notebooks/README.md) | Eighteen teaching examples with learning goals, definitions and guided interpretation |
| [Python implementation](code/src/brainsmodels) | Shared numerical models and figure builders |
| [Exercises](code/solutions/README.md) | Worked exercises and instructor resources |
| [Final presentation](talk/From%20Brains%20to%20Models_20261029_1030.pptx) | 25-slide PowerPoint deck for the 30-minute talk, 78.1 MiB |
| [Slide-to-resource guide](talk/resources.md) | All 25 slides, local examples and pinned external implementations |
| [August background paper](paper/background/august-perspective.pdf) | Section numbering used by notebooks 00–07, including their lettered companions |

This repository is public. Every notebook has an **Open in Colab** badge and runs on a standard CPU runtime. After dependencies are installed, the default numerical examples also run locally without network services, API keys, model downloads or patient data.

## Paper and provenance

The paper connects clinical interpretation and brain–model comparisons with numerical verification of AI-assisted tumor-growth code. 

## Run the notebooks

For a browser session, choose a notebook in the [notebook guide](code/notebooks/README.md), click **Open in Colab**, connect to a CPU runtime, and run the setup cell followed by the remaining cells. Setup downloads the shared Python code and figure references, excludes the large PPTX, and installs the numerical dependencies. Notebook 05b also enables Colab's interactive widget support. The optional external 3-D audit in notebook 13 requires a separate checkout; the default notebook is self-contained.

Each notebook includes learning goals, two reading paths (clinical/neuroscience and mathematical), definitions, equations, guided predictions and discussion of what the results establish. Notebooks 00–07 and their lettered companions provide the broader conceptual background; 08–13 give focused current-paper calculations. See the guide for direct Colab links to all eighteen.

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

### Editors, widgets and exercises

- **VS Code:** open the checkout, install the Python and Jupyter extensions, and select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows as the notebook kernel.
- **Interactive landscape:** notebook 05b uses `ipympl` and `ipywidgets`, included in the environment above. Restart the kernel if the widget does not appear. GIF export needs no additional system software; MP4 export requires `ffmpeg`.
- **Missing imports:** repeat `uv sync --locked --extra dev`, select the project kernel, and restart and run all cells. Check that `import sys; print(sys.executable)` points into the project's `.venv`.
- **Exercises:** expand a notebook's **Show solution** section, or load a solution with `%load ../solutions/01_1_drop_the_causal_mask.py`. See the [exercise guide](code/solutions/README.md) and [instructor discussion notes](code/notebooks/questions_responses.md).
- **Google Colab:** all eighteen notebooks have their own badge and setup. No GitHub authentication or Drive mount is needed to download this public repository. Save a notebook copy in Drive to retain edits, and download generated files before the temporary runtime ends. Setup reuses an existing runtime checkout; to obtain a fresh copy after repository updates, disconnect and delete the runtime, then reconnect and run again.

## Reproduce the paper and check the code

The lock file records the Python dependencies. A separate TeX installation providing `latexmk`, pdfLaTeX and BibTeX is required to compile the paper. The Python checks do not require TeX or presentation-authoring software.

```bash
uv sync --locked --extra dev
make figures PYTHON=.venv/bin/python
make paper
make check PYTHON=.venv/bin/python
```

`make all` runs figure generation, paper compilation and checks in that order. `make paper` can also compile directly from the committed figures. `make clean-paper` removes TeX intermediates while retaining the compiled PDF.

`make check` checks bibliography keys, cited figures, local documentation links, notebook structure and saved errors, numerical reference values, segmentation gradients and interventions, action-specific Q-learning, maze outcomes and planner controls, Transformer gradients, growth-model verification, generated teaching notes, and all 49,152 possible single-unit updates in the 12-unit Hopfield example. It also checks the final PPTX's checksum, byte size, package integrity and slide count against the resource manifest. Other presentation sources and development archives remain excluded; the final PPTX resides under `talk/`.

To execute notebooks afresh in separate kernels:

```bash
make check-notebooks PYTHON=.venv/bin/python
```

This requires the `dev` extra. It executes current-paper notebooks 08–13 in a temporary copy of the project and saves executed notebooks and generated outputs under ignored `.build/`. Use `.venv/bin/python scripts/check_notebooks.py --all` to execute all eighteen notebooks. To check a different notebook, pass its filename:

```bash
.venv/bin/python scripts/check_notebooks.py 01_tiny_transformer.ipynb
```

To exercise the Colab setup path in isolated local kernels:

```bash
.venv/bin/python scripts/check_notebooks.py --colab --all
```

This uses fresh sparse checkouts of a temporary snapshot, installs the declared notebook dependencies, and simulates Colab detection and widget-manager registration. It checks the bootstrap and all calculations; it does not test Google's hosted runtime or browser widget rendering. Results go to `.build/notebook-checks-colab/`. The ordinary notebook checks remain offline after environment setup.

## Scientific scope

The shortcut figure compares fixed analytic rules; no model was trained. The positive predictive value example holds sensitivity and specificity at 90%. The 12-unit Hopfield example has one stored pattern, symmetric weights and asynchronous updates; notebook 05 studies a different 32-unit, two-pattern model. The attention exercise uses hand-specified parameters and separates attention weights, gradients and occlusion effects.

Notebook 09 trains small NumPy networks on designed two-channel image phantoms. Whole images are held out from training and normalization. Annotation-policy agreement, input dependence and hidden-feature interventions are measured separately; changes in human interpretation are discussion questions, not measured results. These examples do not validate a clinical system or establish a theory of consciousness.

The framing and trolley examples train on designed text associations; their main probes reuse training-template prefixes. They do not measure unseen-dilemma generalization. The trolley figure contains no measured human choice rates. The spiking-versus-MLP example changes architecture, encoding and learning rule together, so its accuracy difference does not isolate update locality.

The T-maze compares two planners given a transition/observation model with a learner initially given zero action values. The reward-only planner demonstrates the instrumental value of information. This is not a comparison under equal prior knowledge or a test of biological active inference.

Notebook 13 uses independent one-dimensional and uniform-compartment references. It reports whether the optional pinned 3-D check was run. See the [resource guide](talk/resources.md#check-the-external-implementation) for that additional check and for the actual anatomical demonstrations.

## Rights and attribution

Code and original material retain the repository's [MIT license](LICENSE). The two reproduced glioma figures retain their article-specific CC BY 4.0 attributions in the paper. Third-party material retains its original rights and credit lines. Full reference-article PDFs and archived source presentations are not distributed in this publication copy.
