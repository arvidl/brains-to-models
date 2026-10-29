# Resources for the 30 minute presentation

This guide follows the 25 slides in the [final PowerPoint presentation](From%20Brains%20to%20Models_20261029_1030.pptx), *From Brains to Models_20261029_1030.pptx*.
It connects the presentation to runnable examples and identifies which results
come from separate research repositories. The [resource manifest](resources.json)
records the included PPTX's path, byte size, checksum and slide count, along with
the companion source versions. The presentation can be downloaded from this public repository.

## Slides and supporting material

| Slides | Subject | Resource and suggested use |
|---|---|---|
| 1–2 | Introduction and motivation | [Perspective paper](../paper/main.pdf). Read its opening discussion of performance, mechanism and clinical interpretation. |
| 3–4 | Brains and engineered models | [00: timeline and Marr](../code/notebooks/00_intro_timeline.ipynb), [01: tiny Transformer](../code/notebooks/01_tiny_transformer.ipynb). [03: two learning systems](../code/notebooks/03_bio_vs_artificial.ipynb) is supplementary; its architectures and learning rules differ together. |
| 5–6 | Four views of AI | [04b: Russell–Norvig grid](../code/notebooks/04b_four_views_of_AI.ipynb). Use the cells as evaluation questions. The catalog supplies no measured verdict on a model. |
| 7–9 | Turing-like tests and model selection | [Zhang et al., published study](https://www.nature.com/articles/s41562-026-02590-6), [authors' data and code](https://kreimanlab.com/code/testing-turing-tests/). Distinguish passive judgments from interactive questioning. The paper includes Claude 3.7 and LLaMA 3.3 conversation experiments as well as GPT-4o; model versions vary by experiment. The AI-generated suggestion shown on slide 9 is a proposal, not an evaluated comparison. |
| 10 | Medical AI workflow | [Paper](../paper/main.pdf) and [11: shortcut dependence](../code/notebooks/11_shortcut_dependence.ipynb). The latter compares fixed analytic rules under constructed shifts; no clinical model is trained. |
| 11 | Marr's levels | [00: levels of explanation](../code/notebooks/00_intro_timeline.ipynb). Keep the task fixed when describing purpose, algorithm and implementation. |
| 12–15 | Working and learning with AI | [Co-Existence](https://co-existence.ai), [Bastani et al., learning experiment](https://doi.org/10.1073/pnas.2422633122), [13: verified simulation](../code/notebooks/13_verified_growth_model.ipynb). The 17% reduction concerns the unrestricted-assistance group in a high-school mathematics study; safeguards largely mitigated that effect. Distinguish goal-directed software behavior from claims about self-originated human motivation. |
| 16 | Glioma compartments and anatomy | [Published feasibility study](https://doi.org/10.1080/03091902.2026.2713703), [versioned research implementation](https://github.com/arvidl/high-precision-glioma-segmentation/tree/54d50b76e487183d82ab72d5630a1499e00f43d4), [09: synthetic segmentation exercise](../code/notebooks/09_segmentation_reflection.ipynb). The slide's compartment illustration uses reference masks. Notebook 09 does not reproduce SegResNet inference or the study's patient results. |
| 17 | Fisher–KPP and go-or-grow | [13: equations, units and verification](../code/notebooks/13_verified_growth_model.ipynb), [14a: density and thresholds](../code/notebooks/14a_growth_density_and_thresholds.ipynb), [Hatzikirou et al.](https://doi.org/10.1093/imammb/dqq011), [Everett et al.](https://doi.org/10.3934/mbe.2020207). Fisher–KPP is a population reaction–diffusion model; it has no explicit migratory/proliferative state switching. Treat go-or-grow as a separate mechanistic hypothesis. |
| 18 | Interactive growth demonstration | [14b: self-contained interactive BrainWeb lesson](../code/notebooks/14b_brainweb_growth.ipynb), with bundled anatomy and local NumPy code. The slide's original demonstration comes from the [pinned external BrainWeb explorer](https://github.com/arvidl/brainweb_fisher_kpp/blob/81ecd42e6f37692a711208caa8d24361904f7e65/tumor_growth_explorer.ipynb). The normal BrainWeb anatomy is a phantom; tumor growth is simulated with assumed parameters. |
| 19–20 | Tumor, oxygen and necrosis | [Pinned coupled example](https://github.com/arvidl/brainweb_fisher_kpp/blob/81ecd42e6f37692a711208caa8d24361904f7e65/coupled_example.ipynb), [model source](https://github.com/arvidl/brainweb_fisher_kpp/blob/81ecd42e6f37692a711208caa8d24361904f7e65/src/brainweb_kpp/model.py), [verification notebook](../code/notebooks/13_verified_growth_model.ipynb). The coupled closure and its parameters are illustrative; numerical agreement in a limiting case does not establish physiological validity. |
| 21–22 | Agentic research and reproducibility | [Nature PrimeView](https://doi.org/10.1038/s43586-026-00542-y), [versioned glioma research workflow](https://github.com/arvidl/AI-assisted-glioma-research/tree/b9c1b512d93426392db6c0571f47f1b54ff05eec), [13: verification record](../code/notebooks/13_verified_growth_model.ipynb). The local [14c: generated-data recovery](../code/notebooks/14c_growth_parameter_recovery.ipynb) adds a withheld forecast. Separate generated hypotheses, computational checks and empirical validation. |
| 23–24 | Creative example and discussion | [Knudsen's original post and video](https://www.molekyl.io/p/m030-something-has-changed), [Molekyl](https://www.molekyl.io). These are attributed external creative works, not repository experiments. |
| 25 | Code and further exploration | [Repository starting points](../README.md#start-here). For this talk, start with 00, 04b, 09 and 13; continue with [14a–14c](../code/notebooks/README.md#interactive-growth-controls) for the interactive growth sequence. Use the remaining examples according to the question being studied. |

## Run the local verification notebook

From this checkout, install the locked environment and start Jupyter:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev jupyter lab code/notebooks/13_verified_growth_model.ipynb
```

The default run needs no network, API key, model weights or MRI data after
installation. It checks an independent one-dimensional reference against the
analytic logistic solution, no-flux conservation, blocked faces, a reflecting
boundary counterexample and spatial/time refinement. A uniform-compartment
coupled limit is checked separately from the three-dimensional oxygen PDE.
Results and figures are written under ignored `.build/growth-verification/`.

## Check the external implementation

This optional audit checks the slides' original external solver. The local
14a–14c lessons run independently and do not need this checkout.

Use a separate checkout at the pinned commit; retain its own locked environment.
Create it outside the original development repository. These commands create
a new checkout rather than changing an existing project:

```bash
git clone https://github.com/arvidl/brainweb_fisher_kpp.git ../brainweb-kpp-reference
git -C ../brainweb-kpp-reference checkout --detach 81ecd42e6f37692a711208caa8d24361904f7e65
uv sync --project ../brainweb-kpp-reference --locked
../brainweb-kpp-reference/.venv/bin/python scripts/check_growth.py --reference-checkout ../brainweb-kpp-reference
```

The last command verifies the model file's SHA-256 before import and checks
the actual 3-D solver on synthetic arrays. Bytecode and JIT caches stay outside
the reference checkout. It does not read anatomy or patient records. These
checks are additional to the default local build, and require access to the
separate repository and its dependencies.

To run the notebook's optional section, use the reference project's Python
environment and set `BRAINWEB_KPP_ROOT` to that checkout's absolute path before
starting Jupyter. A default notebook run explicitly records this section as
**not run**. The original, more extensive coupled demonstrations use the
external notebooks and their own data manifest and instructions. The local
Fisher–KPP examples in 14b/14c use this repository's bundled anatomy.

## Supplementary material

[08](../code/notebooks/08_attention_maps.ipynb) and
[10](../code/notebooks/10_two_key_attention.ipynb) distinguish attention weights
from input sensitivity and output contributions.
[12](../code/notebooks/12_hopfield_12_units.ipynb) reproduces the paper's exact
associative-memory calculation. Predictive coding, multistability, spatial
gating, global workspace and values notebooks support the broader paper;
they are not prerequisites for this presentation.

[15: PPV and decisions](../code/notebooks/15_ppv_and_decisions.ipynb) connects
the paper's expected diagnostic counts to prevalence-dependent predictive value
and a stipulated two-action cost model, with separate population and decision controls.

The resource guide records interpretations that should accompany the existing
slides. It does not change the presentation or establish access rights to
third-party sources. Repository visibility remains controlled separately.

## Optional anatomical and inference lessons

Notebooks [14a](../code/notebooks/14a_growth_density_and_thresholds.ipynb), [14b](../code/notebooks/14b_brainweb_growth.ipynb) and [14c](../code/notebooks/14c_growth_parameter_recovery.ipynb) extend the growth example without changing the presentation. They follow the same two reading paths as the other notebooks. The first uses constructed 2-D tissue; the others read the bundled normal BrainWeb phantom, with checksums and physical geometry verified. All three offer local/Colab widget controls; no companion repository or data download is needed for 14a–14c. Tumour density and noisy observations are generated separately from anatomy. These compact examples use an independent scalar-diffusivity solver, rather than reproducing the full coupled external implementation.

The sequence draws on the [Shandong companion notebooks and presentation](https://github.com/arvidl/AI-assisted-glioma-research/tree/b9c1b512d93426392db6c0571f47f1b54ff05eec). In 14c the known tissue ratio and seed are strong assumptions; estimates from one generated noise realization are not patient parameters or uncertainty intervals. See the [paper's growth section](../paper/main.pdf) for the corresponding derivation and figures.
