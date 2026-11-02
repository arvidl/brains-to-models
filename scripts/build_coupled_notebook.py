"""Write notebook 14d's teaching source; execute separately to save reference plots."""
from pathlib import Path
import copy

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
NAME = "14d_coupled_tumour_growth.ipynb"


def build():
    template = nb.read(ROOT / "code/notebooks/14b_brainweb_growth.ipynb", as_version=4)
    setup = next(c.source for c in template.cells if c.cell_type == "code")
    setup = setup.split("from brainsmodels import growth_lessons as lessons")[0]
    setup = setup.replace('"nibabel>=5.3,<6"]',
        '"nibabel>=5.3,<6", "scipy>=1.14,<2", "numba>=0.61,<1", "imageio-ffmpeg>=0.6,<1"]')
    setup += '''from dataclasses import replace
from IPython.display import Image, display
from brainsmodels import growth_coupled as lesson
OUTPUT = root / ".build/growth-14d"
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 95, "font.size": 10})
print("Solver source:", lesson.SOURCE_COMMIT)
print("Anatomy and histological illustration are bundled; no companion checkout is needed.")
'''
    cells = []
    def md(s): cells.append(nb.v4.new_markdown_cell(s.strip()))
    def code(s, **metadata): cells.append(nb.v4.new_code_cell(s.strip(), metadata=metadata))

    md(r'''# 14d — Tumour growth, oxygen and necrosis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/brains-to-models/blob/main/code/notebooks/14d_coupled_tumour_growth.ipynb)

**Related resources:** [Paper](https://github.com/arvidl/brains-to-models/blob/main/paper/main.pdf) · [Presentation guide](https://github.com/arvidl/brains-to-models/blob/main/talk/resources.md)

## Why this notebook

The slide demonstrations move from Fisher–KPP growth on normal BrainWeb anatomy
to a coupled model of viable tumour, oxygen and necrotic material. Here you can
reproduce that progression, change the seed and biological parameters, and view
all four fields at the same time and anatomical location.

The full numerical solver used by the slides is included at a recorded source
version, alongside the existing BrainWeb data. The histological illustration is
a visual guide to the biological questions. The simulated fields represent a
small selection of its processes.

### Learning goals

1. Explain the feedback between viable tumour, oxygen availability and necrosis.
2. Compare a coupled model with a reference sharing its seed, anatomy and transport parameters.
3. Separate numerical verification, assumed mechanisms and biological validation.
4. Reproduce a recorded preset and save its parameters, plots and animation.''')
    md(r'''## Concepts and two reading paths

<details open><summary><strong>For biomedical and imaging readers</strong></summary>

**Viable tumour** is a population density, not a segmentation label. **Oxygen**
is relative availability averaged over a voxel; a value of 0.1 is one tenth of
the stipulated normal baseline, not a blood oxygen saturation measurement.
**Necrotic material** is immobile tumour-cell equivalents. It occupies modelled
growth capacity and does not represent every form of damaged host tissue.

Low oxygen reduces births and can transfer viable cells into the necrotic
compartment. The perfusion rule assumes that accumulated tumour material
reduces oxygen delivery. Its coefficients are illustrative. BrainWeb supplies
normal geometry, rather than measured tumour, oxygen or vascular data.

</details>

<details><summary><strong>For computational readers</strong></summary>

The model couples two evolution equations to a quasi-steady elliptic oxygen
equation. Conservative harmonic face fluxes prevent periodic wrapping and
transport across closed tissue barriers. A diffusion positivity bound controls
the internal step. Exact frozen-oxygen substeps handle births, death transfer
and clearance; their composition is first order in time. A preconditioned
conjugate-gradient solve updates oxygen after each internal step and rejects
non-convergence. Numba compiles the kernels on the first run.

The original solver is bundled unchanged. The existing anatomy loader samples
integer grid spacings, preserving categorical labels. At integer sample
locations this matches the companion's grids; coarser sampling can remove thin
barriers. Changing grid spacing changes the represented geometry as well as
the discretization.

</details>''')
    md('''## Run in Colab or locally

In Colab, connect to a **CPU** runtime and choose **Runtime → Run all**. Setup
downloads this repository's shared code and data, installs the numerical
dependencies, and enables widgets. The presentation and companion repositories
are not downloaded. No GPU, API key, patient data or Drive mount is required.

Locally, use `uv sync --locked --extra dev`, then run this notebook with the
project kernel. The first simulation includes JIT compilation. The default
4-mm, 90-day run is smaller than the slide presets. The 2-mm comparison costs
more CPU time and memory; native 1-mm histories can use several GB.

Save a Colab copy in Drive to retain edits. Download exports before the temporary
runtime ends. GitHub displays the saved illustration and reference plots;
interactive controls require a running kernel.''')
    code(setup, tags=["setup"])
    md(r'''## 1. Start with the biological picture

Identify the invasion border, necrotic areas, vascular structures and different
cell types. Before running a model, predict which of these features could be
represented by one density field, and which require additional variables.''')
    code('display(Image(filename=str(lesson.ASSETS / "comba2021_figure1.png"), width=650))')
    md(r'''**Illustration:** Figure 1 from Comba et al. (2021), *Uncovering
Spatiotemporal Heterogeneity of High-Grade Gliomas*, Frontiers in Oncology
11:703764, [doi:10.3389/fonc.2021.703764](https://doi.org/10.3389/fonc.2021.703764).
© 2021 Comba and coauthors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The diagram and legend are reproduced from the slide's embedded image.

| Visual clue | Representation in this lesson |
|---|---|
| Invasion border | Effective population dispersal through an assigned tissue coefficient |
| Reduced oxygen and necrosis | Relative oxygen field and viable-to-necrotic transfer |
| Vascular structures | A stipulated perfusion rule; vessels and angiogenesis are not resolved |
| Immune cells, reactive astrocytes, perivascular niches and collective invasion | Additional biology to discuss; no explicit state variables here |

Neither model explicitly implements go-or-grow phenotype switching. A necrotic
field alone also does not establish that pseudopalisading structures have been
reproduced.''')
    md(r'''## 2. Write the equations and units

The reference is Fisher–KPP:

$$\partial_t c=\nabla\cdot(D\nabla c)+\rho c(1-c/K).$$

The coupled model uses viable density $c$, necrotic material $z$ and oxygen $o$:

$$\partial_t c=\nabla\cdot(D\nabla c)+\rho g(o)c[1-(c+z)/K]_+-\delta h(o)c,$$
$$\partial_t z=\delta h(o)c-\lambda_z z,$$
$$0=\ell_o^2\nabla^2o+p(1-o)-\beta(c/K)o,
\qquad p=\frac{1}{1+\alpha(c+z)/K}.$$

Here $[a]_+=\max(a,0)$, $g(o)=\min(1,o/\theta_p)$ and
$h(o)=[1-o/\theta_d]_+$, with $0<\theta_d<\theta_p<1$.

| Quantity | Meaning and units | Coupled slide value |
|---|---|---|
| $c,z,K$ | Viable cells, necrotic equivalents and capacity, per mm³ | $K=100000$ |
| $D_{WM},D_{GM}$ | Assigned scalar dispersal coefficients, mm²/day | 0.5, 0.1 |
| $\rho$ | Proliferation rate, day⁻¹ | 0.04 |
| $o,p$ | Relative oxygen and relative delivery, dimensionless | Computed |
| $\ell_o$ | Oxygen smoothing length, mm | 0.2 |
| $\beta,\alpha$ | Uptake and perfusion suppression, dimensionless | 4, 4 |
| $\theta_p,\theta_d$ | Oxygen proliferation/death thresholds | 0.3, 0.1 |
| $\delta,\lambda_z$ | Death-transfer and clearance rates, day⁻¹ | 0.5, 0 |

Grey matter, white matter and the glial class form the growth domain. CSF and
other classes are closed barriers; cells and oxygen have zero normal boundary
flux. Necrotic material is immobile and consumes no oxygen. $K$ limits births,
rather than imposing a hard packing constraint on incoming diffusion.

The two runs share initialization, grid, coefficients and transport domain.
Their reaction equations differ. All rates and thresholds are assigned teaching
values. At 2–4 mm, the 0.2-mm oxygen length and capillary structures remain
unresolved. No treatment, mass effect, fibre anisotropy or vascular dynamics is
included.''')
    md(r'''## 3. Check a limiting case before interpreting the anatomy

**Predict:** With zero oxygen uptake, initially normal oxygen and no necrotic
material, what should happen? The coupled calculation should recover the
matched Fisher–KPP run, with no necrosis. This checks implementation consistency.
It does not test whether the oxygen closure describes measured tumour biology.''')
    code('''brain_small = lesson.Brain(np.ones((9, 7, 5), np.float32),
    np.full((9, 7, 5), 3, np.uint8), np.eye(4), 1.)
p = replace(lesson.preset()[1], origin_mm=(4., 3., 2.), radii_mm=(2.,)*3,
            initial_density=20000., duration_days=2., frame_days=1., dt_days=.1,
            oxygen_uptake_ratio=0.)
limit = lesson.simulate(brain_small, p, compare=True)
np.testing.assert_allclose(limit.final_density, limit.comparison.final_density, rtol=3e-6)
np.testing.assert_array_equal(limit.final_necrosis, 0.)
np.testing.assert_allclose(limit.final_oxygen, 1., atol=1e-6)
print("PASS: zero uptake recovers the reference; oxygen remains normal; no necrosis.")''')
    md(r'''## 4. Compute a small anatomical comparison

The teaching preset uses a Gaussian seed centred at RAS+ $(-30,25,30)$ mm,
standard deviations $(6,6,6)$ mm, peak 80000 cells/mm³ and initial $z=0$.
The Gaussian is truncated at three standard deviations and restricted to the
growth domain. It uses 4-mm sampling, a maximum 0.5-day internal step and saved
states every 6 days through day 90. The controls below can switch to the slide
presets or original Fisher–KPP model.

**Predict:** Where should oxygen fall first? Will a viable-cell deficit relative
to Fisher–KPP necessarily equal the necrotic-cell count? Consider inhibited
births as well as death transfer.''')
    code('''spacing, parameters = lesson.preset("teaching")
brain, result = lesson.run_comparison(parameters, spacing=spacing)
print(json.dumps({"parameters": result.parameters, "numerical_record": result.metadata}, indent=2))
figure = lesson.plot_comparison(brain, result)
figure.savefig(OUTPUT / "comparison.png", dpi=120)
plt.show()
plt.close(figure)''')
    md('''## 5. Read the trajectories alongside the overlays

The viable and reference panels share a fixed 0–1 density scale. Necrosis uses
$z/K$ on its own fixed 0–1 scale. Oxygen uses the normal baseline, with colour
restricted to the tissue domain. Densities below 0.02 K are hidden for display;
they still contribute to integrated cell counts. Full 3-D metrics include every
voxel, independently of the displayed slices.

Cell counts follow from the specified $K$, density and voxel volume. A threshold
volume is a different summary. Neither is an observed MRI lesion volume.''')
    code('''figure = lesson.plot_metrics(result)
figure.savefig(OUTPUT / "trajectories.png", dpi=120)
plt.show()
plt.close(figure)
print("Final viable cells:", result.metrics["cells"][-1])
print("Final necrotic equivalents:", result.metrics["necrotic_cells"][-1])
print("Final reference cells:", result.comparison.metrics["cells"][-1])''')
    md(r'''## 6. Interactive experiment and slide presets

Use **Load preset**, then press **Run growth simulation**. Model parameters and
seed controls change the next calculation; frame, playback, slices, threshold
and opacity read cached histories. All four panels share the selected day and
physical slice positions. Titles report the nearest sampled coordinates. RAS+
increases rightward, anteriorly and superiorly; the views use neurological
orientation. Model changes are marked pending until the run button is pressed.

| Preset | Grid and duration | Purpose |
|---|---|---|
| CPU teaching | 4 mm, 90 days | Default interactive session |
| Slide comparison | 2 mm, 90 days | Coupled model plus matched Fisher–KPP reference |
| Slide Fisher–KPP | 1 mm, 180 days | Original growth demonstration with its recorded seed and parameters |

The presets record numerical settings from the companion used for the slides.
This notebook uses its exact bundled numerical source. Rendering differs from
the original movie layout. Coarse teaching results differ from the slide runs.

**Try these controlled interventions:**

1. Set $\beta=0$. Check recovery of the matched reference and absence of necrosis.
2. Restore $\beta=4$, then set $\alpha=0$. What changes when delivery no longer
   falls with tumour burden?
3. Change only $D_{GM}$ or $\rho$. Compare shape, viable cells and oxygen.
4. Keep a run fixed and change only threshold and slice positions. Which full
   volume metrics remain invariant?
5. Halve the internal step on the same grid. Then change grid spacing separately.
   Which comparison isolates time-stepping sensitivity?

To move the seed, set all three origin coordinates and radii under **Seed &
density**. Its centre must sample grey/white/glial tissue. Invalid parameters
are reported. Automatic substeps report the actual stable step; disabling them
lets you test rejection of an unstable request.''')
    code('''from brainsmodels.growth_coupled_widgets import coupled_explorer
explorer = coupled_explorer(initial_run=(brain, result), output_directory=OUTPUT / "interactive")
display(explorer.widget)''', tags=["interactive"])
    md('''## 7. Save a reproducible result

The default export below saves a poster and JSON record without recomputing.
The JSON includes parameters, actual numerical steps, source version, input
checksums, time points, full-volume metrics and view settings. The widget's
export button can also write a synchronized GIF or MP4; MP4 uses a bundled
FFmpeg executable. Animation rendering takes additional time. Download exports
before a Colab runtime ends.''')
    code('''lesson.export_run(brain, result, OUTPUT / "reference", planes="All three")
print("Reference files:", sorted(p.name for p in (OUTPUT / "reference").iterdir()))
# Optional: export the current interactive result without rerunning the solver.
# b, r = explorer.last_result
# lesson.export_run(b, r, OUTPUT / "movie", planes="Axial", animation="mp4")
# In Colab, download a ZIP after making your exports:
# import shutil
# from google.colab import files
# files.download(shutil.make_archive(str(OUTPUT), "zip", OUTPUT))''')
    md(r'''## 8. Interpret the model against the biological picture

Return to the illustration. The oxygen–necrosis feedback provides a reduced
mechanism for loss of viable cells in oxygen-poor regions. It supplies no direct
evidence for immune interactions, vessel remodelling, collective migration or
pseudopalisading organisation. Oxygen and density measurements, parameter
estimation and withheld biological outcomes would test additional claims.

### Exercises

1. Two runs have similar threshold volumes but different viable-cell counts.
   How could this occur?
2. With births and necrotic clearance set to zero, what total should be conserved?
3. Why is this coupled run's difference from Fisher–KPP not automatically an improvement?
4. Choose one omitted feature from the illustration. Propose one new variable,
   its coupling and a measurement that could test it.

<details><summary><strong>Show reasoning</strong></summary>

1. Thresholding counts selected voxels equally; integrating density weights
   every voxel, including low-density infiltration hidden by the display.
2. On a no-flux domain, the integral of $c+z$ should remain constant. Death
   transfers material between the two compartments. Clearance would remove it.
3. Additional assumptions produce different predictions; independent observations
   are needed to decide whether those assumptions improve the intended use.
4. For example, a vessel-density state could modify oxygen delivery and respond
   to a hypoxia signal. A measured vascular quantity and oxygen data would be
   needed; adding a variable alone supplies no validation.

</details>

**Takeaway:** richer biology increases the questions a model can express and the
evidence needed to assess its answers.

## Sources and reproduction

- [Pinned numerical source and licence](https://github.com/arvidl/brains-to-models/blob/main/code/src/brainsmodels/_vendor/brainweb_kpp/PROVENANCE.md).
- [Bundled illustration and preset provenance](https://github.com/arvidl/brains-to-models/blob/main/code/src/brainsmodels/data/growth/README.md).
- [BrainWeb input checksums, terms and attribution](https://github.com/arvidl/brains-to-models/blob/main/code/src/brainsmodels/data/brainweb/README.md).
- Comba et al. (2021), [original Figure 1 and article](https://doi.org/10.3389/fonc.2021.703764).

For independent checks of the coupled solver, views, exports and real widget
callbacks, run `make check-growth-coupled`. For the local notebook and Colab
bootstrap checks, see the repository's notebook guide.''')
    notebook = nb.v4.new_notebook(cells=cells, metadata=copy.deepcopy(template.metadata))
    notebook.metadata.pop("widgets", None)
    notebook.metadata["colab"]["name"] = NAME
    nb.validate(notebook)
    nb.write(notebook, ROOT / "code/notebooks" / NAME)


if __name__ == "__main__":
    build()
