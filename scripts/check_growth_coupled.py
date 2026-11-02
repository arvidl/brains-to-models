"""Physical limits adapted from the pinned MIT-licensed companion tests.

See code/src/brainsmodels/_vendor/brainweb_kpp/PROVENANCE.md and LICENSE.
Also check offline notebook controls, synchronized views and media exports.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from brainsmodels import growth_coupled as lesson
from brainsmodels.growth_coupled import Brain
from brainsmodels._vendor.brainweb_kpp.model import Parameters, OxygenSolver, coupled_reaction, simulate, get_planes
from brainsmodels.growth_coupled_widgets import coupled_explorer

def cube(shape=(9, 7, 5)):
    return Brain(np.ones(shape, np.float32), np.full(shape, 3, np.uint8), np.eye(4), 1.)

def params(**kwargs):
    return replace(Parameters(model="coupled", origin_mm=(4., 3., 2.), radii_mm=(100.,) * 3,
                              seed_profile="ellipsoid", initial_density=80000., duration_days=2.,
                              frame_days=.5, dt_days=.1), **kwargs)

def test_uniform_oxygen_matches_supply_uptake_balance():
    b, p = cube(), params()
    mask = b.mask
    solver = OxygenSolver(mask, 1., p)
    c, z, k = np.full(mask.shape, .6), np.full(mask.shape, .2), np.ones(mask.shape)
    oxygen = solver.solve(c, z, k)
    supply = 1 / (1 + p.perfusion_suppression * .8)
    np.testing.assert_allclose(oxygen, supply / (supply + p.oxygen_uptake_ratio * .6), rtol=2e-7)
    np.testing.assert_array_equal(solver.solve(c * 0, z * 0, k), 1.)

def test_oxygen_no_flux_barrier_and_no_wrap():
    shape = (9, 4, 3)
    mask = np.ones(shape, bool)
    mask[4] = False
    c, z, k = np.zeros(shape), np.zeros(shape), np.ones(shape)
    c[:4] = 1
    solver = OxygenSolver(mask, 1., params(oxygen_length_mm=2.))
    oxygen = solver.solve(c, z, k)
    np.testing.assert_allclose(oxygen[:4], 1 / 21, rtol=2e-6)
    np.testing.assert_allclose(oxygen[5:], 1., rtol=2e-6)
    assert np.all(oxygen[4] == 0)

def test_oxygen_recovers_nonuniform_neumann_cosine_solution():
    b = cube((20, 5, 3))
    p = params(perfusion_suppression=0., oxygen_length_mm=.8, oxygen_rtol=1e-11)
    mode = np.broadcast_to(np.cos(np.pi * (np.arange(20) + .5) / 20)[:, None, None], b.labels.shape)
    exact = .6 + .05 * mode
    laplace = -4 * np.sin(np.pi / 40) ** 2 * .05 * mode
    c = (p.oxygen_length_mm ** 2 * laplace + 1 - exact) / (p.oxygen_uptake_ratio * exact)
    actual = OxygenSolver(b.mask, 1., p).solve(c, c * 0, np.ones(c.shape))
    np.testing.assert_allclose(actual, exact, rtol=1e-7)

def test_coupled_time_refinement_against_independent_uniform_rk4():
    # No spatial gradients: the oxygen equation has an algebraic solution.
    p = params(duration_days=3., frame_days=3., initial_density=80000.)
    def rhs(state):
        u, z = state
        supply = 1 / (1 + p.perfusion_suppression * (u + z))
        oxygen = supply / (supply + p.oxygen_uptake_ratio * u)
        births = p.rho * min(oxygen / p.oxygen_prolif_level, 1) * u * max(1 - u - z, 0)
        death = p.necrosis_rate * max(1 - oxygen / p.oxygen_death_level, 0) * u
        return np.array([births - death, death - p.necrosis_clearance * z])
    expected = np.array([.8, 0.])
    dt = .002
    for _ in range(round(p.duration_days / dt)):
        a = rhs(expected)
        b = rhs(expected + dt * a / 2)
        c = rhs(expected + dt * b / 2)
        d = rhs(expected + dt * c)
        expected += dt * (a + 2*b + 2*c + d) / 6
    errors = []
    for step in (.2, .025):
        r = simulate(cube(), replace(p, dt_days=step))
        observed = np.array([r.final_density.mean(), r.final_necrosis.mean()]) / p.carrying_capacity
        errors.append(np.linalg.norm(observed - expected))
    assert errors[1] < .35 * errors[0]
    assert errors[1] < .001

def test_anoxic_death_transfers_exactly_and_clearance_is_exponential():
    c, z = np.array([30000.]), np.array([10000.])
    coupled_reaction(c, z, np.array([0.]), np.array([0.]), np.array([100000.]),
                     np.array([0]), 3., .3, .1, .5, 0.)
    np.testing.assert_allclose(c, 30000 * np.exp(-1.5), rtol=1e-12)
    np.testing.assert_allclose(c + z, 40000., rtol=1e-12)
    before = z.copy()
    coupled_reaction(c, z, np.array([1.]), np.array([0.]), np.array([100000.]),
                     np.array([0]), 2., .3, .1, .5, .2)
    np.testing.assert_allclose(z, before * np.exp(-.4), rtol=1e-12)

def test_necrotic_material_occupies_growth_capacity():
    c, z = np.array([20000., 20000.]), np.array([40000., 120000.])
    coupled_reaction(c, z, np.ones(2), np.full(2, .2), np.full(2, 100000.),
                     np.arange(2), 10., .3, .1, 0., 0.)
    expected = 60000 / (1 + (60000 / 20000 - 1) * np.exp(-.2 * .6 * 10))
    np.testing.assert_allclose(c, [expected, 20000.], rtol=1e-12)

def test_zero_uptake_recovers_fisher_kpp_and_matching_reference():
    b = cube()
    p = params(oxygen_uptake_ratio=0., initial_density=20000., radii_mm=(2.,) * 3,
               d_csf=.5)
    progress = []
    coupled = simulate(b, p, compare=True, store_volumes=True,
                       progress=lambda done, total: progress.append(done / total))
    original = simulate(b, replace(p, model="fisher_kpp", d_csf=0.))
    np.testing.assert_allclose(coupled.final_density, original.final_density, rtol=3e-6)
    np.testing.assert_array_equal(coupled.final_necrosis, 0.)
    np.testing.assert_allclose(coupled.final_oxygen, 1., atol=1e-6)
    np.testing.assert_array_equal(coupled.comparison.final_density, original.final_density)
    assert progress[-1] == 1 and np.all(np.diff(progress) >= 0)
    assert len(coupled.oxygen_volumes) == len(coupled.times)

def test_hypoxic_simulation_conserves_live_plus_dead_without_births():
    r = simulate(cube(), params(rho=0.))
    assert r.metrics['necrotic_cells'][-1] > 0
    assert r.metrics['cells'][-1] < r.metrics['cells'][0]
    np.testing.assert_allclose(r.metrics['cells'] + r.metrics['necrotic_cells'],
                               r.metrics['cells'][0], rtol=2e-6)
    assert r.final_density.min() >= 0 and r.final_necrosis.min() >= 0
    assert r.final_oxygen.min() > 0 and r.final_oxygen.max() <= 1
    assert r.metadata['oxygen_solver']['solves'] == r.metadata['internal_steps'] + 1

def test_coupled_excludes_csf_even_with_custom_diffusion():
    b = cube()
    b.labels[5] = 1
    r = simulate(b, params(d_csf=2.), diffusion=np.ones(b.labels.shape))
    assert np.all(r.final_density[5] == 0)
    assert np.all(r.final_oxygen[5] == 0)
    assert np.all(r.final_necrosis[5] == 0)


def test_offline_widgets_and_exports():
    source = ROOT / "code/src/brainsmodels/_vendor/brainweb_kpp/model.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == lesson.SOURCE_SHA256
    assert hashlib.sha256((lesson.ASSETS / "comba2021_figure1.png").read_bytes()).hexdigest() == \
        "1e2b0a0433688a95feab09b7c342642df9d714bb558b5f2d1742344d48cff2fe"
    with tempfile.TemporaryDirectory(prefix="growth-14d-") as tmp, \
            patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")), \
            redirect_stdout(io.StringIO()):
        panel = coupled_explorer(output_directory=Path(tmp)/"widget")
        panel.section_control.value = 3
        assert panel.section_control.value == 3
        panel.section_control.value = 0
        panel.controls["duration_days"].value = 6
        panel.run_button.click()
        assert panel.error is None, panel.error
        pair = panel.last_result
        assert pair is not None
        brain, result = pair
        assert len(result.volumes) == len(result.oxygen_volumes) == len(result.comparison.volumes)
        assert brain.labels.shape == (46, 55, 46)
        for key, value in [("frame", 0), ("x", -38.), ("planes", "Axial"), ("threshold", .2)]:
            panel.controls[key].value = value
            assert panel.last_result is pair, "A display control reran the solver"
        panel.export_button.click()
        assert (Path(tmp)/"widget/comparison.png").exists(), panel.export_status.value
        record = json.loads((Path(tmp)/"widget/run.json").read_text())
        assert record["view"]["coordinates"][0] == -38.
        assert record["view"]["frame"] == 0 and record["view"]["day"] == 0
        panel.controls["oxygen_uptake_ratio"].value = 0
        assert panel.pending
        panel.run_button.click()
        assert panel.error is None
        r = panel.last_result[1]
        np.testing.assert_allclose(r.final_density, r.comparison.final_density, rtol=3e-6)
        panel.controls["model"].value = "fisher_kpp"
        panel.run_button.click()
        assert panel.error is None and panel.last_result[1].comparison is None
        panel.controls["initial_density"].value = 200000
        panel.run_button.click()
        assert panel.error and panel.last_result is None and not panel.run_button.disabled
        panel.preset_control.value = "slide-coupled"
        assert panel.controls["spacing"].value == 2 and panel.controls["dt_days"].value == .25
        assert panel.controls["duration_days"].value == 90 and panel.controls["initial_density"].value == 80000
        for kind in ("gif", "mp4"):
            folder = lesson.export_run(brain, result, Path(tmp)/kind, planes="Axial", animation=kind)
            assert (folder/f"comparison.{kind}").stat().st_size > 1000
        with Image.open(Path(tmp)/"gif/comparison.gif") as gif:
            assert gif.n_frames >= 2
    plt.close("all")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS:", name, flush=True)
