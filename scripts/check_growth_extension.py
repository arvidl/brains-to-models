"""Independent targets for the ND solver, observation operator and rate search."""
import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from brainsmodels.growth_extension import simulate, gaussian_seed, observe, parameter_search
from brainsmodels.brainweb_data import verify_files, load_brainweb
from brainsmodels.growth_verification import logistic


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("Invalid input was accepted")


def check_solver():
    rng = np.random.default_rng(7)
    for shape in [(9, 7), (9, 7, 5)]:
        h = (1.,) * len(shape)
        start = rng.uniform(.01, .3, shape)
        reaction = simulate(start, 0., .12, [0., 3.], spacing_mm=h, dt=.2)
        np.testing.assert_allclose(reaction.density[-1], logistic(start, .12, 1., 3.), atol=2e-14)
        moved = simulate(start, rng.uniform(.05, .4, shape), 0., [0., 3.], spacing_mm=h, dt=.1)
        np.testing.assert_allclose(moved.integral, moved.integral[0], rtol=1e-13)
        assert moved.density.min() >= start.min() - 1e-14
        assert moved.density.max() <= start.max() + 1e-14
        grown = simulate(start, .2, .12, [0., 3.], spacing_mm=h, dt=.1)
        np.testing.assert_allclose(grown.integral-grown.integral[0], grown.reaction_gain, atol=1e-12)
        # A closed internal plane must not leak; this is distinct from mass conservation.
        domain = np.ones(shape, bool)
        domain[4] = False
        start[4:] = 0.
        barrier = simulate(start, .4, 0., [0., 3.], spacing_mm=h, dt=.1, domain=domain)
        np.testing.assert_array_equal(barrier.density[-1, 4:], 0.)
        np.testing.assert_allclose(barrier.integral, barrier.integral[0], rtol=1e-13)
        spike = np.zeros(shape)
        spike[(0,)*len(shape)] = .5
        one_step = simulate(spike, .2, 0., [0., .1], spacing_mm=h, dt=.1)
        np.testing.assert_array_equal(one_step.density[-1, -1], 0.)
    # Independent cosine solution under reflecting boundaries: spatial refinement.
    errors = []
    for n in [12, 24]:
        h = 12/n
        x = (np.arange(n)+.5)*h
        c = .2 + .05*np.cos(np.pi*x[:, None]/12)*np.cos(np.pi*x[None, :]/12)
        exact = .2 + (c-.2)*np.exp(-.2*2*(np.pi/12)**2*2)
        run = simulate(c, .2, 0., [0., 2.], spacing_mm=(h, h), dt=.05*h*h)
        errors.append(float(np.linalg.norm(run.density[-1]-exact)/n))
    assert errors[1] < .35*errors[0], errors
    # Harmonic interface: all three rows begin alike; no y-flux. D_face=2*1*4/5=1.6.
    start = np.tile([1., 0., 0.], (3, 1))
    d = np.tile([1., 4., 4.], (3, 1))
    got = simulate(start, d, 0., [0., .01], spacing_mm=(1., 1.), dt=.01)
    np.testing.assert_allclose(got.density[-1], np.tile([.984, .016, 0.], (3, 1)), atol=1e-14)
    rejects(simulate, start, 1., 0., [0., 1.], spacing_mm=(1., 1.), dt=2.)
    rejects(simulate, start, np.nan, 0., [0., 1.], spacing_mm=(1., 1.))
    rejects(simulate, start, 0., -1., [0., 1.], spacing_mm=(1., 1.))
    rejects(simulate, start, 0., 0., [0., 0.], spacing_mm=(1., 1.))
    rejects(got.threshold_measure, np.nan)
    # Exact reaction retains zero density even when exponential decay underflows.
    start = np.zeros((3, 3))
    start[1, 1] = .2
    fast = simulate(start, 0., 10000., [0., 1.], spacing_mm=(1., 1.), dt=1.)
    np.testing.assert_array_equal(fast.density[-1], start > 0)
    rejects(observe, np.full((4, 4), np.nan))
    np.testing.assert_allclose(observe(np.arange(16.).reshape(4, 4)), [[2.5, 4.5], [10.5, 12.5]])
    # A 2x2x2 block at u=.25 has mean .25, regardless of its eight underlying entries.
    np.testing.assert_allclose(observe(np.full((4, 4, 4), .25)), .25)
    return errors


def check_search():
    axis = np.arange(13.)
    initial = gaussian_seed((axis, axis), (6., 6.), width=2., peak=.2)
    domain = np.ones(initial.shape, bool)
    generator = simulate(initial, .2, .12, [0., 2., 4., 8.], spacing_mm=(1., 1.), dt=.1)
    observations = np.stack([observe(u) for u in generator.density[1:3]])
    fit = parameter_search(initial, .2, [0., 2., 4.], observations,
                           [.5, 1., 2.], [0., .12, .24], spacing_mm=(1., 1.),
                           domain=domain, dt=.1)
    assert fit["scale"] == 1. and fit["rho"] == .12 and not fit["at_boundary"], fit
    forecast = simulate(initial, .2*fit["scale"], fit["rho"], [0., 8.], spacing_mm=(1., 1.), dt=.1)
    target = observe(generator.density[-1])
    error = np.mean((observe(forecast.density[-1])-target)**2)
    persistence_error = np.mean((observations[-1]-target)**2)
    assert error < 1e-20 and persistence_error > 1e-5
    # Extra (withheld) observations cannot be passed as if they matched the fit times.
    rejects(parameter_search, initial, .2, [0., 2., 4.],
            np.stack([observe(u) for u in generator.density[1:]]), [.5, 1.], [0., .12],
            spacing_mm=(1., 1.), domain=domain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainweb", action="store_true", help="Also load bundled anatomy and run its experiments")
    args = parser.parse_args()
    errors = check_solver()
    check_search()
    with tempfile.TemporaryDirectory() as temp:
        rejects(verify_files, temp)
        from brainsmodels.brainweb_data import FILES
        for name in FILES:
            (Path(temp)/name).write_bytes(b"invalid")
        rejects(verify_files, temp)
    if args.brainweb:
        from brainsmodels.growth_lessons import anatomical_experiment, recovery_experiment
        brain = load_brainweb(step_mm=4)
        assert brain.labels.shape == (46, 55, 46)
        np.testing.assert_allclose(brain.affine @ [1., 1., 1., 1.], [-86., -122., -68., 1.])
        _, runs, _ = anatomical_experiment(brain)
        for run in runs.values():
            assert run.balance_error < 1e-10
            np.testing.assert_array_equal(run.density[:, ~brain.domain], 0.)
            assert np.all(run.threshold_measure(.05) >= run.threshold_measure(.2))
        result = recovery_experiment(load_brainweb(step_mm=6))
        s = result["summary"]
        assert s["fit_days"] == [20., 40., 60.] and s["withheld_day"] == 90.
        assert s["withheld_rmse"] < s["persistence_withheld_rmse"] / 5, s
        assert abs(s["estimated_scale"]-1.) <= .1 and abs(s["estimated_rho_per_day"]-.04) <= .005, s
    print("PASS: 2-D/3-D logistic limits, conservation, closed faces, no wraparound, "
          "harmonic interface, refinement, stability rejection, block observations, "
          f"known-rate recovery and a withheld forecast. Cosine errors: {errors}")
    if args.brainweb:
        print("PASS: pinned BrainWeb checksums/geometry, physical coordinates, tissue domain and generated-data forecast.")


if __name__ == "__main__":
    main()
