"""Independent, one-dimensional numerical checks for the talk's growth model.

This small NumPy reference is for verification exercises. It is not the
BrainWeb 3-D implementation and does not reproduce an anatomical result.
Length is mm, time is days, density is cells/mm^3.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REFERENCE_COMMIT = "81ecd42e6f37692a711208caa8d24361904f7e65"
REFERENCE_MODEL_SHA256 = "2c6da092c0b4e4eb6871094b3826485ba45b6887452094aa2bae17c6ba316e50"


def logistic(initial, rho, capacity, time):
    """Exact solution of dc/dt = rho*c*(1-c/K), including c=0."""
    decay = np.exp(-rho * time)
    initial = np.asarray(initial, dtype=float)
    return initial / (decay + initial / capacity * (1 - decay))


def diffuse(initial, diffusion, dx, dt):
    """Conservative cell-centered flux, zero flux on both boundary faces."""
    initial = np.asarray(initial, dtype=float)
    d = np.broadcast_to(diffusion, initial.shape)
    a, b = d[:-1], d[1:]
    face_d = np.divide(2 * a * b, a + b, out=np.zeros_like(a), where=a+b > 0)
    rates = np.zeros_like(initial)
    rates[:-1] += face_d / dx**2
    rates[1:] += face_d / dx**2
    if dt * rates.max() > 1 + 1e-12:
        raise ValueError("Explicit diffusion step violates the positivity bound.")
    flux = np.zeros(len(initial) + 1)
    flux[1:-1] = face_d * np.diff(initial) / dx
    return initial + dt * np.diff(flux) / dx


def evolve(initial, *, length=20., diffusion=.2, rho=.04,
           capacity=100000., duration=2., dt=.05):
    """First-order Lie splitting: explicit diffusion, exact logistic growth."""
    c = np.asarray(initial, dtype=float).copy()
    if c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Use a finite nonnegative one-dimensional density.")
    if not all(np.isfinite(v) for v in (length, rho, capacity, duration, dt)):
        raise ValueError("Parameters must be finite.")
    if min(length, capacity, dt) <= 0 or min(rho, duration) < 0:
        raise ValueError("Use positive length, capacity and step; nonnegative rate and duration.")
    d = np.broadcast_to(diffusion, c.shape)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Diffusion must be finite and nonnegative.")
    dx = length / len(c)
    steps = max(1, int(np.ceil(duration / dt)))
    actual_dt = duration / steps
    for _ in range(steps):
        c = logistic(diffuse(c, d, dx, actual_dt), rho, capacity, actual_dt)
    return c


def uniform_coupled(initial, *, duration=2., dt=.05, rho=.04,
                    capacity=100000., uptake=0., perfusion_suppression=4.,
                    death_rate=.5, clearance=0.):
    """One uniform compartment, algebraic oxygen balance, no oxygen gradients.

    Proposed reduced closure, not a physiological calibration. Frozen-oxygen
    growth, death transfer and clearance are composed at each time step.
    """
    c, z = float(initial), 0.
    if min(initial, duration, rho, uptake, perfusion_suppression, death_rate, clearance) < 0:
        raise ValueError("Rates, uptake, initial density and duration must be nonnegative.")
    if capacity <= 0 or dt <= 0:
        raise ValueError("Capacity and time step must be positive.")
    steps = max(1, int(np.ceil(duration / dt)))
    h = duration / steps
    for _ in range(steps):
        supply = 1 / (1 + perfusion_suppression * (c + z) / capacity)
        oxygen = supply / (supply + uptake * c / capacity)
        available = max(capacity - z, 0.)
        if available > 0 and c < available:
            c = float(logistic(c, rho * min(1., oxygen/.3) * available/capacity, available, h))
        surviving = c * np.exp(-death_rate * max(1 - oxygen/.1, 0.) * h)
        z = (z + c - surviving) * np.exp(-clearance * h)
        c = surviving
    return c, z


def audit():
    """Independent analytic targets and refinement errors, not saved answers."""
    x = (np.arange(40) + .5) * .5
    initial = 10000 + 2000 * np.cos(np.pi * x / 20)
    exact = logistic(initial, .04, 100000., 2.)
    reaction = evolve(initial, diffusion=0.)
    transported = evolve(initial, diffusion=np.linspace(.05, .3, 40), rho=0.)
    spatial_errors = []
    for n in (20, 40, 80):
        dx = 20 / n
        grid = (np.arange(n) + .5) * dx
        start = 10000 + 1000 * np.cos(np.pi * grid / 20)
        target = 10000 + 1000 * np.cos(np.pi * grid / 20) * np.exp(-.2*(np.pi/20)**2*2)
        got = evolve(start, rho=0., dt=.1*dx**2/.2)
        spatial_errors.append(float(np.sqrt(np.mean((got-target)**2))))
    time_reference = evolve(initial, rho=.2, dt=.0015625)
    temporal_errors = [float(np.linalg.norm(evolve(initial, rho=.2, dt=step)-time_reference))
                       for step in (.2, .1, .05)]
    coupled, necrosis = uniform_coupled(20000., uptake=0.)
    return {
        "logistic_max_abs_error": float(np.max(np.abs(reaction-exact))),
        "relative_mass_error": float(abs(transported.sum()-initial.sum())/initial.sum()),
        "diffusion_minimum": float(transported.min()),
        "spatial_rms_errors": spatial_errors,
        "temporal_l2_errors": temporal_errors,
        "zero_uptake_coupled_abs_error": float(abs(coupled-logistic(20000., .04, 100000., 2.))),
        "zero_uptake_necrosis": necrosis,
    }


def production_audit(checkout):
    """Read the pinned external model and run synthetic 3-D checks in memory.

    Requires that project's NumPy/Numba/SciPy dependencies. No BrainWeb files,
    patient images, downloads or writes to the reference checkout are used.
    The source hash is checked before importing code from the chosen path.
    """
    path = Path(checkout) / "src/brainweb_kpp/model.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_MODEL_SHA256:
        raise ValueError(f"Reference model differs from pinned commit {REFERENCE_COMMIT}.")
    spec = importlib.util.spec_from_file_location("_verified_brainweb_kpp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.set_num_threads(1)
    shape = (9, 7, 5)
    brain = SimpleNamespace(t1=np.ones(shape, np.float32), labels=np.full(shape, 3, np.uint8),
                            mask=np.ones(shape, bool), affine=np.eye(4), spacing=1.,
                            voxel=lambda xyz: tuple(np.rint(xyz).astype(int)))
    base = dict(origin_mm=(4., 3., 2.), radii_mm=(1.,)*3, duration_days=1.,
                frame_days=.5, dt_days=.1, rho=.2)
    p = module.Parameters(**base, d_white=0, d_gray=0)
    initial = module.initial_condition(brain, p)
    reaction = module.simulate(brain, p).final_density
    target = logistic(initial, p.rho, p.carrying_capacity, p.duration_days)
    np.testing.assert_allclose(reaction, target, rtol=4e-6, atol=.005)
    no_growth = module.Parameters(**{**base, "rho": 0.})
    moved = module.simulate(brain, no_growth)
    np.testing.assert_allclose(moved.metrics['cells'], moved.metrics['cells'][0], rtol=3e-7)
    coupled_p = module.Parameters(**base, model="coupled", oxygen_uptake_ratio=0.)
    coupled = module.simulate(brain, coupled_p, compare=True)
    ordinary = module.simulate(brain, module.Parameters(**base))
    np.testing.assert_allclose(coupled.final_density, ordinary.final_density, rtol=3e-6)
    np.testing.assert_array_equal(coupled.final_necrosis, 0.)
    return {"model_sha256": REFERENCE_MODEL_SHA256,
            "reaction_max_abs_error": float(np.max(np.abs(reaction-target))),
            "coupled_limit_max_abs_error": float(np.max(np.abs(coupled.final_density-ordinary.final_density)))}
