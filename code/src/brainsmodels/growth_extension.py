"""Density, anatomy and observation exercises for notebooks 14a--14c.

Educational adaptation of the tissue/threshold experiments in glioma_ai at
b9c1b512d93426392db6c0571f47f1b54ff05eec (MIT, Arvid Lundervold, 2026).
This independent NumPy solver uses first-order diffusion/reaction splitting.
It is not the external BrainWeb solver or an imaging/clinical prediction model.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GrowthRun:
    days: np.ndarray
    density: np.ndarray
    spacing_mm: tuple
    integral: np.ndarray
    reaction_gain: np.ndarray
    balance_error: float
    steps: int
    largest_step: float
    step_limit: float

    @property
    def measure_unit(self):
        return "mm²" if self.density.ndim == 3 else "mL"

    def threshold_measure(self, threshold):
        """Area (2-D) or volume (3-D) of u >= threshold; no MRI calibration."""
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be finite and strictly between zero and one.")
        scale = np.prod(self.spacing_mm) / (1000 if len(self.spacing_mm) == 3 else 1)
        axes = tuple(range(1, self.density.ndim))
        return (self.density >= threshold).sum(axis=axes) * scale


def _neighbours(ndim, axis):
    low, high = [slice(None)] * ndim, [slice(None)] * ndim
    low[axis], high[axis] = slice(None, -1), slice(1, None)
    return tuple(low), tuple(high)


def face_rates(diffusion, spacing_mm):
    """Harmonic face diffusivity / spacing²; zero D blocks that face."""
    d = np.asarray(diffusion, dtype=float)
    h = np.asarray(spacing_mm, dtype=float)
    if d.ndim not in (2, 3) or min(d.shape) < 3:
        raise ValueError("Use a 2-D or 3-D grid with at least three cells per axis.")
    if h.shape != (d.ndim,) or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("Supply one positive finite spacing per spatial axis.")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Diffusivity must be finite and nonnegative.")
    faces, outgoing = [], np.zeros_like(d)
    for axis, step in enumerate(h):
        low, high = _neighbours(d.ndim, axis)
        a, b = d[low], d[high]
        rate = np.divide(2*a*b, a+b, out=np.zeros_like(a), where=a+b > 0) / step**2
        faces.append(rate)
        outgoing[low] += rate
        outgoing[high] += rate
    limit = np.inf if outgoing.max() == 0 else float(1 / outgoing.max())
    return faces, limit


def simulate(initial, diffusion, rho, days, *, spacing_mm, dt=.5, domain=None):
    """No-flux finite-volume transport, then exact local logistic reaction.

    u=c/K is dimensionless; D is mm²/day, rho is day^-1 and times are days.
    Internal faces exchange equal and opposite amounts. The outer faces are
    closed. A boolean domain additionally blocks faces to excluded voxels.
    The requested dt must satisfy the diffusion positivity bound. There is no
    density clipping or automatic replacement of an unstable time step.
    """
    u = np.array(initial, dtype=float, copy=True)
    if u.ndim not in (2, 3) or min(u.shape) < 3:
        raise ValueError("Initial density must be a 2-D or 3-D grid.")
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError("Initial normalized density must be finite and in [0, 1].")
    d = np.broadcast_to(np.asarray(diffusion, dtype=float), u.shape).copy()
    r = np.broadcast_to(np.asarray(rho, dtype=float), u.shape).copy()
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("Proliferation rates must be finite and nonnegative.")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Diffusivities must be finite and nonnegative.")
    if domain is not None:
        inside = np.asarray(domain)
        if inside.dtype != bool or inside.shape != u.shape:
            raise ValueError("Domain must be a boolean mask on the initial grid.")
        if np.any(u[~inside] != 0):
            raise ValueError("Initial density must be zero outside the domain.")
        d[~inside], r[~inside] = 0., 0.
    faces, limit = face_rates(d, spacing_mm)
    if not np.isfinite(dt) or dt <= 0 or dt > limit * (1 + 1e-12):
        raise ValueError(f"Use a positive dt no larger than {limit:g} days.")
    times = np.asarray(days, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Snapshot days must start at zero and increase strictly.")
    scale = float(np.prod(spacing_mm) / (1000 if u.ndim == 3 else 1))
    start = float(u.sum() * scale)
    states, mass, gains = [u.copy()], [start], [0.]
    total_gain, balance, steps, largest = 0., 0., 0, 0.
    for left, right in zip(times[:-1], times[1:]):
        count = int(np.ceil((right-left) / dt))
        h = float((right-left) / count)
        for _ in range(count):
            change = np.zeros_like(u)
            for axis, rate in enumerate(faces):
                low, high = _neighbours(u.ndim, axis)
                transfer = rate * (u[high]-u[low])
                change[low] += transfer
                change[high] -= transfer
            transported = u + h * change
            decay = np.exp(-r * h)
            # Preserve u=0 even if exp(-rho*h) underflows to zero.
            updated = np.divide(transported, decay + transported*(1-decay),
                                out=np.zeros_like(transported), where=transported != 0)
            if not np.isfinite(updated).all() or np.any((updated < -1e-12) | (updated > 1 + 1e-12)):
                raise ArithmeticError("Density bounds failed; no clipping was applied.")
            total_gain += float((updated-transported).sum() * scale)
            u = updated
            balance = max(balance, abs(float(u.sum()*scale)-start-total_gain))
        steps += count
        largest = max(largest, h)
        states.append(u.copy())
        mass.append(float(u.sum() * scale))
        gains.append(total_gain)
    return GrowthRun(times.copy(), np.stack(states), tuple(spacing_mm),
                     np.array(mass), np.array(gains), balance, steps, largest, limit)


def gaussian_seed(axes, origin, *, width=6., peak=.12, domain=None):
    """Gaussian in physical coordinates, explicitly masked to the chosen domain."""
    if len(axes) != len(origin) or not np.isfinite(origin).all():
        raise ValueError("Supply a finite origin coordinate for each axis.")
    if not np.isfinite([width, peak]).all() or width <= 0 or not 0 < peak <= 1:
        raise ValueError("Use a positive width and peak in (0, 1].")
    radius2 = 0.
    for axis, (coordinates, centre) in enumerate(zip(axes, origin)):
        shape = [1] * len(axes)
        shape[axis] = len(coordinates)
        radius2 = radius2 + ((np.asarray(coordinates).reshape(shape)-centre)/width)**2
    u = peak * np.exp(-.5 * radius2)
    if domain is not None:
        if np.asarray(domain).dtype != bool or np.shape(domain) != u.shape:
            raise ValueError("Seed domain must be a matching boolean mask.")
        u = np.where(domain, u, 0.)
    return u


def constructed_tissues():
    """An invented 2-D domain, not BrainWeb, an atlas or a patient slice."""
    axis = np.arange(-40., 41.)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    labels = np.where(radius < 35, 2, 0)
    labels[radius < 26] = 3
    labels[(abs(x) < 2) & (y > 8) & (radius < 35)] = 1
    return (axis, axis.copy()), labels


def tissue_diffusion(labels, *, d_white=.5, d_gray=.1):
    """Scalar D in each voxel; heterogeneous but NOT fibre-directed anisotropy.

    GM=2 and glial class=8 share D_gray; WM=3 uses D_white. CSF and all other
    classes have D=0 and are excluded from the growth domain in these exercises.
    """
    if not np.isfinite([d_white, d_gray]).all() or min(d_white, d_gray) < 0:
        raise ValueError("Tissue diffusivities must be finite and nonnegative.")
    return np.where(labels == 3, d_white, np.where(np.isin(labels, [2, 8]), d_gray, 0.))


def observe(density, *, block=2):
    """Non-overlapping block means of normalized density, not an MRI simulator.

    Trim high-index edge cells if needed; the identical declared crop is used
    for generated observations and predictions. No noise or threshold is added.
    """
    u = np.asarray(density, dtype=float)
    if u.ndim not in (2, 3) or not isinstance(block, int) or block < 1 or not np.isfinite(u).all():
        raise ValueError("Use a 2-D/3-D density field and a positive integer block.")
    shape = tuple(n // block for n in u.shape)
    if min(shape) == 0:
        raise ValueError("Observation blocks exceed the image dimensions.")
    trimmed = u[tuple(slice(0, n*block) for n in shape)]
    pairs = tuple(v for n in shape for v in (n, block))
    return trimmed.reshape(pairs).mean(axis=tuple(range(1, 2*u.ndim, 2)))


def parameter_search(initial, base_diffusion, days, observations, scales, rhos,
                     *, spacing_mm, domain, dt=1., block=2, observation_mask=None):
    """Grid least-squares fit using ONLY supplied observation times/arrays.

    The initial state, tissue ratio, domain and observation operator are fixed.
    Candidate ranges are a bounded search design, not a Bayesian prior or a
    confidence interval. The caller supplies withheld times only after fitting.
    """
    scales, rhos = np.asarray(scales, float), np.asarray(rhos, float)
    if (scales.ndim != 1 or rhos.ndim != 1 or len(scales) < 2 or len(rhos) < 2
            or not np.isfinite(scales).all() or not np.isfinite(rhos).all()
            or np.any(scales < 0) or np.any(rhos < 0)):
        raise ValueError("Use finite nonnegative parameter grids with at least two candidates.")
    target = np.asarray(observations, float)
    expected = (len(days)-1, *observe(initial, block=block).shape)
    if target.shape != expected or not np.isfinite(target).all():
        raise ValueError(f"Expected finite observations of shape {expected}.")
    selected = np.ones(expected[1:], bool) if observation_mask is None else np.asarray(observation_mask)
    if selected.dtype != bool or selected.shape != expected[1:] or not selected.any():
        raise ValueError("Observation mask must be a nonempty matching boolean array.")
    losses = np.empty((len(rhos), len(scales)))
    for i, rho in enumerate(rhos):
        for j, scale in enumerate(scales):
            run = simulate(initial, scale*base_diffusion, rho, days,
                           spacing_mm=spacing_mm, dt=dt, domain=domain)
            prediction = np.stack([observe(u, block=block) for u in run.density[1:]])
            losses[i, j] = np.mean((prediction[:, selected]-target[:, selected])**2)
    i, j = np.unravel_index(losses.argmin(), losses.shape)
    return {"scale": float(scales[j]), "rho": float(rhos[i]), "losses": losses,
            "at_boundary": bool(i in (0, len(rhos)-1) or j in (0, len(scales)-1))}
