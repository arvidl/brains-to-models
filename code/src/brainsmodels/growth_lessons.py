"""Shared experiments and figures; the notebooks explain each operation."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from brainsmodels import brainweb_data
from brainsmodels.growth_extension import (
    constructed_tissues, gaussian_seed, observe, parameter_search, simulate, tissue_diffusion,
)

ORIGIN = (-24., -12., 24.)  # RAS+ millimetres in the pinned normal phantom.


def planar_experiment(*, d_white=.5, d_gray=.1, rho=.04, dt=.4,
                      days=(0., 30., 60., 90.)):
    axes, labels = constructed_tissues()
    domain = np.isin(labels, [2, 3])
    initial = gaussian_seed(axes, (-14., 0.), domain=domain)
    d = tissue_diffusion(labels, d_white=d_white, d_gray=d_gray)
    mean_d = float(d[domain].mean())
    settings = dict(rho=rho, days=days, spacing_mm=(1., 1.), dt=dt, domain=domain)
    runs = {
        "Growth only": simulate(initial, 0., **settings),
        "Homogeneous D": simulate(initial, mean_d, **settings),
        "Tissue-dependent D": simulate(initial, d, **settings),
    }
    return axes, labels, runs, mean_d


def anatomical_experiment(brain, *, d_white=.5, d_gray=.1, rho=.04, dt=1.,
                          days=(0., 30., 60., 90.)):
    origin_index = tuple(int(np.argmin(abs(axis-centre))) for axis, centre in zip(brain.axes, ORIGIN))
    if not brain.domain[origin_index]:
        raise ValueError("The declared seed centre is outside the growth domain at this spacing.")
    initial = gaussian_seed(brain.axes, ORIGIN, width=8., domain=brain.domain)
    d = tissue_diffusion(brain.labels, d_white=d_white, d_gray=d_gray)
    mean_d = float(d[brain.domain].mean())
    settings = dict(rho=rho, days=days, spacing_mm=brain.spacing_mm, dt=dt,
                    domain=brain.domain)
    runs = {
        "Homogeneous D": simulate(initial, mean_d, **settings),
        "Tissue-dependent D": simulate(initial, d, **settings),
    }
    return initial, runs, mean_d


def recovery_problem(brain, *, noise_sd=.005, seed=7, true_scale=1., true_rho=.04):
    """Generate a known two-parameter problem before selecting any estimates."""
    if not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("Use a finite, nonnegative observation-noise standard deviation.")
    initial = gaussian_seed(brain.axes, ORIGIN, width=8., domain=brain.domain)
    d = tissue_diffusion(brain.labels)
    fit_days = np.array([0., 20., 40., 60.])
    all_days = np.r_[fit_days, 90.]
    generator = simulate(initial, d*true_scale, true_rho, all_days, spacing_mm=brain.spacing_mm,
                         domain=brain.domain, dt=.5)
    clean = np.stack([observe(u) for u in generator.density[1:]])
    # Select a physical cube before generating the noise or fitting parameters.
    observed_axes = [axis[:len(axis)//2*2].reshape(-1, 2).mean(axis=1) for axis in brain.axes]
    roi = np.ones(clean.shape[1:], bool)
    for i, (axis, centre) in enumerate(zip(observed_axes, ORIGIN)):
        shape = [1, 1, 1]
        shape[i] = len(axis)
        roi &= (abs(axis-centre) <= 36.).reshape(shape)
    observations = clean + np.random.default_rng(seed).normal(0., noise_sd, clean.shape)
    # Gaussian measurement noise is not clipped at 0 or 1.
    scales = np.linspace(.6, 1.4, 9)
    rhos = np.linspace(.025, .055, 7)
    return dict(initial=initial, base_diffusion=d, fit_days=fit_days, days=all_days,
                clean=clean, observations=observations, roi=roi, noise_sd=noise_sd, seed=seed,
                true_scale=true_scale, true_rho=true_rho,
                scales=scales, rhos=rhos, generator=generator,
                spacing_mm=brain.spacing_mm, domain=brain.domain)


def fit_recovery_problem(problem, *, dt=1., fit_indices=(0, 1, 2), scales=None, rhos=None):
    """The fitting call receives neither day 90 nor its observation."""
    p = problem
    indices = np.asarray(fit_indices)
    if (indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices > 2) or np.any(np.diff(indices) <= 0)):
        raise ValueError("Choose increasing early-observation indices from 0, 1, 2; day 90 stays withheld.")
    scales = p["scales"] if scales is None else np.asarray(scales)
    rhos = p["rhos"] if rhos is None else np.asarray(rhos)
    fit = parameter_search(p["initial"], p["base_diffusion"], np.r_[0., p["days"][indices+1]],
                           p["observations"][indices], scales, rhos,
                           spacing_mm=p["spacing_mm"], domain=p["domain"],
                           dt=dt, observation_mask=p["roi"])
    return dict(**fit, fit_indices=indices, dt=dt, scales=scales, rhos=rhos)


def recovery_forecast(problem, fit):
    """Freeze the estimates, then evaluate day 90 and a persistence baseline."""
    p = problem
    observations, roi, clean = p["observations"], p["roi"], p["clean"]
    indices = fit["fit_indices"]
    forecast = simulate(p["initial"], fit["scale"]*p["base_diffusion"], fit["rho"], p["days"],
                        spacing_mm=p["spacing_mm"], domain=p["domain"], dt=fit["dt"])
    predicted = np.stack([observe(u) for u in forecast.density[1:]])
    persistence = observations[indices[-1]]
    summary = {
        "known_scale": p["true_scale"], "known_rho_per_day": p["true_rho"],
        "estimated_scale": fit["scale"], "estimated_rho_per_day": fit["rho"],
        "search_boundary": fit["at_boundary"], "noise_sd": p["noise_sd"], "seed": p["seed"],
        "fit_days": p["days"][indices+1].tolist(), "withheld_day": 90.,
        "fit_rmse": float(np.sqrt(np.mean((predicted[indices][:, roi]-observations[indices][:, roi])**2))),
        "withheld_rmse": float(np.sqrt(np.mean((predicted[3, roi]-observations[3, roi])**2))),
        "persistence_withheld_rmse": float(np.sqrt(np.mean((persistence[roi]-observations[3, roi])**2))),
        "withheld_clean_rmse": float(np.sqrt(np.mean((predicted[3, roi]-clean[3, roi])**2))),
        "observation_voxels_in_roi": int(roi.sum()),
        "generator_dt_days": .5, "fit_dt_days": fit["dt"],
        "shared_spatial_grid_and_model": True,
    }
    return dict(**p, fit=fit, predicted=predicted, forecast=forecast, summary=summary)


def recovery_experiment(brain):
    """Convenience entry point; notebooks run these three stages separately."""
    problem = recovery_problem(brain)
    return recovery_forecast(problem, fit_recovery_problem(problem))


def plot_planar(axes, labels, runs, *, snapshot=-1, thresholds=(.05, .1, .2),
                contour_levels=(.05, .2)):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7.4), constrained_layout=True)
    extent = [axes[0][0]-.5, axes[0][-1]+.5, axes[1][0]-.5, axes[1][-1]+.5]
    for panel, name in zip(ax[0], ["Homogeneous D", "Tissue-dependent D"]):
        panel.imshow(labels.T, origin="lower", extent=extent, cmap="Greys", alpha=.25)
        field = runs[name].density[snapshot]
        image = panel.imshow(np.ma.masked_where(field < .01, field).T,
                             origin="lower", extent=extent, vmin=0, vmax=1, cmap="magma")
        levels = sorted(set(v for v in contour_levels if field.min() < v < field.max()))
        if levels:
            panel.contour(axes[0], axes[1], field.T, levels=levels, colors=["cyan", "lime"], linewidths=1)
        panel.set(title=f"{name}, day {runs[name].days[snapshot]:g}", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=ax[0], label="Normalized density u = c/K", shrink=.8)
    for name, run in runs.items():
        ax[1, 0].plot(run.days, run.integral, "o-", label=name)
    ax[1, 0].set(xlabel="Days", ylabel="Integral of u (mm²)", title="Integrated density")
    ax[1, 0].legend(fontsize=10)
    run = runs["Tissue-dependent D"]
    for threshold in thresholds:
        ax[1, 1].plot(run.days, run.threshold_measure(threshold), "o-", label=f"u ≥ {threshold}")
    ax[1, 1].set(xlabel="Days", ylabel="Area (mm²)", title="One state, three thresholds")
    ax[1, 1].legend()
    return fig


def plot_brainweb(brain, runs, *, snapshot=-1, coordinates=ORIGIN, threshold=.01, alpha=.8):
    """Three orthogonal physical-coordinate views, each through the seed."""
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
    run = runs["Tissue-dependent D"]
    field = run.density[snapshot]
    for panel, fixed, name in zip(ax, [2, 1, 0], ["Axial", "Coronal", "Sagittal"]):
        index = int(np.argmin(abs(brain.axes[fixed]-coordinates[fixed])))
        others = [i for i in range(3) if i != fixed]
        a, b = others
        h = brain.spacing_mm[0] / 2
        extent = [brain.axes[a][0]-h, brain.axes[a][-1]+h,
                  brain.axes[b][0]-h, brain.axes[b][-1]+h]
        base = np.take(brain.t1, index, axis=fixed)
        values = np.take(field, index, axis=fixed)
        panel.imshow(base.T, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
        image = panel.imshow(np.ma.masked_where(values < threshold, values).T, origin="lower", extent=extent,
                             vmin=0, vmax=1, cmap="magma", alpha=alpha)
        panel.set(title=f"{name}: {'xyz'[fixed]} = {brain.axes[fixed][index]:g} mm",
                  xlabel=f"{'xyz'[a]} (RAS+ mm)", ylabel=f"{'xyz'[b]} (RAS+ mm)")
    # The scalar legend shows the density colours before alpha blending with T1.
    from matplotlib.cm import ScalarMappable
    legend = ScalarMappable(norm=image.norm, cmap=image.cmap)
    fig.colorbar(legend, ax=ax, label="u = c/K", shrink=.65, fraction=.025, pad=.02)
    fig.suptitle(f"Day {run.days[snapshot]:g}: simulated density on the normal BrainWeb phantom")
    return fig


def plot_anatomical_measures(runs):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for name, run in runs.items():
        ax[0].plot(run.days, run.integral, "o-", label=name)
    ax[0].set(xlabel="Days", ylabel="Integral of u / 1000 (mL)", title="Normalized burden, not a cell count")
    ax[0].legend(fontsize=8)
    run = runs["Tissue-dependent D"]
    for threshold in [.05, .1, .2]:
        ax[1].plot(run.days, run.threshold_measure(threshold), "o-", label=f"u ≥ {threshold}")
    ax[1].set(xlabel="Days", ylabel="Threshold volume (mL)", title="A chosen reading of the same density field")
    ax[1].legend()
    return fig


def plot_recovery(result):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    scales, rhos = result["fit"]["scales"], result["fit"]["rhos"]
    ds, dr = np.diff(scales)[0]/2, np.diff(rhos)[0]/2
    image = ax[0].imshow(np.sqrt(result["fit"]["losses"]), origin="lower", aspect="auto",
                         extent=[scales[0]-ds, scales[-1]+ds, rhos[0]-dr, rhos[-1]+dr], cmap="viridis")
    ax[0].plot(result["true_scale"], result["true_rho"], "w+", ms=12, mew=2, label="Known")
    ax[0].plot(result["fit"]["scale"], result["fit"]["rho"], "rx", ms=8, mew=2, label="Grid estimate")
    indices = result["fit"]["fit_indices"]
    fit_days = result["days"][indices+1]
    ax[0].set(xlabel="Multiplier of both tissue diffusivities", ylabel="ρ (day⁻¹)",
              title="Fit days: " + ", ".join(f"{d:g}" for d in fit_days))
    ax[0].legend(fontsize=10)
    fig.colorbar(image, ax=ax[0], label="RMSE (normalized-density units)")
    roi, times = result["roi"], result["days"][1:]
    ax[1].axvspan(fit_days[-1], 90, color=".9", label="Forecast interval")
    ax[1].plot(times, result["clean"][:, roi].mean(axis=1), "-", label="Generated noiseless mean")
    ax[1].plot(times, result["predicted"][:, roi].mean(axis=1), "--", label="Fitted-model mean")
    ax[1].plot(fit_days, result["observations"][indices][:, roi].mean(axis=1), "o", label="Fit observations")
    ax[1].plot(times[3:], result["observations"][3:, roi].mean(axis=1), "s", label="Withheld observation")
    ax[1].set(xlabel="Days", ylabel="Mean observed density proxy in fixed cube",
              title="Day 90 was withheld from parameter selection")
    ax[1].legend(fontsize=9)
    return fig


def write_paper_assets(root):
    root = Path(root)
    figures = root / "paper/figures"
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
    axes, labels, planar, mean_d = planar_experiment()
    brain = brainweb_data.load_brainweb(step_mm=4)
    _, anatomical, anatomy_mean = anatomical_experiment(brain)
    recovery = recovery_experiment(brainweb_data.load_brainweb(step_mm=6))
    for name, fig in [("growth_tissue_thresholds", plot_planar(axes, labels, planar)),
                      ("growth_brainweb", plot_brainweb(brain, anatomical)),
                      ("growth_recovery", plot_recovery(recovery))]:
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    values = {
        "teaching_source_commit": "b9c1b512d93426392db6c0571f47f1b54ff05eec",
        "data_commit": brainweb_data.COMMIT,
        "data_files": {name: {"sha256": digest, "bytes": size}
                       for name, (digest, size) in brainweb_data.FILES.items()},
        "planar_mean_D_mm2_per_day": mean_d,
        "brainweb_mean_D_mm2_per_day": anatomy_mean,
        "brainweb_grid_spacing_mm": list(brain.spacing_mm),
        "brainweb_grid_shape": list(brain.labels.shape),
        "origin_ras_mm": list(ORIGIN),
        "day90_normalized_burden_mL": {name: float(run.integral[-1]) for name, run in anatomical.items()},
        "day90_threshold_volumes_mL": {str(t): float(anatomical["Tissue-dependent D"].threshold_measure(t)[-1])
                                         for t in [.05, .1, .2]},
        "max_balance_residual": float(max(r.balance_error for r in [*planar.values(), *anatomical.values()])),
        "recovery": recovery["summary"],
    }
    (figures / "growth_extension_values.json").write_text(json.dumps(values, indent=2) + "\n")
    macros = {
        "GrowthBurden": f"{anatomical['Tissue-dependent D'].integral[-1]:.2f}",
        "GrowthVolumeLow": f"{values['day90_threshold_volumes_mL']['0.05']:.2f}",
        "GrowthVolumeHigh": f"{values['day90_threshold_volumes_mL']['0.2']:.2f}",
        "GrowthForecastError": f"{recovery['summary']['withheld_rmse']:.5f}",
        "GrowthPersistenceError": f"{recovery['summary']['persistence_withheld_rmse']:.5f}",
    }
    (figures / "growth_extension_numbers.tex").write_text(
        "% Generated by scripts/build_growth_lessons.py; do not edit numbers by hand.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()))
    return values
