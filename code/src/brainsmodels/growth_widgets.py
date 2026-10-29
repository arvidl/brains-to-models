"""Live controls for the three growth lessons; all calculations stay local.

Simulation controls require a button press. Display controls only read cached
states. The fixed notebook examples and paper figures do not use widget state.
"""
from html import escape

import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from brainsmodels import growth_lessons as lessons
from brainsmodels.brainweb_data import load_brainweb


def _slider(value, low, high, step, description):
    return widgets.FloatSlider(value=value, min=low, max=high, step=step,
                               description=description, continuous_update=False,
                               readout_format=".3g")


class GrowthExplorer:
    """A button-driven calculation with separately observable display controls."""

    def __init__(self, model, view, compute, draw, *, button_text):
        self.model, self.view = model, view
        self.controls = {**model, **view}
        self.compute, self.draw = compute, draw
        self.last_result = self.last_parameters = None
        self.error = None
        self.pending = False
        self.run_button = widgets.Button(description=button_text, button_style="primary",
                                         layout=widgets.Layout(width="100%", max_width="230px"))
        self.status = widgets.HTML("Choose parameters, then press the run button.")
        self.output = widgets.Output()
        for control in self.controls.values():
            control.style.description_width = "170px"
            control.layout.width = "100%"
            control.layout.max_width = "390px"
            control.layout.min_width = "0px"
            control.layout.flex_flow = "row wrap"
            control.layout.height = "auto"
            control.layout.min_height = "32px"
        for control in model.values():
            control.observe(self._model_changed, names="value")
        for control in view.values():
            control.observe(self._view_changed, names="value")
        self.run_button.on_click(self._run)
        columns = [widgets.VBox([widgets.HTML("<b>Experiment controls</b>"),
                                 *model.values(), self.run_button, self.status],
                                layout=widgets.Layout(width="410px", max_width="100%", min_width="0px"))]
        if view:
            columns.append(widgets.VBox([widgets.HTML("<b>Read the last run (cached states)</b>"),
                                          *view.values()],
                                         layout=widgets.Layout(width="410px", max_width="100%", min_width="0px")))
        controls = widgets.Box(columns, layout=widgets.Layout(
            display="flex", flex_flow="row wrap", gap="12px"))
        self.widget = widgets.VBox([controls, self.output])

    def _model_changed(self, _):
        values = {key: control.value for key, control in self.model.items()}
        self.pending = self.last_parameters is not None and values != self.last_parameters
        if self.last_result is not None:
            self.status.value = ("<b>Parameters changed.</b> Press the run button to apply them; "
                                 "the figure still reads the previous run." if self.pending else
                                 "Showing the last computed run; controls match its parameters.")

    def _view_changed(self, _):
        if self.last_result is not None:
            self._render()

    def _render(self):
        self.output.clear_output(wait=True)
        with self.output:
            self.draw(self.last_result, **{key: w.value for key, w in self.view.items()})

    def _run(self, _):
        self.run_button.disabled = True
        self.status.value = "Computing…"
        self.error = None
        parameters = {key: control.value for key, control in self.model.items()}
        try:
            result = self.compute(**parameters)
            self.last_result, self.last_parameters = result, parameters
            self.pending = False
            self._render()
            self.status.value = "<b>Complete.</b> Showing the parameters applied by this run."
        except (ValueError, ArithmeticError) as error:
            self.last_result = None
            self.error = str(error)
            self.output.clear_output()
            self.status.value = "<b>Run rejected:</b> " + escape(str(error))
        finally:
            self.run_button.disabled = False


def _show(fig):
    display(fig)
    plt.close(fig)


def _tissue_controls(dt, steps):
    return {
        "d_white": _slider(.5, 0., 1., .05, "White D (mm²/day)"),
        "d_gray": _slider(.1, 0., .4, .02, "Grey D (mm²/day)"),
        "rho": _slider(.04, 0., .08, .005, "Growth ρ (1/day)"),
        "dt": widgets.Dropdown(options=steps, value=dt, description="Requested step (days)"),
    }


def planar_explorer():
    """14a: tissue coefficients and time step; cached day and threshold views."""
    model = _tissue_controls(.4, [.1, .2, .4, 1.])
    view = {
        "day": widgets.IntSlider(value=90, min=0, max=120, step=10,
                                 description="Read day", continuous_update=False),
        "threshold": _slider(.2, .01, .5, .01, "Density threshold"),
    }

    def compute(**parameters):
        return lessons.planar_experiment(**parameters, days=np.arange(0., 121., 10.))

    def draw(result, day, threshold):
        axes, labels, runs, mean_d = result
        index = int(np.flatnonzero(runs["Tissue-dependent D"].days == day)[0])
        run = runs["Tissue-dependent D"]
        print(f"Matched domain-mean D: {mean_d:.3f} mm²/day. "
              f"Actual largest step: {run.largest_step:g}; bound: {run.step_limit:g} days.")
        print(f"Day {day}: integral {run.integral[index]:.2f} mm²; "
              f"area at u ≥ {threshold:g}: {run.threshold_measure(threshold)[index]:.0f} mm².")
        _show(lessons.plot_planar(axes, labels, runs, snapshot=index,
                                 thresholds=(threshold,), contour_levels=(threshold,)))

    return GrowthExplorer(model, view, compute, draw, button_text="Run comparison")


def brainweb_explorer():
    """14b: tissue/grid controls and cached orthogonal physical-coordinate views."""
    model = _tissue_controls(1., [.25, .5, 1., 2.])
    model["spacing"] = widgets.Dropdown(options=[2, 4, 6], value=4,
                                        description="Grid spacing (mm)")
    view = {
        "day": widgets.IntSlider(value=90, min=0, max=120, step=10,
                                 description="Read day", continuous_update=False),
        "threshold": _slider(.1, .01, .5, .01, "Density threshold"),
        "x": _slider(-24., -80., 80., 2., "Sagittal x (RAS+ mm)"),
        "y": _slider(-12., -120., 80., 2., "Coronal y (RAS+ mm)"),
        "z": _slider(24., -60., 100., 2., "Axial z (RAS+ mm)"),
        "alpha": _slider(.8, .1, 1., .1, "Overlay opacity"),
    }

    def compute(spacing, **parameters):
        brain = load_brainweb(step_mm=spacing)
        _, runs, mean_d = lessons.anatomical_experiment(
            brain, **parameters, days=np.arange(0., 121., 10.))
        return brain, runs, mean_d

    def draw(result, day, threshold, x, y, z, alpha):
        brain, runs, mean_d = result
        run = runs["Tissue-dependent D"]
        index = int(np.flatnonzero(run.days == day)[0])
        print(f"Grid {brain.labels.shape}, {brain.spacing_mm[0]} mm. Mean D {mean_d:.3f} mm²/day. "
              f"Actual step {run.largest_step:g}; bound {run.step_limit:g} days.")
        print(f"Day {day}: normalized burden {run.integral[index]:.3f} mL; "
              f"volume at u ≥ {threshold:g}: {run.threshold_measure(threshold)[index]:.3f} mL.")
        _show(lessons.plot_brainweb(brain, runs, snapshot=index, coordinates=(x, y, z),
                                   threshold=threshold, alpha=alpha))

    return GrowthExplorer(model, view, compute, draw, button_text="Run 3-D simulation")


def recovery_explorer():
    """14c: generated truth/noise, training times and numerical/search choices."""
    model = {
        "true_scale": _slider(1., .6, 1.4, .1, "True D multiplier"),
        "true_rho": _slider(.04, .025, .055, .005, "True ρ (1/day)"),
        "noise_sd": _slider(.005, 0., .02, .001, "Observation noise SD"),
        "seed": widgets.IntSlider(value=7, min=0, max=50, description="Noise seed",
                                  continuous_update=False),
        "fit_indices": widgets.Dropdown(options=[("20, 40, 60 days", (0, 1, 2)),
                                                   ("20 and 60 days", (0, 2)),
                                                   ("20 days only", (0,)),
                                                   ("60 days only", (2,))],
                                          value=(0, 1, 2), description="Fit observations"),
        "grid_size": widgets.Dropdown(options=[("Coarse: 5 × 5", (5, 5)),
                                                ("Default: 9 × 7", (9, 7)),
                                                ("Fine: 13 × 11", (13, 11))],
                                       value=(9, 7), description="Candidate grid"),
        "dt": widgets.Dropdown(options=[.5, 1., 2.], value=1., description="Fit step (days)"),
    }

    def compute(fit_indices, grid_size, dt, **parameters):
        problem = lessons.recovery_problem(load_brainweb(step_mm=6), **parameters)
        fit = lessons.fit_recovery_problem(problem, fit_indices=fit_indices, dt=dt,
                                           scales=np.linspace(.6, 1.4, grid_size[0]),
                                           rhos=np.linspace(.025, .055, grid_size[1]))
        return lessons.recovery_forecast(problem, fit)

    def draw(result):
        s = result["summary"]
        print(f"Estimated D multiplier {s['estimated_scale']:.3g} (known {s['known_scale']:.3g}); "
              f"ρ {s['estimated_rho_per_day']:.4g} (known {s['known_rho_per_day']:.4g}) /day.")
        print(f"Fit days {s['fit_days']}; day 90 withheld. Withheld RMSE {s['withheld_rmse']:.5f}; "
              f"persistence from day {s['fit_days'][-1]:g}: {s['persistence_withheld_rmse']:.5f}.")
        print("Minimum touches search boundary:", s["search_boundary"],
              "| candidate spacing is not a confidence interval.")
        _show(lessons.plot_recovery(result))

    return GrowthExplorer(model, {}, compute, draw, button_text="Fit and forecast")
