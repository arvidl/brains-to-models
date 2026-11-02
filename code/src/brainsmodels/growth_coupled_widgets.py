"""Notebook controls for the bundled coupled solver and matched reference."""
from dataclasses import replace
from html import escape
from pathlib import Path

import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from brainsmodels import growth_coupled as lesson
from brainsmodels.growth_widgets import GrowthExplorer


def _slider(value, low, high, step, label):
    return w.FloatSlider(value=value, min=low, max=high, step=step, description=label,
                         continuous_update=False, readout_format=".3g")


def coupled_explorer(*, initial_run=None, output_directory="growth-14d-exports"):
    """The run button computes; frame, slice and display controls read histories."""
    _, defaults = lesson.preset()
    model = {
        "model": w.Dropdown(options=[("Tumour + oxygen + necrosis", "coupled"),
                                     ("Original Fisher–KPP", "fisher_kpp")], value="coupled", description="Model"),
        "spacing": w.Dropdown(options=[1, 2, 4, 6], value=4, description="Grid spacing (mm)"),
        "duration_days": w.IntSlider(value=90, min=6, max=180, step=3, description="Duration (days)"),
        "dt_days": w.Dropdown(options=[.1, .25, .5, 1., 2.], value=.5, description="Max step (days)"),
        "frame_days": w.Dropdown(options=[1.5, 3., 6., 9.], value=6., description="Save every (days)"),
        "auto_substep": w.Checkbox(value=True, description="Automatic stable substeps"),
        "origin_x": _slider(-30, -90, 90, 1, "Seed x (RAS+ mm)"),
        "origin_y": _slider(25, -126, 90, 1, "Seed y (RAS+ mm)"),
        "origin_z": _slider(30, -72, 108, 1, "Seed z (RAS+ mm)"),
        "radius_x": _slider(6, 1, 20, 1, "Seed radius x (mm)"),
        "radius_y": _slider(6, 1, 20, 1, "Seed radius y (mm)"),
        "radius_z": _slider(6, 1, 20, 1, "Seed radius z (mm)"),
        "seed_profile": w.Dropdown(options=[("Gaussian (radius = σ)", "gaussian"),
                                             ("Uniform ellipsoid", "ellipsoid")], description="Seed profile"),
        "initial_density": w.FloatText(value=80000, description="Peak cells/mm³"),
        "carrying_capacity": w.FloatText(value=100000, description="Capacity K (cells/mm³)"),
        "d_white": _slider(.5, 0, 1, .05, "White D (mm²/day)"),
        "d_gray": _slider(.1, 0, .5, .02, "Grey D (mm²/day)"),
        "rho": _slider(.04, 0, .1, .005, "Proliferation (1/day)"),
        "oxygen_length_mm": _slider(.2, 0, 2, .1, "Oxygen length (mm)"),
        "oxygen_uptake_ratio": _slider(4, 0, 10, .5, "Oxygen uptake β"),
        "perfusion_suppression": _slider(4, 0, 10, .5, "Perfusion suppression α"),
        "oxygen_prolif_level": _slider(.3, .05, .95, .05, "Proliferation threshold"),
        "oxygen_death_level": _slider(.1, .01, .5, .01, "Death threshold"),
        "necrosis_rate": _slider(.5, 0, 1, .05, "Death rate δ (1/day)"),
        "necrosis_clearance": _slider(0, 0, .2, .01, "Clearance λ (1/day)"),
    }
    view = {
        "frame": w.IntSlider(value=15, min=0, max=15, description="Saved frame", continuous_update=False),
        "x": _slider(-30, -90, 90, 1, "View x (RAS+ mm)"),
        "y": _slider(25, -126, 90, 1, "View y (RAS+ mm)"),
        "z": _slider(30, -72, 108, 1, "View z (RAS+ mm)"),
        "planes": w.Dropdown(options=["All three", "Axial", "Coronal", "Sagittal"], description="Planes"),
        "threshold": _slider(.02, 0, .5, .01, "Hide density below /K"),
        "alpha": _slider(.9, .1, 1, .1, "Overlay opacity"),
    }
    panel = None

    def compute(spacing, origin_x, origin_y, origin_z, radius_x, radius_y, radius_z, **settings):
        parameters = replace(defaults, origin_mm=(origin_x, origin_y, origin_z),
                             radii_mm=(radius_x, radius_y, radius_z), **settings)
        result = lesson.run_comparison(parameters, spacing=spacing)
        # The previous result must not render against a new frame range.
        panel.last_result = None
        view["frame"].max = len(result[1].times) - 1
        view["frame"].value = view["frame"].max
        return result

    def draw(pair, frame, x, y, z, planes, threshold, alpha):
        brain, result = pair
        meta = result.metadata
        print(f"Day {result.times[frame]:g}; Δx {brain.spacing:g} mm; "
              f"actual maximum step {meta['actual_max_dt_days']:g} days; "
              f"safe bound {meta['safe_dt_days']}; {meta['internal_steps']} internal steps.")
        print(f"Viable cells: {result.metrics['cells'][frame]:.4g}", end="")
        if result.comparison is not None:
            print(f"; necrotic equivalents: {result.metrics['necrotic_cells'][frame]:.4g}; "
                  f"minimum tissue oxygen: {result.metrics['min_oxygen'][frame]:.4g}")
        else:
            print()
        figure = lesson.plot_comparison(brain, result, frame=frame, coordinates=(x, y, z),
                                         planes=planes, threshold=threshold, alpha=alpha)
        display(figure)
        plt.close(figure)

    panel = GrowthExplorer(model, view, compute, draw, button_text="Run growth simulation")
    # Report a failed elliptic solve as an actionable notebook message too.
    original_compute = panel.compute
    def guarded_compute(**settings):
        try:
            return original_compute(**settings)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
    panel.compute = guarded_compute
    preset_control = w.Dropdown(options=[("CPU teaching · 4 mm", "teaching"),
                                         ("Slide comparison · 2 mm / 90 days", "slide-coupled"),
                                         ("Slide Fisher–KPP · 1 mm / 180 days", "slide-fisher")],
                                description="Load preset", layout=w.Layout(width="100%"))
    def load_preset(change):
        spacing, p = lesson.preset(change["new"])
        settings = vars(p)
        for key, control in model.items():
            if key == "spacing": control.value = spacing
            elif key.startswith("origin_"): control.value = p.origin_mm["xyz".index(key[-1])]
            elif key.startswith("radius_"): control.value = p.radii_mm["xyz".index(key[-1])]
            else: control.value = settings[key]
        for key, value in zip(("x", "y", "z"), p.origin_mm): view[key].value = value
    preset_control.observe(load_preset, names="value")
    panel.preset_control = preset_control
    groups = [list(model)[:6], list(model)[6:15], list(model)[15:18], list(model)[18:]]
    # Colab's current custom manager fails to construct Tab child views. Basic
    # Dropdown and VBox models also work on older hosted widget managers.
    titles = ["Model & grid", "Seed & density", "Spread & growth", "Oxygen & necrosis"]
    section = w.Dropdown(options=list(zip(titles, range(len(groups)))), value=0,
                         description="Controls", layout=w.Layout(width="100%"))
    parameter_box = w.VBox([model[k] for k in groups[0]])
    section.observe(lambda change: setattr(parameter_box, "children",
                                          tuple(model[k] for k in groups[change["new"]])), names="value")
    panel.section_control = section
    player = w.Play(value=15, min=0, max=15, interval=300, description="Play")
    w.link((player, "value"), (view["frame"], "value"))
    view["frame"].observe(lambda c: setattr(player, "max", c["new"]), names="max")
    panel.player = player
    export_type = w.Dropdown(options=[("Poster + JSON", "poster"), ("Animated GIF", "gif"),
                                     ("MP4 video", "mp4")], description="Export")
    export_button = w.Button(description="Export last run")
    export_status = w.HTML()
    def export(_):
        if panel.last_result is None:
            export_status.value = "Run a simulation first."
            return
        if panel.pending:
            export_status.value = "Parameters changed; run again before exporting."
            return
        export_button.disabled = True
        export_status.value = "Writing export…"
        try:
            brain, result = panel.last_result
            folder = lesson.export_run(brain, result, output_directory,
                coordinates=tuple(view[k].value for k in ("x", "y", "z")),
                planes=view["planes"].value, threshold=view["threshold"].value, alpha=view["alpha"].value,
                frame=view["frame"].value,
                animation=None if export_type.value == "poster" else export_type.value)
            export_status.value = "Saved to " + escape(str(Path(folder).resolve()))
        except (ValueError, RuntimeError, OSError) as error:
            export_status.value = "Export failed: " + escape(str(error))
        finally:
            export_button.disabled = False
    export_button.on_click(export)
    panel.export_button, panel.export_type, panel.export_status = export_button, export_type, export_status
    panel.widget = w.VBox([
        preset_control,
        w.HTML("The 4-mm preset is economical. Native 1-mm histories use several GB. "
               "Preset changes take effect after Run. Seed centres must be in grey/white/glial tissue."),
        section, parameter_box, panel.run_button, panel.status,
        w.HTML("<b>Read the last run · shared time and slice positions</b>"),
        player, *view.values(), w.HBox([export_type, export_button]), export_status, panel.output])
    if initial_run is not None:
        panel.last_result = initial_run
        panel.last_parameters = {key: control.value for key, control in model.items()}
        view["frame"].max = len(initial_run[1].times) - 1
        view["frame"].value = view["frame"].max
        panel.status.value = ("The CPU teaching run above is cached. Move a frame or slice "
                              "control to display it here, or press Run to compute new settings.")
    return panel
