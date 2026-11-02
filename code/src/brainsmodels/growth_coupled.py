"""Self-contained slide demonstrations using the pinned BrainWeb solver.

The numerical source is vendored unchanged; this module supplies the existing
bundled anatomy, teaching presets, synchronized views and portable exports.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brainsmodels.brainweb_data import load_brainweb, FILES
from brainsmodels._vendor.brainweb_kpp.model import Parameters, simulate

SOURCE_COMMIT = "81ecd42e6f37692a711208caa8d24361904f7e65"
SOURCE_SHA256 = "2c6da092c0b4e4eb6871094b3826485ba45b6887452094aa2bae17c6ba316e50"
ASSETS = Path(__file__).resolve().parent / "data/growth"


@dataclass
class Brain:
    t1: np.ndarray
    labels: np.ndarray
    affine: np.ndarray
    spacing: float

    @property
    def mask(self):
        return np.isin(self.labels, [1, 2, 3, 8])

    def voxel(self, xyz):
        ijk = np.rint(np.linalg.solve(self.affine[:3, :3],
                                      np.asarray(xyz) - self.affine[:3, 3])).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= self.labels.shape):
            raise ValueError("Coordinates are outside the BrainWeb image.")
        return tuple(ijk)

    def world(self, ijk):
        return self.affine[:3, :3] @ np.asarray(ijk) + self.affine[:3, 3]


def anatomy(spacing=4):
    data = load_brainweb(step_mm=spacing)
    return Brain(np.ascontiguousarray(data.t1, dtype=np.float32),
                 np.ascontiguousarray(data.labels), data.affine.copy(), float(spacing))


def preset(name="teaching"):
    """Read checked-in values, including separate slide and CPU presets."""
    definitions = json.loads((ASSETS / "presets.json").read_text())
    if name not in definitions:
        raise ValueError(f"Unknown preset: {name}")
    item = definitions[name]
    settings = dict(item["parameters"])
    for key in ("origin_mm", "radii_mm"):
        if key in settings:
            settings[key] = tuple(settings[key])
    return item["spacing_mm"], Parameters(**settings)


def run_comparison(parameters=None, *, spacing=4, progress=None):
    parameters = parameters or preset()[1]
    brain = anatomy(spacing)
    result = simulate(brain, parameters, compare=parameters.model == "coupled",
                      store_volumes=True, progress=progress)
    result.metadata.update(source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256,
                           data_files={k: {"sha256": v[0], "bytes": v[1]} for k, v in FILES.items()},
                           anatomy="normal BrainWeb phantom; tumour fields are simulated")
    return brain, result


def _plane(volume, orientation, indices):
    x, y, z = indices
    if orientation == "Axial":
        return volume[:, :, z].T, (0, 1), 2
    if orientation == "Coronal":
        return volume[:, y, :].T, (0, 2), 1
    return volume[x, :, :].T, (1, 2), 0


def _fields(result, frame):
    k = result.parameters["carrying_capacity"]
    values = [("Viable tumour", np.asarray(result.volumes[frame]) / k, "inferno", "c / K")]
    if result.oxygen_volumes is not None:
        values.extend([
            ("Oxygen availability", result.oxygen_volumes[frame], "viridis", "o / normal baseline"),
            ("Necrotic tumour material", np.asarray(result.necrosis_volumes[frame]) / k,
             "magma", "z / K"),
            ("Fisher–KPP reference", np.asarray(result.comparison.volumes[frame]) / k,
             "inferno", "c / K"),
        ])
    return values


def plot_comparison(brain, result, *, frame=-1, coordinates=None, planes="All three",
                    threshold=.02, alpha=.9):
    """One frame and one set of physical slice coordinates for every field."""
    if not 0 <= threshold <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Display threshold and opacity must lie between zero and one.")
    coordinates = result.parameters["origin_mm"] if coordinates is None else coordinates
    indices = brain.voxel(coordinates)
    actual = brain.world(indices)
    orientations = ["Axial", "Coronal", "Sagittal"] if planes == "All three" else [planes]
    if not set(orientations) <= {"Axial", "Coronal", "Sagittal"}:
        raise ValueError("Choose Axial, Coronal, Sagittal or All three.")
    fields = _fields(result, frame)
    fig = plt.figure(figsize=(max(5, 3.5 * len(fields)), 3.1 * len(orientations) + 1.2),
                     layout="constrained", facecolor="#101923")
    grid = fig.add_gridspec(len(orientations) + 1, len(fields),
                           height_ratios=[1] * len(orientations) + [.045])
    fig.suptitle(f"Tumour–brain growth · day {result.times[frame]:g} · Δx {brain.spacing:g} mm\n"
                 "Normal BrainWeb anatomy · illustrative simulation", color="white", fontsize=15)
    domain = np.isin(brain.labels, [2, 3, 8])
    for col, (title, field, cmap, units) in enumerate(fields):
        for row, orientation in enumerate(orientations):
            ax = fig.add_subplot(grid[row, col])
            background, axes, axis = _plane(brain.t1, orientation, indices)
            image, _, _ = _plane(field, orientation, indices)
            mask, _, _ = _plane(domain, orientation, indices)
            extent = []
            for a in axes:
                low = brain.affine[a, 3] - brain.spacing / 2
                extent.extend([low, low + brain.labels.shape[a] * brain.spacing])
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent)
            hidden = ~mask if units.startswith("o /") else (~mask | (image < threshold))
            overlay = ax.imshow(np.ma.array(image, mask=hidden), cmap=cmap, vmin=0, vmax=1,
                                origin="lower", extent=extent, alpha=alpha)
            ax.set_title((title + "\n" if row == 0 else "") +
                         f"{orientation} · {'xyz'[axis]} = {actual[axis]:g} mm",
                         color="white", fontsize=10)
            left, right, top, bottom = {"Axial": ("L", "R", "A", "P"),
                                        "Coronal": ("L", "R", "S", "I"),
                                        "Sagittal": ("P", "A", "S", "I")}[orientation]
            for text, x, y in [(left, .02, .5), (right, .98, .5), (top, .5, .98), (bottom, .5, .02)]:
                ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
                        color="white", fontsize=7)
            ax.set_axis_off()
        bar = fig.colorbar(overlay, cax=fig.add_subplot(grid[-1, col]), orientation="horizontal")
        bar.set_label(units, color="white", fontsize=9)
        bar.ax.tick_params(colors="white", labelsize=8)
    return fig


def plot_metrics(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    axes[0].plot(result.times, result.metrics["cells"], label="Viable tumour")
    if result.comparison is not None:
        axes[0].plot(result.times, result.comparison.metrics["cells"], label="Fisher–KPP reference")
        axes[0].plot(result.times, result.metrics["necrotic_cells"], label="Necrotic equivalents")
        axes[1].plot(result.times, result.metrics["min_oxygen"], label="Minimum tissue oxygen")
        axes[1].axhline(result.parameters["oxygen_death_level"], color="red", ls=":", label="Death threshold")
        axes[1].axhline(result.parameters["oxygen_prolif_level"], color="gray", ls=":", label="Proliferation threshold")
        axes[1].set(ylabel="Oxygen / normal baseline", ylim=(0, 1.05))
    else:
        axes[1].plot(result.times, result.metrics["volume_20_ml"], label="c ≥ 0.2 K")
        axes[1].plot(result.times, result.metrics["volume_80_ml"], label="c ≥ 0.8 K")
        axes[1].set(ylabel="Threshold volume (mL)")
    axes[0].set(ylabel="Model-implied cells / cell equivalents")
    for ax in axes:
        ax.set_xlabel("Time (days)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    return fig


def export_run(brain, result, directory, *, coordinates=None, planes="Axial", threshold=.02,
               alpha=.9, animation=None, frame=-1):
    """Save a poster, metrics and provenance; optionally GIF or bundled-FFmpeg MP4.

    Every frame is rendered from the same saved histories; no simulation reruns.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib as mpl

    if animation not in (None, "gif", "mp4"):
        raise ValueError("Animation must be None, gif or mp4.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = dict(coordinates=coordinates, planes=planes, threshold=threshold, alpha=alpha)
    fig = plot_comparison(brain, result, frame=frame, **settings)
    fig.savefig(directory / "comparison.png", dpi=120)
    plt.close(fig)
    from importlib.metadata import version
    chosen = result.parameters["origin_mm"] if coordinates is None else coordinates
    record = dict(parameters=result.parameters, metadata=result.metadata,
                  libraries={name: version(name) for name in
                             ("numpy", "scipy", "numba", "matplotlib", "nibabel")},
                  times_days=result.times.tolist(),
                  metrics={k: v.tolist() for k, v in result.metrics.items()},
                  view={**settings, "coordinates": list(chosen),
                        "frame": frame % len(result.times), "day": float(result.times[frame])})
    if result.comparison:
        record["reference_metrics"] = {k: v.tolist() for k, v in result.comparison.metrics.items()}
    (directory / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if animation:
        # Render a raster into a fixed canvas so shared scales and slice positions
        # remain identical across all movie frames, including the final hold.
        def raster(i):
            panel = plot_comparison(brain, result, frame=i, **settings)
            panel.canvas.draw()
            pixels = np.asarray(panel.canvas.buffer_rgba()).copy()
            plt.close(panel)
            return pixels
        figure, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        figure.subplots_adjust(0, 0, 1, 1)
        picture = ax.imshow(raster(0))
        ax.set_axis_off()
        frames = list(range(len(result.times))) + [len(result.times) - 1] * 8
        movie = FuncAnimation(figure, lambda i: (picture.set_data(raster(i)),),
                              frames=frames, interval=200, blit=False)
        try:
            if animation == "gif":
                movie.save(directory / "comparison.gif", writer=PillowWriter(fps=5))
            else:
                from imageio_ffmpeg import get_ffmpeg_exe
                with mpl.rc_context({"animation.ffmpeg_path": get_ffmpeg_exe()}):
                    movie.save(directory / "comparison.mp4", writer=FFMpegWriter(fps=5))
        finally:
            plt.close(figure)
    return directory
