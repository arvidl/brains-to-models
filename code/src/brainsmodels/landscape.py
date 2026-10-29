r"""A 2-D energy landscape as the schematic of multistable descent.

The surface is a teaching slice of activity space — two coordinates, a
handful of wells — after the Kandel-style schematic and the KI-i-helsefagene
notebook. It is not a Hopfield net and not \(5\cdot 10^9\) neurons.
Notebook 05 is the controlled demonstration (Hopfield, Necker, attention). This module
is the map: \(U(\mathbf{X})\) with a gradient path into an attractor.

Companion: ``code/notebooks/05b_multistable_animation.ipynb``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.typing import NDArray

from brainsmodels.repro import DEFAULT_SEED, set_seed

FloatArray = NDArray[np.floating]

_TEAL = "#2c5f6e"
_YELLOW = "#f5c400"
_BLACK = "0.08"
_GREEN = "#1b8a3a"
_RED = "#c0392b"
_INK = "0.15"

# Known local minima on the plotted square (x, y).
ATTRACTORS: tuple[tuple[float, float], ...] = (
    (0.442, -2.638),
    (-1.672, 0.128),
    (2.119, 2.850),
)
# Front well from the teaching-slide camera (elev=22, azim=45).
ATTRACTOR = ATTRACTORS[2]
DEFAULT_START = (0.80, 4.80)
MESH_LIM = 6.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    return repo_root() / "paper" / "figures"


def default_animation_dir() -> Path:
    return repo_root() / "code" / "animations"


def energy(x: FloatArray | float, y: FloatArray | float) -> FloatArray | float:
    """Smooth multi-well U(x, y). Same family as the KI-i-helsefagene demo."""
    return (
        (np.sin(x) + np.sin(0.5 * x)) * np.cos(y)
        + (np.cos(1.5 * x) + np.cos(0.75 * x)) * np.sin(0.5 * y)
        + 0.1 * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
    )


def energy_grad(x: float, y: float) -> tuple[float, float]:
    dx = (
        (np.cos(x) + 0.5 * np.cos(0.5 * x)) * np.cos(y)
        + (-1.5 * np.sin(1.5 * x) - 0.75 * np.sin(0.75 * x)) * np.sin(0.5 * y)
        + 0.2 * x
    )
    dy = (
        (np.sin(x) + np.sin(0.5 * x)) * (-np.sin(y))
        + (np.cos(1.5 * x) + np.cos(0.75 * x)) * 0.5 * np.cos(0.5 * y)
        + 0.2 * y
    )
    return float(dx), float(dy)


def mesh(lim: float = MESH_LIM, n: int = 90) -> tuple[FloatArray, FloatArray, FloatArray]:
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.asarray(energy(xx, yy), dtype=float)


def nearest_attractor(
    xy: tuple[float, float],
    radius: float = 0.5,
) -> tuple[float, float] | None:
    """Return a known well if ``xy`` is inside ``radius``, else None."""
    x, y = float(xy[0]), float(xy[1])
    best: tuple[float, float] | None = None
    best_d = radius * radius
    for ax_, ay_ in ATTRACTORS:
        d = (x - ax_) ** 2 + (y - ay_) ** 2
        if d < best_d:
            best_d = d
            best = (ax_, ay_)
    return best


def descend(
    start: tuple[float, float] = DEFAULT_START,
    n_steps: int = 160,
    lr: float = 0.08,
    tol: float = 2e-4,
) -> FloatArray:
    """Gradient descent on U until a local min, clipped to the plotted square."""
    x, y = float(start[0]), float(start[1])
    pts: list[tuple[float, float]] = []
    for _ in range(n_steps):
        pts.append((x, y))
        gx, gy = energy_grad(x, y)
        if gx * gx + gy * gy < tol * tol:
            break
        x = float(np.clip(x - lr * gx, -MESH_LIM, MESH_LIM))
        y = float(np.clip(y - lr * gy, -MESH_LIM, MESH_LIM))
    well = nearest_attractor((x, y), radius=0.5)
    if well is not None:
        pts.append(well)
    return np.asarray(pts, dtype=float)


def trim_to_attractor(path: FloatArray, dist: float = 0.12) -> FloatArray:
    """Skip remaining descent steps once the state is already in the well."""
    d = np.linalg.norm(path - np.asarray(ATTRACTOR), axis=1)
    hit = np.flatnonzero(d < dist)
    if len(hit) == 0:
        return path
    return np.vstack([path[: hit[0] + 1], ATTRACTOR])


def resample_path(path: FloatArray, n: int, hold: int = 8) -> FloatArray:
    """Arc-length sample a descent and linger briefly on the attractor."""
    path = trim_to_attractor(path)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(path[-1][None, :], n, axis=0)
    n_move = max(2, n - hold)
    s = np.linspace(0.0, cum[-1], n_move)
    moved = np.column_stack(
        [np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])]
    )
    held = np.repeat(path[-1][None, :], hold, axis=0)
    return np.vstack([moved, held])


def _surface(ax: plt.Axes, xx: FloatArray, yy: FloatArray, zz: FloatArray) -> None:
    ax.computed_zorder = False
    ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="turbo",
        linewidth=0.0,
        antialiased=True,
        rstride=2,
        cstride=2,
        edgecolor="none",
        alpha=0.90,
        shade=True,
        zorder=1,
    )
    ax.plot_wireframe(
        xx,
        yy,
        zz,
        rstride=5,
        cstride=5,
        color="0.25",
        linewidth=0.25,
        alpha=0.35,
        zorder=2,
    )
    ax.set_xlabel("activity pattern", labelpad=6, fontsize=11)
    ax.set_ylabel("")
    ax.set_zlabel(r"$U(\mathbf{X}_t)$", labelpad=4, fontsize=11)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=22, azim=45)
    ax.set_box_aspect((1.0, 1.0, 0.48))
    ax.tick_params(pad=-2)


def _path_xyz(path: FloatArray, lift: float = 0.38) -> tuple[FloatArray, FloatArray]:
    z = np.asarray(energy(path[:, 0], path[:, 1]), dtype=float) + lift
    return path, z


def _draw_arrow(
    ax: plt.Axes,
    path: FloatArray,
    z: FloatArray,
    head: bool = True,
) -> None:
    ax.plot(path[:, 0], path[:, 1], z, color=_YELLOW, lw=3.4, zorder=20)
    ax.scatter(path[0, 0], path[0, 1], z[0], c=_YELLOW, s=36, zorder=21, edgecolors="0.2", linewidths=0.3)
    if not head or len(path) < 4:
        return
    k = min(5, len(path) - 1)
    p0, p1 = path[-1 - k], path[-1]
    ax.quiver(
        p0[0],
        p0[1],
        z[-1 - k],
        p1[0] - p0[0],
        p1[1] - p0[1],
        z[-1] - z[-1 - k],
        color=_YELLOW,
        arrow_length_ratio=0.55,
        linewidth=2.4,
        zorder=21,
    )


def _annotate(ax: plt.Axes, path: FloatArray, z: FloatArray, full: bool = True) -> None:
    ax.text(
        path[0, 0] - 0.2,
        path[0, 1] + 0.55,
        z[0] + 0.85,
        "Gradient descent",
        fontsize=10,
        color=_INK,
        zorder=22,
    )
    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        z[-1],
        c=_TEAL,
        s=80,
        edgecolors="white",
        linewidths=0.5,
        zorder=22,
    )
    note = "attractor\n(memory, percept,\ndecision)" if full else "attractor"
    ax.text(
        path[-1, 0] + 0.55,
        path[-1, 1] - 1.6,
        z[-1] + 0.25,
        note,
        fontsize=9,
        color=_INK,
        zorder=22,
    )


def render_paper_figure(path: FloatArray | None = None) -> plt.Figure:
    xx, yy, zz = mesh()
    if path is None:
        path = descend()
    _, z = _path_xyz(path)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    _surface(ax, xx, yy, zz)
    _draw_arrow(ax, path, z)
    _annotate(ax, path, z, full=True)
    ax.text2D(
        0.0,
        1.02,
        r"A   Descent on $U(\mathbf{X})$  ·  a 2-D slice of activity space",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=_INK,
        va="top",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
    return fig


def write_figure(
    out_dir: Path | str | None = None,
    path: FloatArray | None = None,
) -> Path:
    out = Path(out_dir) if out_dir is not None else default_figure_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig = render_paper_figure(path)
    pdf = out / "energy_landscape.pdf"
    png = out / "energy_landscape.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


def _ffmpeg_path() -> Path | None:
    found = shutil.which("ffmpeg")
    for candidate in (found, "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def write_animation(
    out_dir: Path | str | None = None,
    n_frames: int = 56,
    dpi: int = 100,
) -> Path:
    """GIF of a state sliding into a well, plus MP4 when ffmpeg is on the PATH."""
    out = Path(out_dir) if out_dir is not None else default_animation_dir()
    out.mkdir(parents=True, exist_ok=True)
    hold = 8
    path = resample_path(descend(), n_frames, hold=hold)
    xx, yy, zz = mesh(n=70)
    _, z = _path_xyz(path)

    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    _surface(ax, xx, yy, zz)
    _annotate(ax, path, z, full=False)
    (trail,) = ax.plot([], [], [], color=_BLACK, lw=2.6, zorder=10)
    point = ax.scatter([], [], [], c=_BLACK, s=50, edgecolors="0.85", linewidths=0.3, zorder=12)
    ax.set_title(r"Gradient descent on $U(\mathbf{X})$", fontsize=12, pad=4)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    def _frame(i: int) -> tuple:
        sl = path[: i + 1]
        trail.set_data(sl[:, 0], sl[:, 1])
        trail.set_3d_properties(z[: i + 1])
        point._offsets3d = ([path[i, 0]], [path[i, 1]], [z[i]])
        arrived = i >= n_frames - hold
        point.set_color(_TEAL if arrived else _BLACK)
        return trail, point

    anim = animation.FuncAnimation(fig, _frame, frames=n_frames, interval=110, blit=False)
    gif = out / "energy_landscape.gif"
    anim.save(gif, writer="pillow", dpi=dpi)
    print(f"wrote {gif}")

    mp4 = out / "energy_landscape.mp4"
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        print("skipped mp4 (ffmpeg not found)")
    else:
        plt.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
        writer = animation.FFMpegWriter(
            fps=10,
            bitrate=2800,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(mp4, writer=writer, dpi=max(dpi, 130))
        print(f"wrote {mp4}  ({ffmpeg})")
    plt.close(fig)
    return gif


def render_choice_figure(start: tuple[float, float]) -> plt.Figure:
    """2-D map (click target) plus the 3-D descent from ``start``."""
    path = descend(start=start, n_steps=200)
    z_path = np.asarray(energy(path[:, 0], path[:, 1]), dtype=float)
    xx, yy, zz = mesh(n=70)
    lift = 0.28

    fig = plt.figure(figsize=(11.2, 4.8))
    ax0 = fig.add_subplot(1, 2, 1)
    filled = ax0.contourf(xx, yy, zz, levels=22, cmap="turbo")
    ax0.contour(xx, yy, zz, levels=10, colors="0.15", linewidths=0.35)
    cbar = fig.colorbar(filled, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"$U(\mathbf{X})$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    for well in ATTRACTORS:
        ax0.scatter(well[0], well[1], c="0.85", s=36, zorder=5, edgecolors="0.3", linewidths=0.5)
    ax0.plot(path[:, 0], path[:, 1], color=_BLACK, lw=2.0, zorder=6)
    ax0.scatter(path[0, 0], path[0, 1], c=_GREEN, s=42, zorder=7, edgecolors="white", linewidths=0.4)
    ax0.scatter(path[-1, 0], path[-1, 1], c=_RED, s=58, zorder=7, edgecolors="white", linewidths=0.5)
    ax0.set_aspect("equal")
    ax0.set_xlim(-MESH_LIM, MESH_LIM)
    ax0.set_ylim(-MESH_LIM, MESH_LIM)
    ax0.set_xlabel("activity pattern")
    ax0.set_title("Click to drop a start", fontsize=11)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.computed_zorder = False
    ax1.plot_surface(
        xx, yy, zz, cmap="turbo", linewidth=0.0, antialiased=True,
        rstride=2, cstride=2, edgecolor="none", alpha=0.88, shade=True, zorder=1,
    )
    ax1.plot(path[:, 0], path[:, 1], z_path + lift, color=_BLACK, lw=2.4, zorder=10)
    ax1.scatter(path[0, 0], path[0, 1], z_path[0] + lift, c=_GREEN, s=36, zorder=11, edgecolors="white", linewidths=0.4)
    ax1.scatter(path[-1, 0], path[-1, 1], z_path[-1] + lift, c=_RED, s=55, zorder=12, edgecolors="white", linewidths=0.4)
    ax1.set_xlabel("activity pattern", labelpad=4, fontsize=9)
    ax1.set_ylabel("")
    ax1.set_zlabel(r"$U(\mathbf{X}_t)$", labelpad=2, fontsize=9)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.view_init(elev=22, azim=45)
    ax1.set_box_aspect((1.0, 1.0, 0.48))
    ax1.set_title("Descent on $U$", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.90, wspace=0.12)
    fig._choice_ax = ax0  # type: ignore[attr-defined]
    return fig


def explore_descent() -> None:
    """ipywidgets: drop a start on the 2-D map; rotate the 3-D panel with the mouse."""
    import sys

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "Interactive basins need ipywidgets. In this repo: `uv add ipywidgets`. "
            "On Colab, run `pip install ipywidgets ipympl` if widget mode is unavailable."
        ) from exc

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            if "google.colab" in sys.modules:
                from google.colab import output as colab_output

                colab_output.enable_custom_widget_manager()
            ip.run_line_magic("matplotlib", "widget")
    except Exception:
        pass

    x_w = widgets.FloatSlider(
        value=DEFAULT_START[0], min=-MESH_LIM, max=MESH_LIM, step=0.05,
        description="x", continuous_update=False, readout_format=".2f",
    )
    y_w = widgets.FloatSlider(
        value=DEFAULT_START[1], min=-MESH_LIM, max=MESH_LIM, step=0.05,
        description="y", continuous_update=False, readout_format=".2f",
    )
    hint = widgets.HTML(
        "<i>Click the contour to drop a start (green). The path ends at a red attractor. "
        "Drag the 3-D panel to rotate — the camera stays when you pick a new start.</i>"
    )
    stats = widgets.HTML("")
    busy = {"on": False}
    lift = 0.28

    xx, yy, zz = mesh(n=70)
    fig = plt.figure(figsize=(11.2, 4.8))
    ax0 = fig.add_subplot(1, 2, 1)
    filled = ax0.contourf(xx, yy, zz, levels=22, cmap="turbo")
    ax0.contour(xx, yy, zz, levels=10, colors="0.15", linewidths=0.35)
    cbar = fig.colorbar(filled, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"$U(\mathbf{X})$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    for well in ATTRACTORS:
        ax0.scatter(well[0], well[1], c="0.85", s=36, zorder=5, edgecolors="0.3", linewidths=0.5)
    (line0,) = ax0.plot([], [], color=_BLACK, lw=2.0, zorder=6)
    start0 = ax0.scatter([], [], c=_GREEN, s=42, zorder=7, edgecolors="white", linewidths=0.4)
    end0 = ax0.scatter([], [], c=_RED, s=58, zorder=7, edgecolors="white", linewidths=0.5)
    ax0.set_aspect("equal")
    ax0.set_xlim(-MESH_LIM, MESH_LIM)
    ax0.set_ylim(-MESH_LIM, MESH_LIM)
    ax0.set_xlabel("activity pattern")
    ax0.set_title("Click to drop a start", fontsize=11)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.computed_zorder = False
    ax1.plot_surface(
        xx, yy, zz, cmap="turbo", linewidth=0.0, antialiased=True,
        rstride=2, cstride=2, edgecolor="none", alpha=0.88, shade=True, zorder=1,
    )
    (line1,) = ax1.plot([], [], [], color=_BLACK, lw=2.4, zorder=10)
    start1 = ax1.scatter([], [], [], c=_GREEN, s=36, zorder=11, edgecolors="white", linewidths=0.4)
    end1 = ax1.scatter([], [], [], c=_RED, s=55, zorder=12, edgecolors="white", linewidths=0.4)
    ax1.set_xlabel("activity pattern", labelpad=4, fontsize=9)
    ax1.set_ylabel("")
    ax1.set_zlabel(r"$U(\mathbf{X}_t)$", labelpad=2, fontsize=9)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.view_init(elev=22, azim=45)
    ax1.set_box_aspect((1.0, 1.0, 0.48))
    ax1.set_title("Descent on $U$  ·  drag to rotate", fontsize=11)
    ax1.mouse_init()
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.90, wspace=0.12)

    def _set_path() -> None:
        start = (float(x_w.value), float(y_w.value))
        path = descend(start=start, n_steps=200)
        z = np.asarray(energy(path[:, 0], path[:, 1]), dtype=float) + lift
        z0 = float(energy(start[0], start[1]))
        z1 = float(energy(float(path[-1, 0]), float(path[-1, 1])))
        stats.value = (
            f"<code>start ({start[0]:+.2f}, {start[1]:+.2f})  U={z0:+.3f}  →  "
            f"attractor ({path[-1, 0]:+.3f}, {path[-1, 1]:+.3f})  U={z1:+.3f}</code>"
        )
        line0.set_data(path[:, 0], path[:, 1])
        start0.set_offsets(path[0:1])
        end0.set_offsets(path[-1:])
        line1.set_data(path[:, 0], path[:, 1])
        line1.set_3d_properties(z)
        start1._offsets3d = ([path[0, 0]], [path[0, 1]], [z[0]])
        end1._offsets3d = ([path[-1, 0]], [path[-1, 1]], [z[-1]])
        fig.canvas.draw_idle()

    def _click(event) -> None:
        if busy["on"] or event.inaxes is not ax0:
            return
        if event.xdata is None or event.ydata is None:
            return
        busy["on"] = True
        x_w.value = float(np.clip(event.xdata, -MESH_LIM, MESH_LIM))
        y_w.value = float(np.clip(event.ydata, -MESH_LIM, MESH_LIM))
        busy["on"] = False
        _set_path()

    def _on_change(_change=None) -> None:
        if busy["on"]:
            return
        _set_path()

    fig.canvas.mpl_connect("button_press_event", _click)
    x_w.observe(_on_change, names="value")
    y_w.observe(_on_change, names="value")
    display(hint, widgets.HBox([x_w, y_w]), stats)
    _set_path()
    plt.show()


def save_paper_figure(
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    animate: bool = True,
) -> Path:
    set_seed(seed)
    path = descend()
    pdf = write_figure(out_dir, path=path)
    if animate:
        write_animation()
    return pdf


def main() -> None:
    save_paper_figure()


if __name__ == "__main__":
    main()
