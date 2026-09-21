"""Plot actual saved C++ trajectories and independently calculated errors."""
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "navtrace-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.evaluate import aligned_poses

COLORS = {"dr": "#b85b36", "ekf": "#2376ae", "gated": "#278154"}
LABELS = {"dr": "Dead reckoning", "ekf": "Basic EKF", "gated": "Gated EKF"}


def plot_results(output, errors, outage_start_s, outage_end_s):
    output = Path(output)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, scenario in zip(axes, ("nominal", "dropout")):
        for method in COLORS:
            times, estimate, truth = aligned_poses(output / "inputs/truth.csv", output / f"{scenario}-{method}.csv")
            ax.plot(estimate[:, 0], estimate[:, 1], color=COLORS[method], lw=1.6,
                    linestyle="--" if method == "gated" else "-", label=LABELS[method])
        ax.plot(truth[:, 0], truth[:, 1], color="#222222", lw=1.5, label="Analytical truth")
        window = (times / 1e6 >= outage_start_s) & (times / 1e6 < outage_end_s)
        ax.plot(truth[window, 0], truth[window, 1], color="#e2b83a", lw=5, alpha=.45,
                label=f"{outage_start_s:g}–{outage_end_s:g} s reference window" if scenario == "nominal" else "GPS outage truth segment")
        ax.scatter(truth[0, 0], truth[0, 1], color="black", marker="o", s=25, zorder=5)
        ax.set(title=scenario.capitalize(), xlabel="World x (m)", ylabel="World y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("NavTrace · C++ replay trajectories · development seed")
    fig.savefig(output / "trajectories.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for col, scenario in enumerate(("nominal", "dropout")):
        for method in COLORS:
            times, position, heading = errors[f"{scenario}-{method}"]
            for row, values in ((0, position), (2, np.rad2deg(heading))):
                axes[row, col].plot(times, values, color=COLORS[method], lw=1.2,
                                   linestyle="--" if method == "gated" else "-", label=LABELS[method])
            if method != "dr":
                axes[1, col].plot(times, position, color=COLORS[method], lw=1.2,
                                  linestyle="--" if method == "gated" else "-", label=LABELS[method])
        for row in range(3):
            axes[row, col].axvspan(outage_start_s, outage_end_s, color="#e2b83a", alpha=.16,
                                 label="Reference window" if scenario == "nominal" else "GPS outage")
            axes[row, col].grid(alpha=.2)
        axes[0, col].set(title=scenario.capitalize(), ylabel="Position error (m)")
        axes[1, col].set(ylabel="EKF position error detail (m)")
        axes[2, col].set(xlabel="Time (s)", ylabel="Wrapped heading error (deg)")
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=8)
    fig.suptitle("NavTrace · errors at gyro timestamps after simultaneous updates")
    fig.savefig(output / "errors.png", dpi=170)
    plt.close(fig)
