"""Plots retain every evaluated seed and show both favorable and adverse results."""
import numpy as np

from tools.plot_results import COLORS, LABELS, plt
from matplotlib.lines import Line2D


def mark_faults(axis, scenario):
    if scenario["outage_us"] is not None:
        a, b = scenario["outage_us"]
        axis.axvspan(a / 1e6, b / 1e6, color="#e2b83a", alpha=.18)
    for offset in scenario["offsets"]:
        axis.axvspan(offset["start_us"] / 1e6, offset["end_us"] / 1e6, color="#a078b4", alpha=.2)


def plot_study(output, protocol, seeds, metrics, series):
    scenarios = protocol["scenarios"]
    count = len(scenarios)
    for seed in seeds:
        fig, axes = plt.subplots(2, count, figsize=(3.7 * count, 7), squeeze=False, constrained_layout=True)
        for col, scenario in enumerate(scenarios):
            for method in COLORS:
                data = series[(seed, scenario["id"], method)]
                axes[0, col].plot(data["time_s"], data["position_error_m"], color=COLORS[method], lw=1, label=LABELS[method])
                if method != "dr":
                    axes[1, col].plot(data["time_s"], data["position_error_m"], color=COLORS[method], lw=1,
                                      linestyle="--" if method == "gated" else "-", label=LABELS[method])
            for row in range(2):
                mark_faults(axes[row, col], scenario)
                axes[row, col].grid(alpha=.2)
                axes[row, col].set(xlabel="Time (s)", ylabel="Position error (m)")
            axes[0, col].set_title(scenario["label"], fontsize=10)
            axes[1, col].axhline(protocol["settling"]["position_limit_m"], color="black", lw=.8, ls=":", label="Settling limit")
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        fig.suptitle(f"NavTrace M2 · seed {seed} · all methods above, EKF detail below\nYellow: no GPS; purple: injected GPS error")
        fig.savefig(output / f"seed-{seed}-errors.png", dpi=150)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8, count * 2), 5.5), constrained_layout=True)
    seed_offsets = np.linspace(-.05, .05, len(seeds)) if len(seeds) > 1 else [0]
    markers = ["o", "s", "^"]
    for col, scenario in enumerate(scenarios):
        for method_offset, method in zip((-.22, 0, .22), COLORS):
            values = []
            for i, seed in enumerate(seeds):
                value = next(r["position_rmse_m"] for r in metrics if r["seed"] == seed and r["scenario"] == scenario["id"] and r["method"] == method)
                values.append(value)
                axis.scatter(col + method_offset + seed_offsets[i], value, color=COLORS[method], marker=markers[i % len(markers)], s=38)
            axis.hlines(np.median(values), col + method_offset - .09, col + method_offset + .09, color=COLORS[method], lw=2)
    axis.set_yscale("log")
    axis.set_xticks(range(count), [s["label"].replace(" + ", " +\n") for s in scenarios], fontsize=9)
    axis.set(ylabel="Whole-run position RMSE (m, log scale)", title="NavTrace M2 · every seed shown; short bars are medians")
    axis.grid(axis="y", alpha=.2)
    handles = [Line2D([0], [0], color=COLORS[m], label=LABELS[m]) for m in COLORS]
    handles += [Line2D([0], [0], color="gray", linestyle="", marker=markers[i % len(markers)], label=f"Seed {seed}") for i, seed in enumerate(seeds)]
    axis.legend(handles=handles, fontsize=8, ncol=2)
    fig.savefig(output / "position-rmse.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(len(seeds), count, figsize=(3.7 * count, 2.8 * len(seeds)), squeeze=False, constrained_layout=True)
    for row, seed in enumerate(seeds):
        for col, scenario in enumerate(scenarios):
            axis = axes[row, col]
            gps = series[(seed, scenario["id"], "gated")]["gps"]
            for status, color, marker in (("accepted", "#2376ae", "."), ("rejected", "#b54233", "x")):
                selected = [r for r in gps if r["status"] == status]
                axis.scatter([int(r["timestamp_us"]) / 1e6 for r in selected], [float(r["nis"]) for r in selected],
                             s=12, color=color, marker=marker, label=status)
            axis.axhline(float(gps[0]["threshold"]), color="black", linestyle=":", lw=1)
            axis.set_yscale("symlog", linthresh=1)
            axis.set_ylim(bottom=0)
            axis.set(xlabel="Time (s)", ylabel=f"Seed {seed}\nGPS NIS")
            mark_faults(axis, scenario)
            axis.grid(alpha=.2)
            if row == 0:
                axis.set_title(scenario["label"], fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Gated EKF · squared innovation distance · dotted line: gate threshold\nNIS scale linear below 1, logarithmic above; yellow: outage; purple: injected GPS error")
    fig.savefig(output / "gate-decisions.png", dpi=150)
    plt.close(fig)
