"""Equal-axis saved-sample shapes and all predefined position diagnostics."""
import numpy as np
from matplotlib.patches import Ellipse
from tools.plot_results import plt
from tools.position_moment_metrics import ELLIPSE_THRESHOLD, METRICS, MOMENTS, principal_axes


def add_ellipse(ax, snapshot, origin, color, label):
    covariance = np.asarray(snapshot['covariance'])[:2, :2]
    values, axes = principal_axes(covariance)
    if values[0] <= 0:
        raise ValueError('cannot plot a non-positive ellipse')
    radii = np.sqrt(ELLIPSE_THRESHOLD*values)
    center = np.asarray(snapshot['mean'])[:2]-origin
    angle = np.degrees(np.arctan2(axes[1, 1], axes[0, 1]))
    ax.add_patch(Ellipse(center, 2*radii[1], 2*radii[0], angle=angle,
                         fill=False, edgecolor=color, linewidth=1.5, label=label))
    ax.plot(*center, '+', color=color, ms=8)
    # The largest radius bounds every orientation; no samples or ellipses are cropped.
    return float(np.max(np.abs(center))+radii[-1])


def save_figure(fig, path, check_budget):
    try:
        check_budget()
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches='tight')
        check_budget()
    finally:
        plt.close(fig)


def plot_results(output, reference, cases, pooled, outcomes, protocol, check_budget=lambda: None):
    recordings, times = protocol['recording_seeds'], protocol['report_times_us']
    count, seed = protocol['sample_counts'][-1], protocol['reference_seeds'][0]
    fig, axes = plt.subplots(len(recordings), len(times), figsize=(19, 8), squeeze=False)
    try:
        for i, recording in enumerate(recordings):
            for j, timestamp in enumerate(times):
                check_budget()
                ax = axes[i, j]
                case = next(c for c in cases if (c['recording_seed'], c['timestamp_us']) == (recording, timestamp))
                ax.set_title(f'{recording} · {timestamp/1e6:g} s')
                if 'baseline' not in case:
                    ax.text(.5, .5, 'unavailable', transform=ax.transAxes, ha='center')
                    continue
                with np.load(reference/f'case-{recording}/n{count}-r{seed}-t{timestamp}.npz', allow_pickle=False) as archive:
                    position = archive['states'][:, :2]
                origin = np.asarray(case['baseline']['mean'])[:2]
                difference = position-origin
                ax.scatter(difference[:, 0], difference[:, 1], s=2, alpha=.16, color='#478dc0')
                ax.plot(*(position.mean(axis=0)-origin), 'x', color='#a9500e', label='reference mean')
                bound = max(float(np.abs(difference).max()), .01)
                for method, color in [('baseline', 'black'), ('candidate', '#9d259d')]:
                    if method in case:
                        try:
                            bound = max(bound, add_ellipse(ax, case[method], origin, color, method))
                        except ValueError:
                            ax.text(.03, .97, f'{method} ellipse unavailable', transform=ax.transAxes, va='top', fontsize=6)
                bound *= 1.05
                ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
                       xlabel='x relative to baseline (m)', ylabel='y relative to baseline (m)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=6)
        fig.suptitle(f'Saved conditional samples: N={count}, seed {seed}; full ranges, equal spatial axes\n'
                     'Nominal 95% marginal ellipses. The 22 s endpoint is before returning GPS.')
        save_figure(fig, output/'position-shape.png', check_budget)
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(len(recordings), len(MOMENTS), figsize=(25, 7), squeeze=False)
    try:
        for i, recording in enumerate(recordings):
            for j, metric in enumerate(MOMENTS):
                check_budget()
                ax = axes[i, j]
                group = [o for o in outcomes if o['recording_seed'] == recording and o['metric'] == metric
                         and o['status'] == 'complete']
                for method, color in [('baseline', 'black'), ('candidate', '#9d259d')]:
                    valid = [o for o in group if method in o]
                    ax.errorbar([o['timestamp_us']/1e6 for o in valid], [o[method]['signed_error'] for o in valid],
                                yerr=[o['agreement_band'] for o in valid], fmt='.-', color=color, label=method, capsize=2)
                ax.axhline(0., color='gray', ls=':', lw=1)
                ax.set_title(f'{recording}: {metric}', fontsize=8)
                ax.set_xlabel('Time (s)'); ax.grid(alpha=.2); ax.legend(fontsize=6)
        fig.suptitle('Method minus pooled high-count reference: all seven moment metrics\n'
                     'Bars are ±(4 pooled MCSE + 1e-12), descriptive diagnostics; no trial-level calibration claim.')
        save_figure(fig, output/'moment-errors.png', check_budget)
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(len(recordings), 2, figsize=(12, 7), squeeze=False)
    try:
        for i, recording in enumerate(recordings):
            for j, metric in enumerate(METRICS[-2:]):
                check_budget()
                ax = axes[i, j]
                for sample_count in protocol['sample_counts']:
                    group = [p for p in pooled if p['recording_seed'] == recording and p['sample_count'] == sample_count
                             and p['metrics'][metric]['status'] == 'complete']
                    ax.errorbar([p['timestamp_us']/1e6 for p in group], [p['metrics'][metric]['value'] for p in group],
                        yerr=[4*p['metrics'][metric]['mcse']+1e-12 for p in group], fmt='.-', capsize=2, label=f'N={sample_count}')
                ax.axhline(.95, color='black', ls=':', lw=1, label='nominal 0.95')
                ax.set(title=f'{recording}: {metric}', xlabel='Time (s)', ylabel='Conditional sample mass', ylim=(-.03, 1.03))
                ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('Mass inside each method’s marginal position ellipse, all saved counts\n'
                     'Equal pooling of three reference seeds; bars show ±(4 MCSE + 1e-12).')
        save_figure(fig, output/'ellipse-mass.png', check_budget)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    try:
        check_budget()
        matrix = np.zeros((len(recordings)*len(times), len(METRICS)))
        labels = []
        for i, (recording, timestamp) in enumerate((s, t) for s in recordings for t in times):
            labels.append(f'{recording} · {timestamp/1e6:g} s')
            for j, metric in enumerate(METRICS):
                outcome = next(o for o in outcomes if (o['recording_seed'], o['timestamp_us'], o['metric']) == (recording, timestamp, metric))
                matrix[i, j] = 2 if outcome['resolved'] else 1 if outcome['status'] == 'complete' else 0
                text = 'resolved' if outcome['resolved'] else 'unresolved' if outcome['status'] == 'complete' else outcome['status']
                ax.text(j, i, text, ha='center', va='center', fontsize=7)
        ax.imshow(matrix, vmin=0, vmax=2, cmap='RdYlGn', aspect='auto', alpha=.55)
        ax.set_xticks(range(len(METRICS)), METRICS, rotation=30, ha='right')
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title('All 90 numerical resolution checks: stability and precision are both required\n'
                     'Resolved mismatches remain approximation failures; unresolved values are inconclusive.')
        save_figure(fig, output/'resolution.png', check_budget)
    finally:
        plt.close(fig)
