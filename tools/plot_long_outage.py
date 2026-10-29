"""Plots of actual C++ outputs and independent ensemble error geometry."""
from tools.plot_results import plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2


def plot_traces(output, seed, traces):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    error_fig, error_axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for column, case in enumerate(['nominal', 'dropout']):
        plotted_truth = False
        for method, color in [('ekf', '#2877b5'), ('gated', '#d47816')]:
            if (case, method) not in traces:
                continue
            times, states, _, truth, stats, _ = traces[(case, method)]
            if not plotted_truth:
                axes[column].plot(truth[:, 0], truth[:, 1], color='black', ls='--', label='truth')
                during = (times >= 2000000) & (times < 22000000)
                if case == 'dropout':
                    axes[column].plot(truth[during, 0], truth[during, 1], color='gold', lw=4, alpha=.4, label='GPS absent')
                plotted_truth = True
            axes[column].plot(states[:, 0], states[:, 1], color=color, label=method, alpha=.8)
            error_axes[0, column].plot(times/1e6, np.linalg.norm(states[:, :2]-truth[:, :2], axis=1), color=color, label=method)
            error_axes[1, column].plot(times/1e6, stats['heading_error'], color=color, label=method)
        axes[column].set(title=case, xlabel='World x (m)', ylabel='World y (m)', aspect='equal')
        axes[column].grid(alpha=.2); axes[column].legend()
        for row in range(2):
            ax = error_axes[row, column]
            ax.set_title(case); ax.grid(alpha=.2); ax.legend()
            if case == 'dropout':
                ax.axvspan(2, 22, color='gold', alpha=.2)
        error_axes[1, column].set_xlabel('Time (s)')
    error_axes[0, 0].set_ylabel('Position error (m)')
    error_axes[1, 0].set_ylabel('Wrapped yaw error (rad)')
    fig.suptitle(f'Matched-model trajectories · predefined seed {seed}')
    error_fig.suptitle(f'Measured error · predefined seed {seed} · yellow: GPS absent')
    for figure, name in [(fig, 'trajectories'), (error_fig, 'errors')]:
        figure.tight_layout(); figure.savefig(output/f'{name}.png', dpi=150); plt.close(figure)


def plot_ensemble(output, ensemble, geometry, points):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, case in enumerate(['nominal', 'dropout']):
        for row, statistic in enumerate(['position', 'full']):
            ax = axes[row, column]
            for method, color in [('ekf', '#2877b5'), ('gated', '#d47816')]:
                group = [r for r in ensemble if (r['scenario'], r['method'], r['statistic'], r['phase'])
                         == (case, method, statistic, 'post_events')]
                times = [r['timestamp_us']/1e6 for r in group]
                values = [np.nan if r['mean_nees'] is None else r['mean_nees'] for r in group]
                ax.plot(times, values, '.-', color=color, label=method)
                if method == 'ekf' and all(r['reference_lower'] is not None for r in group):
                    ax.fill_between(times, [r['reference_lower'] for r in group],
                                    [r['reference_upper'] for r in group], color='gray', alpha=.2, label='95% Gaussian reference')
            if case == 'dropout':
                ax.axvspan(2, 22, color='gold', alpha=.12)
            ax.set_title(f'{case}: {statistic} NEES'); ax.grid(alpha=.2); ax.legend(fontsize=8)
            ax.set_yscale('symlog', linthresh=1); ax.set_ylim(bottom=0)
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 1].set_xlabel('Time (s)')
    fig.suptitle('One sample per independent seed at each fixed time\nNonlinear Gaussian reference is approximate; gated comparison is descriptive; full ranges shown')
    fig.tight_layout(); fig.savefig(output/'ensemble-nees.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, case in enumerate(['nominal', 'dropout']):
        for column, method in enumerate(['ekf', 'gated']):
            ax = axes[row, column]
            group = [p for p in points if (p['scenario'], p['method'], p['phase']) == (case, method, 'pre_gps')]
            ax.scatter([p.get('error_x', np.nan) for p in group], [p.get('error_y', np.nan) for p in group],
                       s=15, color='black', label='independent errors')
            stats = next(r for r in geometry if (r['scenario'], r['method'], r['phase']) == (case, method, 'pre_gps'))
            bound = max([abs(p.get(key, 0)) for p in group for key in ['error_x', 'error_y']] + [.1])
            for field, center, color, label in [
                    ('mean_reported_covariance', [0, 0], '#2877b5', 'mean reported P, centered at zero'),
                    ('centered_sample_covariance', stats['mean_error'], '#d47816', 'empirical covariance, centered at mean error')]:
                if stats[field] is None:
                    continue
                values, vectors = np.linalg.eigh(stats[field])
                # A zero empirical eigenvalue is possible (e.g. identical errors).
                radii = np.sqrt(chi2.ppf(.95, 2)*np.maximum(values, 0))
                angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
                ax.add_patch(Ellipse(center, 2*radii[1], 2*radii[0], angle=angle,
                                     fill=False, edgecolor=color, label=label))
                bound = max(bound, float(np.max(np.abs(center)))+float(radii[-1]))
            ax.set(xlim=(-bound*1.1, bound*1.1), ylim=(-bound*1.1, bound*1.1), aspect='equal',
                   title=f'{case} / {method}: {stats["valid_runs"]}/{stats["planned_runs"]} valid',
                   xlabel='World x error (m)', ylabel='World y error (m)')
            ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle('22 s immediately BEFORE GPS correction: estimate minus truth\nEllipses use the 2D 95% Gaussian scale as geometric summaries, not a shared confidence region')
    fig.tight_layout(); fig.savefig(output/'return-geometry.png', dpi=150); plt.close(fig)
