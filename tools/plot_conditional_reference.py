"""Show sampled position shape, directional moments and Monte Carlo variation."""
from tools.plot_results import plt
from matplotlib.patches import Ellipse
import numpy as np


def plot_results(output, rows, recordings, counts, seeds):
    valid = [r for r in rows if r['status'] == 'complete']
    times = sorted({r['timestamp_us'] for r in rows})
    fig, axes = plt.subplots(len(recordings), len(times), figsize=(18, 4*len(recordings)), squeeze=False)
    for i, recording in enumerate(recordings):
        for j, timestamp in enumerate(times):
            ax = axes[i, j]
            selected = [r for r in valid if (r['recording_seed'], r['sample_count'], r['reference_seed'], r['timestamp_us'])
                        == (recording, max(counts), seeds[0], timestamp)]
            if not selected:
                ax.set_title('unavailable'); continue
            row = selected[0]
            samples = np.load(output/f'case-{recording}/n{max(counts)}-r{seeds[0]}-t{timestamp}.npz')['states']
            difference = samples[:, :2]-[row['ekf_x_m'], row['ekf_y_m']]
            ax.scatter(difference[:, 0], difference[:, 1], s=2, alpha=.18, color='#2877b5')
            values, vectors = np.array(row['ekf_eigenvalues']), np.array(row['ekf_eigenvectors'])
            radii = np.sqrt(5.991464547107979*values)
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            ax.add_patch(Ellipse((0, 0), 2*radii[1], 2*radii[0], angle=angle, fill=False, edgecolor='black', lw=1.5))
            ax.plot(row['mean_difference_x_m'], row['mean_difference_y_m'], '+', color='#d47816', ms=9)
            bound = 1.1*max(float(np.abs(difference).max()), float(radii[-1]), .01)
            ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
                   xlabel='World x relative to EKF (m)', ylabel='World y relative to EKF (m)',
                   title=f'{recording} · {timestamp/1e6:g} s')
            ax.grid(alpha=.2)
    fig.suptitle(f'Conditional position samples, N={max(counts)}, reference seed {seeds[0]}\nBlack: nominal EKF 95% ellipse · orange +: reference mean · endpoint at 22 s is PRE-GPS')
    fig.tight_layout(); fig.savefig(output/'position-shape.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(len(recordings), 3, figsize=(14, 4*len(recordings)), squeeze=False)
    for i, recording in enumerate(recordings):
        for seed in seeds:
            group = sorted([r for r in valid if r['recording_seed'] == recording and r['sample_count'] == max(counts)
                            and r['reference_seed'] == seed], key=lambda r:r['timestamp_us'])
            t = [r['timestamp_us']/1e6 for r in group]
            axes[i, 0].plot(t, [r['mean_difference_norm_m'] for r in group], '.-', label=str(seed))
            for j, name in enumerate(['narrow', 'wide'], start=1):
                axes[i, j].plot(t, [r[f'variance_{name}_m2']/r[f'ekf_variance_{name}_m2'] for r in group], '.-', label=f'{seed}: own mean')
                axes[i, j].plot(t, [r[f'second_moment_{name}_m2']/r[f'ekf_variance_{name}_m2'] for r in group], '--', label=f'{seed}: EKF mean')
        axes[i, 0].set_title(f'{recording}: mean displacement (m)')
        for j, name in enumerate(['narrow', 'wide'], start=1):
            axes[i, j].set_title(f'{recording}: {name} directional moment / EKF variance')
            axes[i, j].set_yscale('log'); axes[i, j].axhline(1., color='black', ls=':', lw=1)
        for ax in axes[i]:
            ax.set_xlabel('Time (s), after available wheel/gyro'); ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle('Highest sample count, all independent reference seeds\nSolid moments use the reference mean; dashed moments include displacement from the EKF mean')
    fig.tight_layout(); fig.savefig(output/'distribution-over-time.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(len(recordings), 3, figsize=(13, 4*len(recordings)), squeeze=False)
    for i, recording in enumerate(recordings):
        for seed in seeds:
            group = sorted([r for r in valid if r['recording_seed'] == recording and r['reference_seed'] == seed
                            and r['timestamp_us'] == max(times)], key=lambda r:r['sample_count'])
            n = [r['sample_count'] for r in group]
            axes[i, 0].plot(n, [r['mean_difference_norm_m'] for r in group], '.-', label=str(seed))
            axes[i, 1].plot(n, [r['variance_narrow_m2']/r['ekf_variance_narrow_m2'] for r in group], '.-', label=str(seed))
            axes[i, 2].plot(n, [r['ellipse_mass'] for r in group], '.-', label=str(seed))
        for ax, name in zip(axes[i], ['Mean displacement (m)', 'Narrow variance ratio', 'Mass inside EKF ellipse']):
            ax.set_title(f'{recording}: {name}'); ax.set_xscale('log', base=2); ax.set_xlabel('Independent path samples')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[i, 2].set_ylim(0, 1)
    fig.suptitle('22 s PRE-GPS convergence: three sample counts and independent random seeds\nTwo consumed recordings, not independent navigation trials per path; ellipse mass is not ground-truth coverage')
    fig.tight_layout(); fig.savefig(output/'convergence.png', dpi=160, bbox_inches='tight'); plt.close(fig)
