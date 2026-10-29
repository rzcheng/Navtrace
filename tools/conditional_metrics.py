"""Distribution comparisons and finite-sample numerical-resolution checks."""
import numpy as np

from tools.conditional_reference import circular_summary, factor_covariance, multiply_samples


METRICS = ['mean_x_m', 'mean_y_m', 'variance_narrow_m2', 'variance_wide_m2',
           'second_moment_narrow_m2', 'second_moment_wide_m2', 'ellipse_mass']


def distribution_metrics(samples, ekf_mean, ekf_covariance):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 6 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError('expected finite N x 6 samples, N >= 2')
    ekf_mean = np.asarray(ekf_mean, dtype=float)
    if ekf_mean.shape != (6,) or not np.isfinite(ekf_mean).all():
        raise ValueError('invalid EKF mean')
    ekf_covariance, _ = factor_covariance(ekf_covariance, 6)
    p, factor = factor_covariance(ekf_covariance[:2, :2], 2)
    values, vectors = np.linalg.eigh(p)
    position = samples[:, :2]; mean = position.mean(axis=0)
    covariance = np.cov(position, rowvar=False)
    ref_values, ref_vectors = np.linalg.eigh(covariance)
    own = multiply_samples(vectors.T, (position-mean).T).T
    displaced = multiply_samples(vectors.T, (position-ekf_mean[:2]).T).T
    whitened = np.linalg.solve(factor, (position-ekf_mean[:2]).T)
    inside = np.sum(whitened**2, axis=0) <= 5.991464547107979  # 95% chi-square, d=2.
    heading = circular_summary(samples[:, 2])
    row = dict(mean_x_m=float(mean[0]), mean_y_m=float(mean[1]),
        ekf_x_m=float(ekf_mean[0]), ekf_y_m=float(ekf_mean[1]),
        mean_difference_x_m=float(mean[0]-ekf_mean[0]), mean_difference_y_m=float(mean[1]-ekf_mean[1]),
        mean_difference_norm_m=float(np.linalg.norm(mean-ekf_mean[:2])),
        ellipse_mass=float(inside.mean()), covariance=covariance.tolist(),
        reference_eigenvalues=ref_values.tolist(), reference_eigenvectors=ref_vectors.tolist(),
        ekf_covariance=p.tolist(), ekf_eigenvalues=values.tolist(), ekf_eigenvectors=vectors.tolist(),
        heading_mean_rad=heading['mean_rad'], heading_resultant=heading['resultant'],
        heading_mean_defined=heading['mean_defined'], bias_mean_rad_per_s=float(samples[:, 5].mean()),
        bias_std_rad_per_s=float(samples[:, 5].std(ddof=1)), ekf_heading_rad=float(ekf_mean[2]),
        ekf_bias_rad_per_s=float(ekf_mean[5]), ekf_bias_std_rad_per_s=float(np.sqrt(ekf_covariance[5, 5])),
        ekf_heading_resultant=float(np.exp(-ekf_covariance[2, 2]/2)))
    for i, name in enumerate(['narrow', 'wide']):
        row[f'variance_{name}_m2'] = float(np.sum(own[:, i]**2)/(len(samples)-1))
        row[f'second_moment_{name}_m2'] = float(np.mean(displaced[:, i]**2))
        row[f'ekf_variance_{name}_m2'] = float(values[i])
    return row


def compare_samples(samples, ekf_mean, ekf_covariance, batches=16):
    if type(batches) is not int or batches < 2 or len(samples) % batches or len(samples)//batches < 2:
        raise ValueError('sample count must divide into independent batches of size >= 2')
    row = distribution_metrics(samples, ekf_mean, ekf_covariance)
    batch_rows = [distribution_metrics(batch, ekf_mean, ekf_covariance) for batch in np.split(samples, batches)]
    for name in METRICS:
        row[name+'_mcse'] = float(np.std([r[name] for r in batch_rows], ddof=1)/np.sqrt(batches))
    row.update(samples=len(samples), batches=batches, weighting='independent unweighted paths',
               effective_sample_size=None, ancestry=None)
    return row


def convergence(rows, protocol):
    expected = {(s, n, r, t) for s in protocol['recording_seeds'] for n in protocol['sample_counts']
                for r in protocol['reference_seeds'] for t in protocol['report_times_us']}
    actual = [(r['recording_seed'], r['sample_count'], r['reference_seed'], r['timestamp_us']) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('missing or duplicate reference matrix entries')
    comparisons = []
    medium, high = protocol['sample_counts'][-2:]
    for seed in protocol['recording_seeds']:
        for timestamp in protocol['report_times_us']:
            groups = [[r for r in rows if r['recording_seed'] == seed and r['sample_count'] == n
                       and r['timestamp_us'] == timestamp] for n in [medium, high]]
            valid = all(r['status'] == 'complete' for group in groups for r in group)
            for name in METRICS:
                result = dict(recording_seed=seed, timestamp_us=timestamp, metric=name,
                              medium_count=medium, high_count=high, resolved=False, status='inconclusive_failure')
                if valid:
                    pooled = [float(np.mean([r[name] for r in group])) for group in groups]
                    ses = [float(np.sqrt(sum(r[name+'_mcse']**2 for r in group))/len(group)) for group in groups]
                    difference = abs(pooled[1]-pooled[0])
                    stability_limit = protocol['stability_standard_errors']*np.hypot(*ses)+1e-12
                    if name in ['mean_x_m', 'mean_y_m']:
                        index = ['mean_x_m', 'mean_y_m'].index(name)
                        precision_limit = protocol['mean_precision_fraction_of_ekf_std']*np.sqrt(groups[1][0]['ekf_covariance'][index][index])
                    elif name == 'ellipse_mass':
                        precision_limit = protocol['mass_precision_absolute']
                    else:
                        precision_limit = protocol['moment_precision_fraction']*max(abs(pooled[1]), 1e-12)
                    stable = difference <= stability_limit
                    precise = protocol['precision_standard_errors']*ses[1] <= precision_limit
                    result.update(medium_mean=pooled[0], high_mean=pooled[1], medium_mcse=ses[0], high_mcse=ses[1],
                        medium_high_difference=difference, stability_limit=float(stability_limit), stable=bool(stable),
                        precision_limit=float(precision_limit), precise=bool(precise), resolved=bool(stable and precise),
                        high_replicate_min=min(r[name] for r in groups[1]), high_replicate_max=max(r[name] for r in groups[1]),
                        high_replicate_std=float(np.std([r[name] for r in groups[1]], ddof=1)),
                        status='resolved' if stable and precise else 'inconclusive_resolution')
                comparisons.append(result)
    return comparisons
