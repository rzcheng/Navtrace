"""Position-only saved-sample metrics; no sampling or estimator inputs here."""
import numpy as np

METRICS = (
    'mean_x_m', 'mean_y_m', 'variance_narrow_m2', 'variance_wide_m2',
    'covariance_mixed_m2', 'second_moment_narrow_m2', 'second_moment_wide_m2',
    'baseline_ellipse_mass', 'candidate_ellipse_mass',
)
MOMENTS = METRICS[:7]
ELLIPSE_THRESHOLD = 5.991464547107979


def principal_axes(covariance):
    """Ascending axes with a deterministic, physically immaterial column sign."""
    p = np.asarray(covariance, dtype=float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError('expected finite 2 x 2 position covariance')
    if not np.allclose(p, p.T, rtol=0, atol=1e-10*max(1., np.max(np.abs(p)))):
        raise ValueError('position covariance is asymmetric')
    values, axes = np.linalg.eigh(p)
    if values[0] < -1e-10*max(1., np.max(np.abs(p))):
        raise ValueError('position covariance is not positive semidefinite')
    for column in range(2):
        pivot = np.argmax(np.abs(axes[:, column]))
        if axes[pivot, column] < 0:
            axes[:, column] *= -1
    return values, axes


def ellipse_mass(position, mean, covariance):
    # A Cholesky solve both requires strict positivity and avoids an inverse.
    if mean is None or covariance is None:
        raise ValueError('candidate snapshot unavailable')
    p = np.asarray(covariance, dtype=float)[:2, :2]
    principal_axes(p)
    factor = np.linalg.cholesky(p)
    whitened = np.linalg.solve(factor, (position-np.asarray(mean)[:2]).T)
    return float(np.mean(np.sum(whitened*whitened, axis=0) <= ELLIPSE_THRESHOLD))


def deterministic_moments(mean, covariance, baseline_mean, axes):
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    position_covariance = covariance[:2, :2]
    values, directions = principal_axes(position_covariance)
    projected = axes.T @ position_covariance @ axes
    displacement = axes.T @ (mean[:2]-np.asarray(baseline_mean)[:2])
    return dict(mean_x_m=float(mean[0]), mean_y_m=float(mean[1]),
                variance_narrow_m2=float(projected[0, 0]),
                variance_wide_m2=float(projected[1, 1]),
                covariance_mixed_m2=float(projected[0, 1]),
                second_moment_narrow_m2=float(projected[0, 0]+displacement[0]**2),
                second_moment_wide_m2=float(projected[1, 1]+displacement[1]**2),
                position_mean=mean[:2].tolist(), position_covariance=position_covariance.tolist(),
                eigenvalues=values.tolist(), eigenvectors=directions.tolist())


def sample_values(position, baseline_mean, baseline_covariance, candidate_mean,
                  candidate_covariance, axes):
    """Return each metric separately so a singular ellipse does not erase moments."""
    mean = position.mean(axis=0)
    # Direct contraction avoids the local large, thin BLAS warning path.
    centered = np.einsum('ni,ij->nj', position-mean, axes, optimize=False)
    displaced = np.einsum('ni,ij->nj', position-np.asarray(baseline_mean)[:2], axes, optimize=False)
    covariance = centered.T @ centered/(len(position)-1)
    values = dict(mean_x_m=float(mean[0]), mean_y_m=float(mean[1]),
                  variance_narrow_m2=float(covariance[0, 0]),
                  variance_wide_m2=float(covariance[1, 1]),
                  covariance_mixed_m2=float(covariance[0, 1]),
                  second_moment_narrow_m2=float(np.mean(displaced[:, 0]**2)),
                  second_moment_wide_m2=float(np.mean(displaced[:, 1]**2)))
    result = {name: dict(status='complete', value=value) for name, value in values.items()}
    for name, m, p in [('baseline_ellipse_mass', baseline_mean, baseline_covariance),
                       ('candidate_ellipse_mass', candidate_mean, candidate_covariance)]:
        try:
            result[name] = dict(status='complete', value=ellipse_mass(position, m, p))
        except (ValueError, np.linalg.LinAlgError) as error:
            result[name] = dict(status='failed', error=str(error))
    return result


def compare_samples(samples, baseline_mean, baseline_covariance, candidate_mean,
                    candidate_covariance, batches=16, check_budget=lambda: None):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 6 or not np.isfinite(samples).all():
        raise ValueError('expected finite N x 6 saved samples')
    if type(batches) is not int or batches < 2 or len(samples) % batches or len(samples)//batches < 2:
        raise ValueError('sample count must divide into batches of at least two')
    _, axes = principal_axes(np.asarray(baseline_covariance)[:2, :2])
    position = samples[:, :2]  # Wrapped saved yaw is deliberately never used.
    candidate_error = None
    if candidate_mean is not None:
        try:
            cm, cp = np.asarray(candidate_mean), np.asarray(candidate_covariance)
            if cm.shape != (6,) or cp.shape != (6, 6) or not np.isfinite(cm).all() or not np.isfinite(cp).all():
                raise ValueError('invalid candidate snapshot')
            principal_axes(cp[:2, :2])
        except ValueError as error:
            candidate_mean = candidate_covariance = None
            candidate_error = str(error)
    metrics = sample_values(position, baseline_mean, baseline_covariance, candidate_mean,
                            candidate_covariance, axes)
    batch_values = []
    for batch in np.split(position, batches):
        check_budget()
        batch_values.append(sample_values(batch, baseline_mean, baseline_covariance,
                                         candidate_mean, candidate_covariance, axes))
    for name, metric in metrics.items():
        if metric['status'] == 'complete':
            metric['mcse'] = float(np.std([batch[name]['value'] for batch in batch_values], ddof=1)
                                   / np.sqrt(batches))
            metric['batch_values'] = [batch[name]['value'] for batch in batch_values]
    position_mean = position.mean(axis=0)
    position_covariance = np.cov(position, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = principal_axes(position_covariance)
    baseline = deterministic_moments(baseline_mean, baseline_covariance, baseline_mean, axes)
    candidate = (deterministic_moments(candidate_mean, candidate_covariance, baseline_mean, axes)
                 if candidate_mean is not None else None)
    description = dict(position_mean=position_mean.tolist(), position_covariance=position_covariance.tolist(),
                       eigenvalues=eigenvalues.tolist(), eigenvectors=eigenvectors.tolist(),
                       baseline=baseline, candidate=candidate, common_axes=axes.tolist())
    for name, method in [('baseline', baseline), ('candidate', candidate)]:
        if method is not None:
            difference = np.asarray(method['position_mean'])-position_mean
            cov_difference = np.asarray(method['position_covariance'])-position_covariance
            description[name+'_minus_reference'] = dict(mean_difference=difference.tolist(),
                mean_difference_norm_m=float(np.linalg.norm(difference)),
                covariance_difference=cov_difference.tolist(),
                covariance_difference_frobenius_m2=float(np.linalg.norm(cov_difference)))
    if candidate_error is not None:
        metrics['candidate_ellipse_mass']['error'] = candidate_error
    return dict(status='complete' if all(m['status'] == 'complete' for m in metrics.values()) else 'failed',
                metrics=metrics, description=description, batches=batches)


def planned_rows(protocol):
    return [dict(recording_seed=s, timestamp_us=t, sample_count=n, reference_seed=r,
                 status='not_started', metrics={name: dict(status='not_started') for name in METRICS})
            for s in protocol['recording_seeds'] for t in protocol['report_times_us']
            for n in protocol['sample_counts'] for r in protocol['reference_seeds']]


def pool_metric(group, metric):
    entries = [row['metrics'][metric] for row in group]
    if any(entry['status'] != 'complete' for entry in entries):
        return dict(status='failed' if any(e['status'] == 'failed' for e in entries) else 'not_started')
    points = [entry['value'] for entry in entries]
    return dict(status='complete', value=float(np.mean(points)),
                mcse=float(np.sqrt(sum(entry['mcse']**2 for entry in entries))/len(entries)),
                replicate_min=min(points), replicate_max=max(points),
                replicate_std=float(np.std(points, ddof=1)))


def summarize(rows, protocol):
    """Pool all counts and return all resolution outcomes, including missing work."""
    key = lambda row: tuple(row[k] for k in ['recording_seed', 'timestamp_us', 'sample_count', 'reference_seed'])
    if len(rows) != len(planned_rows(protocol)) or set(map(key, rows)) != set(map(key, planned_rows(protocol))):
        raise ValueError('missing or duplicate matrix entries')
    pooled, outcomes = [], []
    for recording in protocol['recording_seeds']:
        for timestamp in protocol['report_times_us']:
            groups = {n: [r for r in rows if (r['recording_seed'], r['timestamp_us'], r['sample_count'])
                          == (recording, timestamp, n)] for n in protocol['sample_counts']}
            by_count = {}
            for count, group in groups.items():
                metrics = {name: pool_metric(group, name) for name in METRICS}
                descriptions = [r.get('description') for r in group]
                description = None
                if all(d is not None for d in descriptions):
                    # Average own-mean covariances, without between-replicate scatter.
                    covariance = np.mean([d['position_covariance'] for d in descriptions], axis=0)
                    mean = np.mean([d['position_mean'] for d in descriptions], axis=0)
                    eigenvalues, eigenvectors = principal_axes(covariance)
                    description = dict(position_mean=mean.tolist(), position_covariance=covariance.tolist(),
                        eigenvalues=eigenvalues.tolist(), eigenvectors=eigenvectors.tolist(),
                        baseline=descriptions[0]['baseline'], candidate=descriptions[0]['candidate'])
                    for method in ['baseline', 'candidate']:
                        if description[method] is not None:
                            dm = np.asarray(description[method]['position_mean'])-mean
                            dp = np.asarray(description[method]['position_covariance'])-covariance
                            description[method+'_minus_reference'] = dict(mean_difference=dm.tolist(),
                                mean_difference_norm_m=float(np.linalg.norm(dm)), covariance_difference=dp.tolist(),
                                covariance_difference_frobenius_m2=float(np.linalg.norm(dp)))
                summary = dict(recording_seed=recording, timestamp_us=timestamp, sample_count=count,
                               metrics=metrics, description=description)
                by_count[count] = summary
                pooled.append(summary)
            medium, high = [by_count[n] for n in protocol['sample_counts'][-2:]]
            for name in METRICS:
                a, b = medium['metrics'][name], high['metrics'][name]
                outcome = dict(recording_seed=recording, timestamp_us=timestamp, metric=name,
                               status='not_started', resolved=False, reason='required metric unavailable')
                if a['status'] != 'complete' or b['status'] != 'complete':
                    outcome['status'] = 'failed' if 'failed' in [a['status'], b['status']] else 'not_started'
                    outcomes.append(outcome)
                    continue
                if name in ['mean_x_m', 'mean_y_m']:
                    index = ['mean_x_m', 'mean_y_m'].index(name)
                    description = high['description']
                    if description is None:
                        outcome.update(status='failed', reason='baseline covariance unavailable')
                        outcomes.append(outcome)
                        continue
                    limit = protocol['mean_precision_fraction_of_ekf_std']*np.sqrt(
                        description['baseline']['position_covariance'][index][index])
                elif name in METRICS[-2:]:
                    limit = protocol['mass_precision_absolute']
                elif name == 'covariance_mixed_m2':
                    narrow, wide = [high['metrics'][n] for n in ['variance_narrow_m2', 'variance_wide_m2']]
                    if narrow['status'] != 'complete' or wide['status'] != 'complete':
                        outcome.update(status='failed', reason='required reference variances unavailable')
                        outcomes.append(outcome)
                        continue
                    limit = protocol['mixed_covariance_precision_fraction_of_geometric_reference_std']*max(
                        np.sqrt(narrow['value']*wide['value']), protocol['moment_floor_m2'])
                else:
                    limit = protocol['moment_precision_fraction']*max(abs(b['value']), protocol['moment_floor_m2'])
                stability_limit = protocol['stability_standard_errors']*np.hypot(a['mcse'], b['mcse']) + protocol['roundoff_allowance_in_metric_units']
                stable = abs(b['value']-a['value']) <= stability_limit
                precise = protocol['precision_standard_errors']*b['mcse'] <= limit
                band = protocol['agreement_standard_errors']*b['mcse'] + protocol['roundoff_allowance_in_metric_units']
                outcome.update(status='complete', reason='resolved' if stable and precise else 'inconclusive_resolution',
                    resolved=bool(stable and precise), stable=bool(stable), precise=bool(precise),
                    medium=a, high=b, medium_high_difference=abs(b['value']-a['value']),
                    stability_limit=float(stability_limit), precision_limit=float(limit), agreement_band=float(band))
                if name in MOMENTS and high['description'] is not None:
                    for method in ['baseline', 'candidate']:
                        deterministic = high['description'][method]
                        if deterministic is not None:
                            error = deterministic[name]-b['value']
                            outcome[method] = dict(value=deterministic[name], signed_error=error,
                                                  within_agreement_band=bool(abs(error) <= band))
                    if 'baseline' in outcome and 'candidate' in outcome:
                        outcome['candidate_minus_baseline'] = outcome['candidate']['value']-outcome['baseline']['value']
                        outcome['absolute_error_change'] = abs(outcome['candidate']['signed_error'])-abs(outcome['baseline']['signed_error'])
                elif name in METRICS[-2:]:
                    outcome.update(difference_from_nominal=b['value']-.95,
                                   nominal_within_agreement_band=bool(abs(b['value']-.95) <= band))
                outcomes.append(outcome)
    return pooled, outcomes
