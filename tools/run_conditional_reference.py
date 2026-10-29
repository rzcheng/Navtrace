"""Bounded offline conditional reference using two frozen C++ recordings."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.conditional_reference import Observation, build_history, factor_covariance, sample_endpoint
from tools.conditional_metrics import compare_samples, convergence
from tools.run_consistency_study import clean_json, records
from tools.run_demo import git_output, sha256

STATE_NAMES = ['px_m', 'py_m', 'yaw_rad', 'speed_m_per_s', 'yaw_rate_rad_per_s', 'gyro_bias_rad_per_s']


def json_write(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


def peak_memory_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**2 if sys.platform == 'darwin' else 1024)


def read_snapshots(trajectory, times, end_us):
    snapshots = {}
    with gzip.open(trajectory, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            t = int(row['timestamp_us'])
            if t > end_us or (t == end_us and row['sensor'] == 'gps'):
                break  # Never take the returning GPS correction as the endpoint.
            if t in times:
                mean = np.array([float(row[name]) for name in STATE_NAMES])
                covariance = np.array([[float(row[f'p{i}{j}']) for j in range(6)] for i in range(6)])
                covariance, _ = factor_covariance(covariance, 6)
                if not np.isfinite(mean).all():
                    raise ValueError('invalid baseline snapshot mean')
                snapshots[t] = (mean, covariance)
    if set(snapshots) != set(times):
        raise ValueError('missing required baseline snapshots')
    return snapshots


def read_observations(path, checkpoint_us, end_us):
    events = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            t = int(row['timestamp_us'])
            if t > end_us or (t == end_us and row['sensor'] == 'gps'):
                break
            if t <= checkpoint_us:
                continue
            if row['sensor'] not in ['gyro', 'wheel']:
                raise ValueError('unexpected GPS inside the outage')
            events.append(Observation(t, row['sensor'], float(row['value1']), float(row['cov11'])))
    return events


def verify_baseline(study, protocol):
    baseline = json.loads((study/'manifest.json').read_text())
    if (baseline['git_commit'] != protocol['baseline_commit'] or baseline['git_dirty']
            or baseline['status'] != 'complete' or baseline['completed_replays'] != 120):
        raise ValueError('unexpected or incomplete baseline study')
    for name, digest in baseline['artifacts_sha256'].items():
        if sha256(study/name) != digest:
            raise ValueError(f'baseline artifact hash differs: {name}')
    for name, digest in baseline['source_sha256'].items():
        if name.startswith(('src/', 'include/', 'apps/')) and sha256(ROOT/name) != digest:
            raise ValueError(f'production source differs from baseline: {name}')
    if sha256(ROOT/'scenarios/filter.conf') != protocol['filter_sha256']:
        raise ValueError('filter differs from frozen baseline')
    metrics = json.loads((study/'metrics.json').read_text())
    lockout, non_lockout = [], []
    for row in metrics:
        if row['scenario'] != 'dropout' or row['method'] != 'gated':
            continue
        seed = row['seed']
        with gzip.open(study/f'seed-{seed}/dropout-gated-gps.csv.gz', 'rt') as stream:
            events = [r for r in csv.DictReader(stream) if 22000000 <= int(r['timestamp_us']) <= 32000000]
        if len(events) == 51 and all(r['status'] == 'rejected' for r in events):
            lockout.append(seed)
        if events[0]['status'] == 'accepted' and row['settling_start_delay_s'] is not None:
            non_lockout.append(seed)
    if [min(non_lockout), min(lockout)] != protocol['recording_seeds']:
        raise ValueError('case selection differs from the frozen rule')
    return baseline


def load_case(study, recording_seed, protocol):
    folder = study/f'seed-{recording_seed}'
    end = protocol['report_times_us'][-1]
    snapshots = read_snapshots(folder/'dropout-gated.csv.gz', protocol['report_times_us'], end)
    with gzip.open(folder/'dropout-gated-gps.csv.gz', 'rt') as stream:
        row = next(r for r in csv.DictReader(stream) if int(r['timestamp_us']) == end)
    logged_mean = np.array([float(row[f'pre_x{i}']) for i in range(6)])
    logged_covariance = np.array([[float(row[f'pre_p{i}{j}']) for j in range(6)] for i in range(6)])
    if not np.array_equal(snapshots[end][0], logged_mean) or not np.array_equal(snapshots[end][1], logged_covariance):
        raise ValueError('endpoint does not match logged pre-GPS snapshot')
    observations = read_observations(folder/'inputs/dropout.csv', protocol['checkpoint_us'], end)
    config = dict(line.split('=', 1) for line in (study/'filter.conf').read_text().splitlines() if line and not line.startswith('#'))
    diffusion = [float(config[k]) for k in ['speed_diffusion_m2_per_s3', 'yaw_rate_diffusion_rad2_per_s3', 'bias_diffusion_rad2_per_s3']]
    mean, covariance = snapshots[protocol['checkpoint_us']]
    history = build_history(mean, covariance, protocol['checkpoint_us'], observations, end, diffusion,
                            float(config['max_step_s']), float(config['max_gap_s']))
    agreement = []
    for timestamp, (state, p) in snapshots.items():
        index = history.indices[timestamp]
        difference = history.means[index]-state[2:]
        difference[0] = (difference[0]+np.pi) % (2*np.pi)-np.pi
        covariance_difference = np.max(np.abs(history.covariances[index]-p[2:, 2:]))
        if np.max(np.abs(difference)) > 1e-9 or covariance_difference > 1e-9:
            raise ValueError('linear latent moments disagree with the frozen C++ baseline')
        agreement.append(dict(timestamp_us=timestamp, max_mean_difference=float(np.max(np.abs(difference))),
                              max_covariance_difference=float(covariance_difference),
                              observations=history.observation_counts[timestamp], log_likelihood=history.log_likelihoods[timestamp]))
    return history, snapshots, observations, agreement


def run_study(study, output, pilot=False, pilot_results=None):
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol_path = ROOT/'scenarios/conditional_reference.json'
    protocol = json.loads(protocol_path.read_text())
    if (protocol['sample_counts'] != [512, 2048, 8192] or len(protocol['reference_seeds']) != 3
            or protocol['recording_seeds'] != [83001, 83021] or protocol['checkpoint_us'] != 1990000
            or protocol['report_times_us'] != [1990000, 5000000, 10000000, 15000000, 22000000]):
        raise ValueError('runner settings differ from the frozen bounded design')
    if output.exists():
        raise FileExistsError(output)
    pilot_manifest = None
    if not pilot:
        if pilot_results is None:
            raise ValueError('run and supply --pilot-results before the full diagnostic')
        pilot_manifest = json.loads((Path(pilot_results)/'manifest.json').read_text())
        if (pilot_manifest['status'] != 'complete' or pilot_manifest['mode'] != 'pilot'
                or pilot_manifest['protocol_sha256'] != sha256(protocol_path)
                or not pilot_manifest['full_run_within_budget_estimate']):
            raise ValueError('pilot does not authorize this bounded run')
    baseline = verify_baseline(study, protocol)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output/'protocol.json')
    sources = [p for folder in ['src', 'include', 'apps', 'tools', 'tests', 'docs', 'scenarios']
               for p in (ROOT/folder).rglob('*') if p.suffix in ['.cpp', '.hpp', '.py', '.md', '.json', '.conf']]
    recordings = protocol['recording_seeds'][:1] if pilot else protocol['recording_seeds']
    counts = protocol['sample_counts'][:1] if pilot else protocol['sample_counts']
    seeds = [protocol['pilot_seed']] if pilot else protocol['reference_seeds']
    expected = [(s, n, r, t) for s in recordings for n in counts for r in seeds for t in protocol['report_times_us']]
    manifest = dict(status='running', mode='pilot' if pilot else 'full', created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_output('rev-parse', 'HEAD'), git_dirty=bool(git_output('status', '--porcelain')),
        git_branch=git_output('branch', '--show-current'), command=sys.argv, protocol_sha256=sha256(protocol_path),
        baseline_manifest_sha256=sha256(study/'manifest.json'), baseline_commit=baseline['git_commit'],
        baseline_artifacts_verified=len(baseline['artifacts_sha256']), baseline_path=str(study),
        python=sys.version, platform=platform.platform(), production_build=baseline['build_info'],
        dependencies={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'matplotlib', 'pytest']},
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
        expected_distributions=len(expected), expected_runs=len(recordings)*len(counts)*len(seeds),
        sample_counts=counts, recording_seeds=recordings, reference_seeds=seeds, failures=[])
    if pilot_manifest:
        manifest['pilot_manifest_sha256'] = sha256(Path(pilot_results)/'manifest.json')
        manifest['pilot_full_wall_estimate_s'] = pilot_manifest['estimated_full_wall_s']
    start = time.monotonic()
    def check_budget():
        if time.monotonic()-start > protocol['full_run_wall_budget_s']:
            raise TimeoutError('frozen wall-time budget exceeded')
        if peak_memory_mib() > protocol['memory_budget_mib']:
            raise MemoryError('frozen peak memory budget exceeded')
    rows = []
    json_write(output/'manifest.json', manifest)
    try:
        for recording in recordings:
            history, snapshots, observations, agreement = load_case(study, recording, protocol)
            case = output/f'case-{recording}'; case.mkdir()
            json_write(case/'checkpoint.json', dict(timestamp_us=protocol['checkpoint_us'], phase='post_events',
                mean=history.initial_mean.tolist(), covariance=history.initial_covariance.tolist()))
            json_write(case/'ekf-snapshots.json', {str(t): dict(mean=m.tolist(), covariance=c.tolist()) for t, (m, c) in snapshots.items()})
            records(case/'latent-agreement', agreement)
            with (case/'observations.csv').open('x', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['timestamp_us', 'sensor', 'value', 'variance'])
                writer.writerows((o.timestamp_us, o.sensor, o.value, o.variance) for o in observations)
            for count in counts:
                for seed in seeds:
                    for timestamp in protocol['report_times_us']:
                        check_budget()
                        key = dict(recording_seed=recording, sample_count=count, reference_seed=seed, timestamp_us=timestamp)
                        begin = time.monotonic()
                        try:
                            samples = sample_endpoint(history, timestamp, count, seed, recording, check_budget)
                            result = compare_samples(samples, *snapshots[timestamp], protocol['batches'])
                            path = case/f'n{count}-r{seed}-t{timestamp}.npz'
                            with path.open('xb') as stream:
                                np.savez_compressed(stream, states=samples)
                            rows.append(dict(**key, status='complete', **result, elapsed_s=time.monotonic()-begin,
                                             observations=history.observation_counts[timestamp]))
                        except (TimeoutError, MemoryError) as error:
                            rows.append(dict(**key, status='failed', error=str(error)))
                            manifest['failures'].append(dict(**key, error=str(error)))
                            raise
                        except Exception as error:
                            rows.append(dict(**key, status='failed', error=str(error)))
                            manifest['failures'].append(dict(**key, error=str(error)))
                    print(f'Finished recording {recording}, N={count}, reference seed {seed}', flush=True)
        comparisons = [] if pilot else convergence(rows, protocol)
        records(output/'comparisons', rows)
        if comparisons:
            records(output/'convergence', comparisons)
        sampling_wall = time.monotonic()-start
        from tools.plot_conditional_reference import plot_results
        plot_results(output, rows, recordings, counts, seeds)
        check_budget()
        manifest.update(status='complete' if not manifest['failures'] else 'complete_with_failures', sampling_and_metrics_wall_s=sampling_wall)
        if pilot:
            # Scale the entire pilot computation conservatively, then allow 20 s for plotting.
            estimate = sampling_wall*(2*3*sum(protocol['sample_counts'])/counts[0])+20
            manifest.update(estimated_full_wall_s=estimate,
                            full_run_within_budget_estimate=estimate < protocol['full_run_wall_budget_s']
                            and peak_memory_mib() < protocol['memory_budget_mib'])
        else:
            manifest.update(resolved_comparisons=sum(r['resolved'] for r in comparisons),
                            inconclusive_comparisons=sum(not r['resolved'] for r in comparisons))
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        observed = {(r['recording_seed'], r['sample_count'], r['reference_seed'], r['timestamp_us']) for r in rows}
        rows.extend(dict(recording_seed=s, sample_count=n, reference_seed=r, timestamp_us=t, status='not_started')
                    for s, n, r, t in expected if (s, n, r, t) not in observed)
        if not (output/'comparisons.json').exists():
            records(output/'comparisons', rows)
        manifest.update(elapsed_s=time.monotonic()-start, peak_memory_mib=peak_memory_mib(),
                        completed_distributions=sum(r['status'] == 'complete' for r in rows),
                        failed_distributions=sum(r['status'] == 'failed' for r in rows),
                        not_started_distributions=sum(r['status'] == 'not_started' for r in rows))
        manifest['artifacts_sha256'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                                       if p.is_file() and p.name != 'manifest.json'}
        json_write(output/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['status', 'completed_distributions', 'elapsed_s', 'peak_memory_mib']}, indent=2))
    if pilot:
        print(f"Conservative full-study estimate: {manifest['estimated_full_wall_s']:.1f} s")
    return rows, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'results/long-outage-study')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pilot-results', type=Path)
    args = parser.parse_args()
    _, manifest = run_study(args.study, args.output, args.pilot, args.pilot_results)
    sys.exit(0 if manifest['status'] == 'complete' else 1)
