"""Run the frozen matched-model 20-second clean GPS outage study."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.consistency import ensemble_nees, innovation_directions, state_statistics
from tools.consistency_scenarios import generate_matched
from tools.fault_scenarios import apply_faults
from tools.fault_metrics import audit_event_outputs
from tools.generate_scenario import SENSOR_HEADER, write_csv
from tools.position_geometry import position_geometry
from tools.run_consistency_study import clean_json, compress_csv, evaluate, records
from tools.run_demo import git_output, sha256


def validate_protocol(protocol):
    seeds = protocol['evaluation_seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
            or type(protocol['development_seed']) is not int
            or protocol['development_seed'] < 0 or protocol['development_seed'] in seeds):
        raise ValueError('seed roles must be unique, nonnegative and disjoint')
    # This runner implements one frozen experiment, not a general scenario API.
    if (protocol['duration_s'] != 32 or protocol['outage_us'] != [2000000, 22000000]
            or protocol['methods'] != ['ekf', 'gated'] or protocol['max_replays'] != 120
            or protocol['settling'] != dict(position_limit_m=1., dwell_us=2000000, max_gap_us=10000)):
        raise ValueError('settings do not match the documented long-outage design')
    times = protocol['fixed_times_us']
    if (not times or any(type(t) is not int or t % 10000 or not 0 <= t <= 32000000 for t in times)
            or times != sorted(set(times)) or 22000000 not in times):
        raise ValueError('invalid fixed timestamps')


def scenarios(protocol):
    return [dict(id='nominal', outage_us=None, offsets=[], recovery_windows=[]),
            dict(id='dropout', outage_us=protocol['outage_us'], offsets=[],
                 recovery_windows=[dict(id='clean_return', reference_us=22000000,
                                        observation_end_us=32000000)])]


def make_pair(folder, seed, filter_path, protocol):
    generate_matched(folder, seed, filter_path, duration_s=protocol['duration_s'])
    with (folder/'nominal.csv').open(newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != SENSOR_HEADER:
            raise ValueError('unexpected sensor schema')
        nominal = list(reader)
    annotations = {}
    for scenario in scenarios(protocol):
        rows, annotation = apply_faults(nominal, scenario)
        if scenario['id'] == 'dropout':
            write_csv(folder/'dropout.csv', SENSOR_HEADER, rows)
        annotations[scenario['id']] = annotation
    (folder/'annotations.json').write_text(json.dumps(annotations, indent=2)+'\n')
    return annotations


def read_pre_gps(path, timestamp):
    """Read the actual prediction at the return event, before GPS correction."""
    with path.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['timestamp_us']) == timestamp]
    if len(rows) != 1:
        raise ValueError('expected exactly one GPS snapshot at the return time')
    row = rows[0]
    state = np.array([float(row[f'pre_x{i}']) for i in range(6)])
    covariance = np.array([[float(row[f'pre_p{i}{j}']) for j in range(6)] for i in range(6)])
    return state, covariance


def snapshot(key, timestamp, phase, state, covariance, truth):
    row = dict(**key, timestamp_us=timestamp, phase=phase)
    row.update({name: float(values[0]) for name, values in
                state_statistics(state[None], truth[None], covariance[None]).items()})
    error = state[:2] - truth[:2]
    row.update(error_x=float(error[0]), error_y=float(error[1]))
    row.update({f'p{i}{j}': float(covariance[i, j]) for i in range(2) for j in range(2)})
    try:
        directions = innovation_directions(error, covariance[:2, :2])
        for i, name in enumerate(['narrow', 'wide']):
            row[name+'_variance'] = float(directions['eigenvalues'][i])
            row[name+'_nees_contribution'] = float(directions['contributions'][i])
    except ValueError:
        pass  # Invalid statistics remain null; the planned sample still exists.
    return row


def summarize(runs, points, seeds, protocol):
    expected = {(s, c, m) for s in seeds for c in ['nominal', 'dropout'] for m in protocol['methods']}
    run_keys = [(r['seed'], r['scenario'], r['method']) for r in runs]
    phases = [(t, 'post_events') for t in protocol['fixed_times_us']] + [(22000000, 'pre_gps')]
    expected_points = {(*key, t, phase) for key in expected for t, phase in phases}
    point_keys = [(r['seed'], r['scenario'], r['method'], r['timestamp_us'], r['phase']) for r in points]
    if len(run_keys) != len(expected) or set(run_keys) != expected:
        raise ValueError('incomplete or duplicate run matrix')
    if len(point_keys) != len(expected_points) or set(point_keys) != expected_points:
        raise ValueError('incomplete or duplicate fixed-time matrix')
    ensemble, geometry, summary = [], [], []
    for case in ['nominal', 'dropout']:
        for method in protocol['methods']:
            for timestamp, phase in phases:
                group = [p for p in points if (p['scenario'], p['method'], p['timestamp_us'], p['phase'])
                         == (case, method, timestamp, phase)]
                key = dict(scenario=case, method=method, timestamp_us=timestamp, phase=phase)
                for name, dimension in [('position', 2), ('full', 6)]:
                    stats = ensemble_nees([p.get(name+'_nees', np.nan) for p in group], dimension)
                    coverage = np.array([p.get('position_covered_95', np.nan) for p in group])
                    ensemble.append(dict(**key, statistic=name, **stats,
                        coverage_valid_runs=int(np.isfinite(coverage).sum()),
                        position_covered_runs=int(np.sum(coverage == 1))))
                errors = [[p.get('error_x', np.nan), p.get('error_y', np.nan)] for p in group]
                covariances = [[[p.get(f'p{i}{j}', np.nan) for j in range(2)] for i in range(2)] for p in group]
                geometry.append(dict(**key, **position_geometry(errors, covariances)))
            group = [r for r in runs if r['scenario'] == case and r['method'] == method]
            good = [r for r in group if r['status'] == 'complete']
            row = dict(scenario=case, method=method, planned_runs=len(group),
                       completed_runs=len(good), failed_runs=len(group)-len(good))
            for field in ['position_rmse_m', 'outage_position_rmse_m', 'outage_end_position_error_m',
                          'final_position_error_m', 'heading_rmse_rad']:
                values = [r[field] for r in good if r.get(field) is not None]
                row[field+'_median'] = float(np.median(values)) if values else None
            row.update(clean_gps_considered=sum(r['clean_gps_considered'] for r in good),
                       clean_gps_rejected=sum(r['clean_gps_rejected'] for r in good),
                       clean_gps_accepted=sum(r['clean_gps_considered']-r['clean_gps_rejected'] for r in good),
                       return_gps_accepted=sum(r['return_gps_status'] == 'accepted' for r in good),
                       settled=sum(r['settling_start_delay_s'] is not None for r in good),
                       settled_but_final_over_1m=sum(r['settled_but_final_over_1m'] for r in good))
            summary.append(row)
    return ensemble, geometry, summary


def run_study(binary, output, protocol_path=ROOT/'scenarios/long_outage.json', smoke=False):
    binary, output, protocol_path = Path(binary).resolve(), Path(output).resolve(), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    filter_path = ROOT/'scenarios/filter.conf'
    if sha256(filter_path) != protocol['frozen_filter_sha256']:
        raise ValueError('filter differs from frozen protocol')
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('build navtrace_replay first')
    seeds = [protocol['development_seed']] if smoke else protocol['evaluation_seeds']
    expected = len(seeds)*4
    if expected > protocol['max_replays']:
        raise ValueError('study exceeds predefined run limit')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(filter_path, output/'filter.conf')
    shutil.copyfile(protocol_path, output/'protocol.json')
    sources = [p for folder in ['include', 'src', 'apps', 'tools', 'tests', 'docs', 'scenarios']
               for p in (ROOT/folder).rglob('*') if p.suffix in ['.cpp', '.hpp', '.py', '.md', '.json', '.conf']]
    sources += [ROOT/n for n in ['AGENTS.md', 'README.md', 'CMakeLists.txt', 'requirements-lock.txt']]
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), seeds=seeds,
        seed_role='development' if smoke else 'held_out_long_outage', expected_replays=expected,
        git_commit=git_output('rev-parse', 'HEAD'), git_branch=git_output('branch', '--show-current'),
        git_dirty=bool(git_output('status', '--porcelain')), binary=str(binary), binary_sha256=sha256(binary),
        filter_sha256=sha256(filter_path), protocol_sha256=sha256(protocol_path), python=sys.version,
        platform=platform.platform(), build_info=(binary.parent/'build_info.txt').read_text(),
        dependencies={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'matplotlib', 'pytest', 'cmake']},
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
        commands=[], failures=[], uncompressed_output_sha256={})
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False)+'\n')
    save_manifest()
    begin = time.monotonic()
    runs, points, return_events = [], [], []
    try:
        with (output/'replay.log').open('x') as log:
            for seed in seeds:
                folder = output/f'seed-{seed}'
                annotations = make_pair(folder/'inputs', seed, filter_path, protocol)
                traces = {}
                for scenario in scenarios(protocol):
                    case = scenario['id']
                    for method in protocol['methods']:
                        key = dict(seed=seed, scenario=case, method=method)
                        trajectory, gps = folder/f'{case}-{method}.csv', folder/f'{case}-{method}-gps.csv'
                        command = [str(binary), str(folder/f'inputs/{case}.csv'), str(output/'filter.conf'),
                                   method, str(trajectory), str(gps)]
                        manifest['commands'].append(command)
                        try:
                            proc = subprocess.run(command, capture_output=True, text=True)
                            log.write(' '.join(command)+'\n'+proc.stdout+proc.stderr); log.flush()
                            proc.check_returncode()
                            audit_event_outputs(Path(command[1]), trajectory, method)
                            result, _, series = evaluate(trajectory, gps, folder/'inputs/truth.csv', annotations[case],
                                scenario, seed, method, 'matched_long', [])
                            times, states, covariances, truth, stats, events = series
                            samples = []
                            for timestamp in protocol['fixed_times_us']:
                                index = int(np.flatnonzero(times == timestamp)[0])
                                samples.append(snapshot(key, timestamp, 'post_events', states[index], covariances[index], truth[index]))
                            state, covariance = read_pre_gps(gps, 22000000)
                            return_index = int(np.flatnonzero(times == 22000000)[0])
                            samples.append(snapshot(key, 22000000, 'pre_gps', state, covariance, truth[return_index]))
                            event = next(r for r in events if r['timestamp_us'] == 22000000)
                            result.update(return_gps_status=event['status'], return_nis=event['nis'],
                                settled_but_final_over_1m=result['settling_start_delay_s'] is not None and result['final_position_error_m'] > 1)
                            runs.append(result); points.extend(samples); return_events.append(dict(**key, **event))
                            if seed == min(seeds):
                                traces[(case, method)] = series
                        except Exception as error:
                            failure = dict(**key, error=str(error))
                            manifest['failures'].append(failure); runs.append(dict(**failure, status='failed'))
                            points.extend(dict(**key, timestamp_us=t, phase='post_events') for t in protocol['fixed_times_us'])
                            points.append(dict(**key, timestamp_us=22000000, phase='pre_gps'))
                            return_events.append(dict(**key, status='failed', error=str(error)))
                        finally:
                            for path in [trajectory, gps]:
                                if path.exists():
                                    manifest['uncompressed_output_sha256'][str(path.relative_to(output))] = compress_csv(path)
                            save_manifest()
                if traces:
                    from tools.plot_long_outage import plot_traces
                    plot_traces(output, seed, traces)
                print(f'Finished seed {seed}', flush=True)
        ensemble, geometry, summary = summarize(runs, points, seeds, protocol)
        for name, rows in [('metrics', runs), ('fixed-time-statistics', points), ('return-events', return_events),
                           ('ensemble', ensemble), ('geometry', geometry), ('summary', summary)]:
            records(output/name, rows)
        from tools.plot_long_outage import plot_ensemble
        plot_ensemble(output, ensemble, geometry, points)
        manifest.update(status='complete' if not manifest['failures'] else 'complete_with_failures',
                        elapsed_s=time.monotonic()-begin)
        manifest['artifacts_sha256'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                                       if p.is_file() and p.name != 'manifest.json'}
        manifest['retained_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        print(json.dumps(summary, indent=2))
        print(f"{expected} replays; {len(manifest['failures'])} failures; {manifest['elapsed_s']:.1f} s; "
              f"{manifest['retained_bytes']/1e6:.1f} MB; {output}")
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest.update(attempted_replays=len(manifest['commands']),
            completed_replays=sum(r['status'] == 'complete' for r in runs),
            failed_replays=sum(r['status'] == 'failed' for r in runs),
            not_started_replays=expected-len(manifest['commands']))
        if not (output/'metrics.json').exists():
            records(output/'metrics', runs); records(output/'fixed-time-statistics', points)
            records(output/'return-events', return_events)
        save_manifest()
    return runs, ensemble


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-smoke', action='store_true', help='four development replays; no evaluation seeds')
    args = parser.parse_args()
    results, _ = run_study(args.binary, args.output, smoke=args.development_smoke)
    sys.exit(1 if any(r['status'] != 'complete' for r in results) else 0)
