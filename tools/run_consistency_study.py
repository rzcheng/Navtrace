"""Execute the frozen M3 study; truth is read only after each C++ replay."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
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
from tools.consistency import state_statistics, ensemble_nees, elapsed_settling
from tools.consistency_scenarios import generate_stress, generate_matched
from tools.event_diagnostics import audit_gps_events
from tools.evaluate import compute_metrics
from tools.fault_metrics import gps_decisions, audit_event_outputs
from tools.observability import run_observability
from tools.run_demo import sha256, git_output
from tools.run_fault_study import save_records


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def records(path, rows):
    # Failed runs remain rows, with unavailable quantities blank/null.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    save_records(path, [clean_json({key: row.get(key) for key in keys}) for row in rows])


def load_states(estimate_path, truth_path):
    data = np.loadtxt(estimate_path, delimiter=',', skiprows=1, usecols=[0, *range(2, 44)], ndmin=2)
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) < 0):
        raise ValueError('invalid estimate values/timestamps')
    last = np.r_[np.diff(data[:, 0]) != 0, True]
    data = data[last]
    truth = np.loadtxt(truth_path, delimiter=',', skiprows=1, ndmin=2)
    if truth.shape != (len(data), 7) or not np.array_equal(data[:, 0], truth[:, 0]):
        raise ValueError('truth/estimate grid mismatch; no interpolation is allowed')
    times = data[:, 0].astype(np.int64)
    if times[0] != 0 or np.any(np.diff(times) != 10000):
        raise ValueError('missing 100 Hz evaluation samples')
    return times, data[:, 1:7], data[:, 7:].reshape(-1, 6, 6), truth[:, 1:]


def longest_clean_rejection(rows, corrupted):
    first = previous = None
    longest = 0.
    for row in rows:
        t = int(row['timestamp_us'])
        if row['status'] != 'rejected' or t in corrupted:
            first = previous = None
            continue
        if previous is None or t - previous > 200000:
            first = t
        longest = max(longest, (t - first) / 1e6)
        previous = t
    return longest


def evaluate(estimate_path, gps_path, truth_path, annotation, scenario, seed, method, kind, fixed_times):
    times, states, covariances, truth = load_states(estimate_path, truth_path)
    stats = state_statistics(states, truth, covariances)
    errors = np.linalg.norm(states[:, :2] - truth[:, :2], axis=1)
    interval = scenario['outage_us'] or [0, int(times[-1]) + 1]
    result = compute_metrics(times, states[:, :3], truth[:, :3], *interval)
    if scenario['outage_us'] is None:
        # The whole-run fallback above supports common pose metrics only.
        # A recording without an outage has no outage-window measurements.
        for key in list(result):
            if key.startswith('outage_'):
                result[key] = 0 if key == 'outage_samples' else None
    result.update(seed=seed, scenario=scenario['id'], method=method, kind=kind, status='complete',
                  max_position_error_m=float(errors.max()),
                  position_coverage_95_descriptive_time_fraction=float(np.nanmean(stats['position_covered_95'])),
                  full_nees_undefined=int((~stats['valid_full']).sum()),
                  position_nees_undefined=int((~stats['valid_position']).sum()),
                  max_abs_heading_error_rad=float(np.max(np.abs(stats['heading_error']))))
    events = audit_gps_events(gps_path)
    result.update(gps_decisions(events, annotation, method))
    corrupted = set(annotation['corrupted_gps_timestamps_us'])
    result['longest_clean_rejection_span_s'] = longest_clean_rejection(events, corrupted)
    result.update(settling_status='not_applicable', settling_start_delay_s=None,
                  settling_confirmation_delay_s=None, followup_s=None, first_clean_gps_accepted_delay_s=None)
    if scenario['recovery_windows']:
        window = scenario['recovery_windows'][0]
        reference, end = window['reference_us'], window['observation_end_us']
        result.update(elapsed_settling(times, errors, reference, end))
        accepted = [r['timestamp_us'] for r in events if reference <= r['timestamp_us'] <= end and
                    r['status'] == 'accepted' and r['timestamp_us'] not in corrupted]
        result['first_clean_gps_accepted_delay_s'] = (accepted[0] - reference) / 1e6 if accepted else None
    points = []
    for timestamp in fixed_times:
        indices = np.flatnonzero(times == timestamp)
        if len(indices) != 1:
            raise ValueError('missing predefined ensemble timestamp')
        index = indices[0]
        point = {'seed': seed, 'scenario': scenario['id'], 'method': method, 'kind': kind, 'timestamp_us': timestamp}
        point.update({key: float(values[index]) for key, values in stats.items()})
        point['position_error_m'] = float(errors[index])
        points.append(point)
    series = (times, states, covariances, truth, stats, events)
    return result, points, series


def summarize(points, runs, seeds_by_kind, cases, fixed_times_by_kind):
    expected_runs = {(kind, seed, scenario, method) for kind, scenarios in cases.items()
                     for seed in seeds_by_kind[kind] for scenario in scenarios
                     for method in (['ekf'] if kind == 'matched' else ['ekf', 'gated'])}
    actual_runs = [(r['kind'], r['seed'], r['scenario'], r['method']) for r in runs]
    expected_points = {(*key, t) for key in expected_runs for t in fixed_times_by_kind[key[0]]}
    actual_points = [(r['kind'], r['seed'], r['scenario'], r['method'], r['timestamp_us']) for r in points]
    if set(actual_runs) != expected_runs or len(actual_runs) != len(expected_runs):
        raise ValueError('incomplete or duplicate run matrix')
    if set(actual_points) != expected_points or len(actual_points) != len(expected_points):
        raise ValueError('incomplete or duplicate fixed-time matrix')
    ensemble = []
    for kind, scenarios in cases.items():
        seeds = seeds_by_kind[kind]
        methods = ['ekf'] if kind == 'matched' else ['ekf', 'gated']
        for scenario in scenarios:
            for method in methods:
                subset = [r for r in points if r['kind'] == kind and r['scenario'] == scenario and r['method'] == method]
                for timestamp in fixed_times_by_kind[kind]:
                    rows = [r for r in subset if r['timestamp_us'] == timestamp]
                    if {r['seed'] for r in rows} != set(seeds) or len(rows) != len(seeds):
                        raise ValueError('incomplete or duplicate ensemble')
                    for name, dimension in [('full', 6), ('position', 2)]:
                        summary = ensemble_nees([r.get(name + '_nees', np.nan) for r in rows], dimension)
                        summary.update(kind=kind, scenario=scenario, method=method, timestamp_us=timestamp, statistic=name,
                                       interpretation='approximate nonlinear matched-model reference' if kind == 'matched' else
                                       'descriptive stress/fault/gated comparison, not a calibration test')
                        coverage = np.array([r.get('position_covered_95', np.nan) for r in rows])
                        summary['coverage_valid_runs'] = int(np.isfinite(coverage).sum())
                        summary['position_covered_runs'] = int(np.sum(coverage == 1))
                        ensemble.append(summary)
    summary, paired = [], []
    for scenario in cases['stress']:
        for method in ['ekf', 'gated']:
            rows = [r for r in runs if r['kind'] == 'stress' and r['scenario'] == scenario and r['method'] == method]
            if len(rows) != len(seeds_by_kind['stress']):
                raise ValueError('incomplete run matrix')
            good = [r for r in rows if r['status'] == 'complete']
            summary.append(dict(scenario=scenario, method=method, total_runs=len(rows), numerical_or_output_failures=len(rows)-len(good),
                                rmse_median_m=float(np.median([r['position_rmse_m'] for r in good])) if good else None,
                                rmse_min_m=min((r['position_rmse_m'] for r in good), default=None),
                                rmse_max_m=max((r['position_rmse_m'] for r in good), default=None),
                                settled=sum(r['settling_start_delay_s'] is not None for r in good),
                                not_observed=sum(r['settling_status']=='not_observed' for r in good),
                                clean_gps_rejected=sum(r['clean_gps_rejected'] for r in good),
                                clean_gps_considered=sum(r['clean_gps_considered'] for r in good),
                                fault_gps_accepted=sum(r['fault_gps_accepted'] for r in good),
                                fault_gps_considered=sum(r['fault_gps_considered'] for r in good)))
        for seed in seeds_by_kind['stress']:
            rows = {r['method']:r for r in runs if r['kind']=='stress' and r['scenario']==scenario and r['seed']==seed}
            delta = (rows['gated']['position_rmse_m'] - rows['ekf']['position_rmse_m']
                     if all(r['status']=='complete' for r in rows.values()) else None)
            paired.append(dict(scenario=scenario, seed=seed, gated_minus_basic_rmse_m=delta))
    return ensemble, summary, paired


def compress_csv(path):
    original = sha256(path)
    with path.open('rb') as source, path.with_suffix('.csv.gz').open('xb') as destination:
        with gzip.GzipFile(fileobj=destination, mode='wb', mtime=0) as archive:
            shutil.copyfileobj(source, archive)
    path.unlink()
    return original


def run_study(binary, output, protocol_path=ROOT/'scenarios/consistency.json', smoke=False):
    binary, output = Path(binary).resolve(), Path(output).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    filter_path = ROOT/'scenarios/filter.conf'
    if sha256(filter_path) != protocol['frozen_filter_sha256']:
        raise ValueError('filter differs from frozen protocol')
    if (set(protocol['stress_seeds']) & set(protocol['matched_seeds']) or
            protocol['development_seed'] in protocol['stress_seeds'] + protocol['matched_seeds']):
        raise ValueError('seed roles overlap')
    helper = binary.with_name('navtrace_linearize')
    if any(not p.is_file() or not os.access(p, os.X_OK) for p in [binary, helper]):
        raise ValueError('build replay and offline linearize executables first')
    seeds = {'stress':[protocol['development_seed']] if smoke else protocol['stress_seeds'],
             'matched':[protocol['development_seed']] if smoke else protocol['matched_seeds']}
    scenarios = [s for s in protocol['scenarios'] if s['id']=='early_offset'] if smoke else protocol['scenarios']
    expected = len(seeds['stress'])*len(scenarios)*2+len(seeds['matched'])
    if expected > protocol['max_replays']:
        raise ValueError('study exceeds predefined run limit')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output/'protocol.json')
    shutil.copyfile(filter_path, output/'filter.conf')
    source_paths = [p for folder in ['include','src','apps','tools','tests','docs','scenarios'] for p in (ROOT/folder).rglob('*')
                    if p.is_file() and p.suffix in ['.cpp','.hpp','.py','.md','.json','.conf']]
    source_paths += [ROOT/n for n in ['CMakeLists.txt','requirements-lock.txt','AGENTS.md','README.md']]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running', seed_role='development' if smoke else 'held_out_m3',
                    seeds=seeds, expected_replays=expected, git_commit=git_output('rev-parse','HEAD'),
                    git_branch=git_output('branch','--show-current'), git_dirty=bool(git_output('status','--porcelain')),
                    binary=str(binary), binary_sha256=sha256(binary), linearize_sha256=sha256(helper), protocol_sha256=sha256(Path(protocol_path)),
                    filter_sha256=sha256(filter_path), python=sys.version, platform=platform.platform(),
                    dependencies={p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','pytest','cmake']},
                    build_info=(binary.parent/'build_info.txt').read_text(),
                    source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted(source_paths)}, commands=[],
                    uncompressed_output_sha256={}, failures=[])
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False)+'\n')
    save_manifest()
    begin = time.monotonic()
    runs, points = [], []
    try:
        with (output/'replay.log').open('x') as log:
            for kind in ['stress','matched']:
                for seed in seeds[kind]:
                    folder = output/f'{kind}-{seed}'
                    if kind=='stress':
                        annotations = generate_stress(folder/'inputs',seed,protocol)
                        planned = scenarios
                    else:
                        generate_matched(folder/'inputs',seed,filter_path,duration_s=protocol['matched_duration_s'])
                        with (folder/'inputs/nominal.csv').open() as stream:
                            gps_times=[int(r['timestamp_us']) for r in csv.DictReader(stream) if r['sensor']=='gps']
                        annotations={'matched':dict(gps_timestamps_us=gps_times,corrupted_gps_timestamps_us=[],truth_file='truth.csv')}
                        planned=[dict(id='matched',outage_us=None,recovery_windows=[])]
                    for scenario in planned:
                        plot_series={}
                        name=scenario['id']
                        annotation=annotations[name]
                        input_path=folder/f"inputs/{name if kind=='stress' else 'nominal'}.csv"
                        for method in (['ekf','gated'] if kind=='stress' else ['ekf']):
                            trajectory=folder/f'{name}-{method}.csv'
                            gps=folder/f'{name}-{method}-gps.csv'
                            command=[str(binary),str(input_path),str(output/'filter.conf'),method,str(trajectory),str(gps)]
                            manifest['commands'].append(command)
                            fixed=protocol[f'fixed_{kind}_times_us']
                            try:
                                proc=subprocess.run(command,capture_output=True,text=True)
                                log.write(' '.join(command)+'\n'+proc.stdout+proc.stderr); log.flush()
                                proc.check_returncode()
                                audit_event_outputs(input_path,trajectory,method)
                                result, samples, series=evaluate(trajectory,gps,folder/'inputs'/annotation['truth_file'],annotation,
                                                               scenario,seed,method,kind,fixed)
                                runs.append(result);points.extend(samples)
                                if kind=='stress' and seed==min(seeds[kind]):
                                    plot_series[method]=series
                            except Exception as error:
                                failure=dict(kind=kind,seed=seed,scenario=name,method=method,error=str(error))
                                manifest['failures'].append(failure)
                                runs.append(dict(**failure,status='failed'))
                                points.extend(dict(kind=kind,seed=seed,scenario=name,method=method,timestamp_us=t) for t in fixed)
                            finally:
                                for path in [trajectory,gps]:
                                    if path.exists():
                                        manifest['uncompressed_output_sha256'][str(path.relative_to(output))]=compress_csv(path)
                                save_manifest()
                        if plot_series:
                            from tools.plot_consistency import plot_case
                            plot_case(output,scenario,seed,plot_series)
                    print(f'Finished {kind} seed {seed}',flush=True)
        cases={'stress':[s['id'] for s in scenarios],'matched':['matched']}
        ensemble,summary,paired=summarize(points,runs,seeds,cases,{k:protocol[f'fixed_{k}_times_us'] for k in seeds})
        records(output/'metrics',runs); records(output/'fixed-time-statistics',points)
        records(output/'ensemble',ensemble); records(output/'summary',summary);records(output/'paired-comparisons',paired)
        run_observability(helper,output/'observability')
        manifest.update(status='complete' if not manifest['failures'] else 'complete_with_failures', replays=len(runs),
                        elapsed_s=time.monotonic()-begin)
        manifest['artifacts_sha256']={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        manifest['retained_bytes']=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        print(json.dumps(summary,indent=2))
        print(f"{len(runs)} replays; failures={len(manifest['failures'])}; {manifest['elapsed_s']:.1f} s; {manifest['retained_bytes']/1e6:.1f} MB; {output}")
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        manifest.update(attempted_replays=len(manifest['commands']),
                        completed_replays=sum(r['status']=='complete' for r in runs),
                        failed_replays=sum(r['status']=='failed' for r in runs),
                        not_started_replays=expected-len(manifest['commands']))
        if not (output/'metrics.json').exists():
            records(output/'metrics',runs)
            records(output/'fixed-time-statistics',points)
        save_manifest()
    return runs,ensemble


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='three development replays, no reserved seeds')
    args=parser.parse_args()
    run_study(args.binary,args.output,smoke=args.smoke)
