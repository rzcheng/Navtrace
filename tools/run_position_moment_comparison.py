"""One guarded offline comparison against the preserved conditional samples."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

_ENTRY_STARTED = time.monotonic()

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.conditional_reference import Observation
from tools.position_moment_metrics import METRICS, compare_samples, planned_rows, summarize
from tools.run_conditional_reference import read_observations, read_snapshots

PROTOCOL_SHA256 = '261309e0338ff048c5dc1626a7b24569930a390f6b51071e4eaaa09ab02a20e5'
PROTOCOL_PATH = Path('scenarios/position_moments.json')


def peak_memory_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**2 if sys.platform == 'darwin' else 1024)


class Budget:
    def __init__(self, started=None, wall_s=600, memory_mib=512):
        self.started = time.monotonic() if started is None else started
        self.wall_s, self.memory_mib = wall_s, memory_mib

    def check(self):
        if time.monotonic()-self.started > self.wall_s:
            raise TimeoutError('frozen 600 s wall budget exceeded')
        if peak_memory_mib() > self.memory_mib:
            raise MemoryError('frozen 512 MiB peak RSS budget exceeded')


def sha256(path, check_budget=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            check_budget()
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def json_write(path, value):
    # Keep NaN/Infinity out of records. Such values are computation failures.
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def safe_member(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'artifact escapes its input root: {name}')
    return path


def audit_sources(root, protocol, check_budget):
    for name, expected in protocol['preserved_source_sha256'].items():
        check_budget()
        if sha256(safe_member(root, name), check_budget) != expected:
            raise ValueError(f'preserved source hash differs: {name}')
    if sha256(root/'scenarios/filter.conf', check_budget) != protocol['filter_sha256']:
        raise ValueError('frozen filter configuration differs')


def audit_input(root, source_root, specification, check_budget):
    manifest_path = root/'manifest.json'
    if sha256(manifest_path, check_budget) != specification['manifest_sha256']:
        raise ValueError(f'frozen manifest hash differs: {root}')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['git_commit'] != specification['git_commit'] or manifest['git_dirty']
            or manifest['status'] != 'complete' or manifest['protocol_sha256'] != specification['protocol_sha256']):
        raise ValueError('unexpected historical manifest metadata')
    sources, artifacts = manifest['source_sha256'], manifest['artifacts_sha256']
    if len(sources) != specification['expected_source_count'] or len(artifacts) != specification['expected_artifact_count']:
        raise ValueError('frozen source or artifact count differs')
    for name, expected in sources.items():
        check_budget()
        historical = git(source_root, 'show', f"{specification['git_commit']}:{name}")
        if hashlib.sha256(historical).hexdigest() != expected:
            raise ValueError(f'historical source hash differs: {name}')
    for name, expected in artifacts.items():
        if sha256(safe_member(root, name), check_budget) != expected:
            raise ValueError(f'frozen artifact hash differs: {name}')
    if sha256(root/'protocol.json', check_budget) != specification['protocol_sha256']:
        raise ValueError('frozen input protocol differs')
    return manifest


def preflight(long_outage, reference, output, budget, root=ROOT):
    root, long_outage, reference, output = [Path(p).resolve() for p in [root, long_outage, reference, output]]
    budget.check()
    if output.exists():
        raise FileExistsError(f'output must be fresh: {output}')
    for protected in [long_outage, reference, root]:
        if protected == output or protected.is_relative_to(output):
            raise ValueError('output may not contain frozen inputs or source checkout')
    if output.is_relative_to(long_outage) or output.is_relative_to(reference):
        raise ValueError('output must be outside both frozen input roots')
    if output.is_relative_to(root) and output.relative_to(root).parts[0] != 'results':
        raise ValueError('output inside checkout must be in ignored results directory')
    if git(root, 'status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('comparison requires a clean committed checkout')
    protocol_path = root/PROTOCOL_PATH
    if sha256(protocol_path, budget.check) != PROTOCOL_SHA256:
        raise ValueError('pinned protocol hash differs')
    if hashlib.sha256(git(root, 'show', f'HEAD:{PROTOCOL_PATH}')).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('committed protocol hash differs')
    protocol = json.loads(protocol_path.read_text())
    specification = root/protocol['specification_path']
    if (sha256(specification, budget.check) != protocol['specification_sha256'] or
            hashlib.sha256(git(root, 'show', 'HEAD:'+protocol['specification_path'])).hexdigest() != protocol['specification_sha256']):
        raise ValueError('frozen specification differs')
    audit_sources(root, protocol, budget.check)
    inputs = {name: audit_input(path, root, protocol['frozen_inputs'][name], budget.check)
              for name, path in [('long_outage', long_outage), ('conditional_reference', reference)]}
    expected_samples = {f'case-{s}/n{n}-r{r}-t{t}.npz' for s in protocol['recording_seeds']
                        for n in protocol['sample_counts'] for r in protocol['reference_seeds']
                        for t in protocol['report_times_us']}
    actual_samples = {name for name in inputs['conditional_reference']['artifacts_sha256'] if name.endswith('.npz')}
    if actual_samples != expected_samples or len(actual_samples) != protocol['frozen_inputs']['conditional_reference']['expected_sample_arrays']:
        raise ValueError('saved sample matrix differs')
    if sha256(long_outage/'filter.conf', budget.check) != protocol['filter_sha256']:
        raise ValueError('saved filter configuration differs')
    if output.is_relative_to(root):
        relative = output.relative_to(root)
        result = subprocess.run(['git', '-C', str(root), 'check-ignore', str(relative)], stdout=subprocess.PIPE)
        if result.returncode:
            raise ValueError('comparison output must be ignored by Git')
    source_hashes = {}
    for name in git(root, 'ls-files', '-z').decode().split('\0'):
        if name and (root/name).is_file():
            source_hashes[name] = sha256(root/name, budget.check)
    budget.check()
    return protocol, dict(
        git_commit=git(root, 'rev-parse', 'HEAD').decode().strip(),
        git_branch=git(root, 'branch', '--show-current').decode().strip(), git_dirty=False,
        source_sha256=source_hashes,
        input_roots={'long_outage': str(long_outage), 'conditional_reference': str(reference)},
        inputs={name: dict(manifest_sha256=protocol['frozen_inputs'][name]['manifest_sha256'],
                          git_commit=m['git_commit'], source_sha256=m['source_sha256'],
                          artifacts_sha256=m['artifacts_sha256']) for name, m in inputs.items()},
        verified_artifact_count=sum(len(m['artifacts_sha256']) for m in inputs.values()),
        verified_sample_count=len(actual_samples))


def load_case(long_outage, reference, recording, protocol, check_budget):
    check_budget()
    end = protocol['report_times_us'][-1]
    original = long_outage/f'seed-{recording}'
    saved = reference/f'case-{recording}'
    snapshots = read_snapshots(original/'dropout-gated.csv.gz', protocol['report_times_us'], end)
    saved_snapshots = json.loads((saved/'ekf-snapshots.json').read_text())
    for timestamp, (mean, covariance) in snapshots.items():
        check_budget()
        other = saved_snapshots[str(timestamp)]
        if not np.array_equal(mean, other['mean']) or not np.array_equal(covariance, other['covariance']):
            raise ValueError('saved comparator differs from original C++ recording')
    checkpoint = json.loads((saved/'checkpoint.json').read_text())
    initial_mean, initial_covariance = snapshots[protocol['checkpoint_us']]
    if (checkpoint['timestamp_us'] != protocol['checkpoint_us'] or checkpoint['phase'] != 'post_events'
            or not np.array_equal(initial_mean, checkpoint['mean'])
            or not np.array_equal(initial_covariance, checkpoint['covariance'])):
        raise ValueError('saved checkpoint differs from original C++ recording')
    events = read_observations(original/'inputs/dropout.csv', protocol['checkpoint_us'], end)
    with (saved/'observations.csv').open(newline='') as stream:
        saved_events = [Observation(int(r['timestamp_us']), r['sensor'], float(r['value']), float(r['variance']))
                        for r in csv.DictReader(stream)]
    if events != saved_events:
        raise ValueError('saved observation prefix differs from original observations')
    counts = [sum(event.timestamp_us <= timestamp for event in events) for timestamp in protocol['report_times_us']]
    if counts != protocol['expected_observation_counts']:
        raise ValueError('observation prefix counts differ')
    config = dict(line.split('=', 1) for line in (long_outage/'filter.conf').read_text().splitlines()
                  if line and not line.startswith('#'))
    diffusion = [float(config[k]) for k in ['speed_diffusion_m2_per_s3', 'yaw_rate_diffusion_rad2_per_s3', 'bias_diffusion_rad2_per_s3']]
    check_budget()
    return snapshots, events, diffusion, float(config['max_step_s']), float(config['max_gap_s'])


def validate_candidate(snapshot, baseline, expected_count, tolerance, checkpoint=False):
    mean, covariance = np.asarray(snapshot['mean']), np.asarray(snapshot['covariance'])
    if mean.shape != (6,) or covariance.shape != (6, 6) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('invalid candidate snapshot')
    scale = 1e-10*max(1., np.max(np.abs(covariance)))
    if np.max(np.abs(covariance-covariance.T)) > scale or np.linalg.eigvalsh(covariance)[0] < -scale:
        raise ValueError('invalid candidate snapshot covariance')
    difference = mean[2:]-baseline[0][2:]
    difference[0] = (difference[0]+np.pi) % (2*np.pi)-np.pi
    mean_error = float(np.max(np.abs(difference)))
    covariance_error = float(np.max(np.abs(covariance[2:, 2:]-baseline[1][2:, 2:])))
    result = dict(max_mean_difference=mean_error, max_covariance_difference=covariance_error,
                  observations=int(snapshot['observations']), expected_observations=expected_count)
    if mean_error > tolerance or covariance_error > tolerance or snapshot['observations'] != expected_count:
        raise ValueError('candidate latent marginal or observation count differs: '+json.dumps(result))
    if checkpoint and (not np.array_equal(mean, baseline[0]) or not np.array_equal(covariance, baseline[1])):
        raise ValueError('candidate checkpoint must equal input exactly')
    return result


def mark_failed(row, error):
    row.update(status='failed', error=str(error))
    for metric in row['metrics'].values():
        if metric['status'] == 'not_started':
            metric.update(status='failed', error=str(error))


def recheck_inputs(root, protocol, metadata, budget):
    audit_sources(root, protocol, budget.check)
    for name, info in metadata['inputs'].items():
        base = Path(metadata['input_roots'][name])
        if sha256(base/'manifest.json', budget.check) != info['manifest_sha256']:
            raise ValueError(f'input manifest changed during comparison: {name}')
        for path, digest in info['artifacts_sha256'].items():
            if sha256(safe_member(base, path), budget.check) != digest:
                raise ValueError(f'input changed during comparison: {name}/{path}')
    for name, digest in metadata['source_sha256'].items():
        if sha256(root/name, budget.check) != digest:
            raise ValueError(f'source changed during comparison: {name}')
    if (git(root, 'status', '--porcelain', '--untracked-files=all').strip() or
            git(root, 'rev-parse', 'HEAD').decode().strip() != metadata['git_commit']):
        raise ValueError('checkout changed during comparison')


def run_comparison(long_outage, reference, output, *, root=ROOT, started=None):
    budget = Budget(started)
    long_outage, reference, output, root = [Path(p).resolve() for p in [long_outage, reference, output, root]]
    protocol, metadata = preflight(long_outage, reference, output, budget, root)
    # Only import the independently reviewed candidate after the complete input audit.
    from tools.position_moments import run_candidate
    output.mkdir(parents=True, exist_ok=False)
    rows = planned_rows(protocol)
    cases = [dict(recording_seed=s, timestamp_us=t, status='not_started')
             for s in protocol['recording_seeds'] for t in protocol['report_times_us']]
    pooled, outcomes = summarize(rows, protocol)
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, protocol_sha256=PROTOCOL_SHA256, specification_sha256=protocol['specification_sha256'],
        python=sys.version, platform=platform.platform(),
        dependencies={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'matplotlib', 'pytest']},
        wall_budget_s=600, memory_budget_mib=512, expected_case_time_entries=len(cases),
        expected_distribution_rows=len(rows), expected_metric_entries=len(rows)*len(METRICS),
        expected_resolution_entries=len(outcomes), failures=[], plots_status='not_started',
        final_input_audit_status='not_started', **metadata)
    # Persist the complete matrix before computation, so interrupted work stays visible.
    json_write(output/'comparisons.json', rows)
    json_write(output/'case-snapshots.json', cases)
    json_write(output/'resolution.json', outcomes)
    json_write(output/'manifest.json', manifest)
    active_row = active_case = None
    budget_stopped = False
    try:
        budget.check()
        (output/'protocol.json').write_bytes((root/PROTOCOL_PATH).read_bytes())
        (output/'position_moment_protocol.md').write_bytes((root/protocol['specification_path']).read_bytes())
        for recording in protocol['recording_seeds']:
            snapshots = candidate = None
            try:
                active_case = next(c for c in cases if c['recording_seed'] == recording)
                snapshots, events, diffusion, max_step, max_gap = load_case(long_outage, reference, recording, protocol, budget.check)
                initial = snapshots[protocol['checkpoint_us']]
                # Pass only checkpoint, original observations, timestamps, and fixed config.
                candidate = run_candidate(initial[0].copy(), initial[1].copy(), protocol['checkpoint_us'], events,
                    protocol['report_times_us'], diffusion, max_step_s=max_step, max_gap_s=max_gap, check_budget=budget.check)
            except (TimeoutError, MemoryError):
                raise
            except Exception as error:
                manifest['failures'].append(dict(recording_seed=recording, stage='candidate_or_input', error=str(error)))
            for index, timestamp in enumerate(protocol['report_times_us']):
                active_case = next(c for c in cases if (c['recording_seed'], c['timestamp_us']) == (recording, timestamp))
                budget.check()
                baseline = snapshots[timestamp] if snapshots is not None else None
                proposed = None
                if baseline is not None:
                    active_case['baseline'] = dict(mean=baseline[0].tolist(), covariance=baseline[1].tolist())
                try:
                    if candidate is None or baseline is None:
                        raise ValueError('candidate or original snapshots unavailable')
                    proposed = candidate[timestamp]
                    agreement = validate_candidate(proposed, baseline, protocol['expected_observation_counts'][index],
                        protocol['latent_agreement_absolute_tolerance'], checkpoint=index == 0)
                    active_case['candidate'] = dict(mean=np.asarray(proposed['mean']).tolist(),
                        covariance=np.asarray(proposed['covariance']).tolist(), observations=int(proposed['observations']))
                    active_case.update(status='complete', latent_agreement=agreement)
                except Exception as error:
                    active_case.update(status='failed', error=str(error))
                    manifest['failures'].append(dict(recording_seed=recording, timestamp_us=timestamp, stage='snapshot', error=str(error)))
                    proposed = None
                for row in rows:
                    if (row['recording_seed'], row['timestamp_us']) != (recording, timestamp):
                        continue
                    active_row = row
                    try:
                        budget.check()
                        if baseline is None:
                            raise ValueError('original comparator unavailable')
                        path = reference/f"case-{recording}/n{row['sample_count']}-r{row['reference_seed']}-t{timestamp}.npz"
                        with np.load(path, allow_pickle=False) as archive:
                            samples = archive['states']
                        if samples.shape != (row['sample_count'], 6):
                            raise ValueError('saved sample array shape differs')
                        begin = time.monotonic()
                        with np.errstate(over='raise', invalid='raise', divide='raise'):
                            result = compare_samples(samples, *baseline,
                                proposed['mean'] if proposed else None, proposed['covariance'] if proposed else None,
                                protocol['batches'], budget.check)
                        row.update(result, elapsed_s=time.monotonic()-begin)
                        budget.check()
                    except (TimeoutError, MemoryError):
                        raise
                    except Exception as error:
                        mark_failed(row, error)
                        manifest['failures'].append(dict(recording_seed=recording, timestamp_us=timestamp,
                            sample_count=row['sample_count'], reference_seed=row['reference_seed'], stage='metrics', error=str(error)))
                    active_row = None
                json_write(output/'case-snapshots.json', cases)
                json_write(output/'comparisons.json', rows)
                active_case = None
        budget.check()
        pooled, outcomes = summarize(rows, protocol)
        json_write(output/'pooled.json', pooled)
        json_write(output/'resolution.json', outcomes)
        budget.check()
        from tools.plot_position_moments import plot_results
        try:
            plot_results(output, reference, cases, pooled, outcomes, protocol, budget.check)
            manifest['plots_status'] = 'complete'
        except (TimeoutError, MemoryError):
            manifest['plots_status'] = 'failed'
            raise
        except Exception as error:
            manifest['plots_status'] = 'failed'
            manifest['failures'].append(dict(stage='plots', error=str(error)))
        budget.check()
        try:
            recheck_inputs(root, protocol, metadata, budget)
            manifest['final_input_audit_status'] = 'complete'
        except Exception:
            manifest['final_input_audit_status'] = 'failed'
            raise
    except (TimeoutError, MemoryError) as error:
        budget_stopped = True
        if active_row is not None:
            mark_failed(active_row, error)
        elif active_case is not None:
            active_case.update(status='failed', error=str(error))
        manifest['failures'].append(dict(stage='budget', error=str(error)))
    except Exception as error:
        if active_row is not None:
            mark_failed(active_row, error)
        elif active_case is not None:
            active_case.update(status='failed', error=str(error))
        manifest['failures'].append(dict(stage='execution', error=str(error)))
    # Retain the last completed summaries after a budget stop; do not resume metrics.
    if not budget_stopped:
        try:
            budget.check()
            pooled, outcomes = summarize(rows, protocol)
        except (TimeoutError, MemoryError) as error:
            budget_stopped = True
            manifest['failures'].append(dict(stage='summary_budget', error=str(error)))
    json_write(output/'comparisons.json', rows)
    json_write(output/'case-snapshots.json', cases)
    json_write(output/'pooled.json', pooled)
    json_write(output/'resolution.json', outcomes)
    manifest.update(completed_distributions=sum(r['status'] == 'complete' for r in rows),
        failed_distributions=sum(r['status'] == 'failed' for r in rows),
        not_started_distributions=sum(r['status'] == 'not_started' for r in rows),
        completed_metrics=sum(m['status'] == 'complete' for r in rows for m in r['metrics'].values()),
        resolved_metrics=sum(r['resolved'] for r in outcomes), inconclusive_metrics=sum(not r['resolved'] for r in outcomes),
        budget_stopped=budget_stopped, artifacts_sha256={})
    try:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'manifest.json':
                manifest['artifacts_sha256'][str(path.relative_to(output))] = sha256(path, budget.check)
        manifest['output_hash_status'] = 'complete'
        budget.check()
    except (TimeoutError, MemoryError) as error:
        manifest['output_hash_status'] = 'failed'
        manifest['budget_stopped'] = True
        manifest['failures'].append(dict(stage='output_hashes', error=str(error)))
    required_complete = (all(r['status'] == 'complete' for r in rows) and all(c['status'] == 'complete' for c in cases)
                        and all(o['status'] == 'complete' for o in outcomes)
                        and manifest['plots_status'] == 'complete' and manifest['final_input_audit_status'] == 'complete'
                        and manifest['output_hash_status'] == 'complete')
    manifest.update(status='complete' if required_complete and not manifest['failures'] else 'failed',
                    elapsed_s=time.monotonic()-budget.started, peak_memory_mib=peak_memory_mib())
    json_write(output/'manifest.json', manifest)
    # Include final record writing in the wall/RSS accounting and terminal check.
    try:
        budget.check()
    except (TimeoutError, MemoryError) as error:
        manifest.update(status='failed', budget_stopped=True)
        manifest['failures'].append(dict(stage='final_record_budget', error=str(error)))
    manifest.update(elapsed_s=time.monotonic()-budget.started, peak_memory_mib=peak_memory_mib())
    json_write(output/'manifest.json', manifest)
    return rows, manifest


def main():
    started = _ENTRY_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-outage', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_existed_before = args.output.exists()
    try:
        _, manifest = run_comparison(args.long_outage, args.reference, args.output, started=started)
    except Exception as error:
        created_output = not output_existed_before and args.output.exists()
        stage = 'failed after output creation' if created_output else 'refused before output'
        print(f'Comparison {stage}: {error}', file=sys.stderr)
        return 2
    print(json.dumps({k: manifest[k] for k in ['status', 'completed_distributions', 'completed_metrics',
        'resolved_metrics', 'elapsed_s', 'peak_memory_mib']}, indent=2))
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
