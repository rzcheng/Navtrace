"""Reproduce consumed M2 cases with richer logging, without changing estimates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.event_diagnostics import audit_gps_events
from tools.consistency import state_statistics
from tools.run_demo import sha256,git_output
from tools.run_consistency_study import records,load_states
from tools.plot_consistency import plot_case


def post_return_sensor_corrections(path, start_us=45000000):
    """Add up each sensor's corrections after GPS returns in the 100 Hz recording."""
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    state_columns = ['px_m', 'py_m', 'yaw_rad', 'speed_m_per_s',
                     'yaw_rate_rad_per_s', 'gyro_bias_rad_per_s']
    totals = {
        sensor: dict(events=0, position_correction_path_m=0.,
                     absolute_yaw_correction_rad=0., absolute_bias_correction_rad_per_s=0.)
        for sensor in ['gyro', 'wheel', 'gps']
    }
    for previous, current in zip(rows, rows[1:]):
        if int(current['timestamp_us']) < start_us:
            continue
        before = np.array([float(previous[column]) for column in state_columns])
        after = np.array([float(current[column]) for column in state_columns])
        dt = (int(current['timestamp_us']) - int(previous['timestamp_us'])) / 1e6
        if not 0 <= dt <= .01000000001:
            raise ValueError('event decomposition requires original complete 100 Hz recording')

        # First account for motion between samples, then measure the correction.
        predicted = before.copy()
        predicted[:2] += dt * before[3] * np.array([np.cos(before[2]), np.sin(before[2])])
        predicted[2] += dt * before[4]
        change = after - predicted
        change[2] = (change[2] + np.pi) % (2 * np.pi) - np.pi

        sensor_total = totals[current['sensor']]
        sensor_total['events'] += 1
        sensor_total['position_correction_path_m'] += float(np.linalg.norm(change[:2]))
        sensor_total['absolute_yaw_correction_rad'] += abs(float(change[2]))
        sensor_total['absolute_bias_correction_rad_per_s'] += abs(float(change[5]))
    return [dict(sensor=sensor, **values) for sensor, values in totals.items()]


def diagnose(binary,output,source=ROOT/'results/gps-fault-study'):
    binary,output,source=Path(binary).resolve(),Path(output).resolve(),Path(source).resolve()
    output.mkdir(parents=True,exist_ok=False)
    old_manifest=json.loads((source/'manifest.json').read_text())
    config=ROOT/'scenarios/filter.conf'
    if sha256(config)!=old_manifest['filter_sha256']:
        raise ValueError('M2 filter hash differs')
    protocol=json.loads((source/'protocol.json').read_text())
    all_events,selected,comparisons,components=[],[],[],[]
    manifest=dict(git_commit=git_output('rev-parse','HEAD'),git_dirty=bool(git_output('status','--porcelain')),
                  binary_sha256=sha256(binary),filter_sha256=sha256(config),source_manifest_sha256=sha256(source/'manifest.json'),
                  seed_role='consumed_M2_diagnosis',commands=[],input_sha256={})
    for seed in [70001,70002,70003]:
        old=source/f'seed-{seed}';folder=output/f'seed-{seed}';folder.mkdir()
        truth=np.loadtxt(old/'inputs/base/truth.csv',delimiter=',',skiprows=1)
        provenance=json.loads((old/'inputs/base/generator.json').read_text())
        bias=provenance['settings']['true_gyro_bias_rad_per_s']
        truth=np.column_stack([truth,np.full(len(truth),bias)])
        truth_path=folder/'truth.csv'
        np.savetxt(truth_path,truth,delimiter=',',header='timestamp_us,px_m,py_m,yaw_rad,speed_m_per_s,yaw_rate_rad_per_s,gyro_bias_rad_per_s',comments='')
        for case in ['nominal','dropout_offset']:
            plot_series={}
            for method in ['ekf','gated']:
                trajectory=folder/f'{case}-{method}.csv';gps=folder/f'{case}-{method}-gps.csv'
                input_path=old/f'inputs/{case}.csv'
                command=[str(binary),str(input_path),str(config),method,str(trajectory),str(gps)]
                manifest['commands'].append(command);manifest['input_sha256'][str(input_path)]=sha256(input_path)
                subprocess.run(command,check=True,capture_output=True,text=True)
                if sha256(trajectory)!=sha256(old/trajectory.name):
                    raise ValueError('diagnostics changed M2 trajectory bytes')
                with (old/gps.name).open() as stream:
                    reader=csv.DictReader(stream);fields=reader.fieldnames;before=list(reader)
                with gps.open() as stream:
                    after=list(csv.DictReader(stream))
                if before!=[{k:r[k] for k in fields} for r in after]:
                    raise ValueError('diagnostics changed an existing M2 GPS field')
                if case=='dropout_offset':
                    components.extend(dict(seed=seed,method=method,**row) for row in post_return_sensor_corrections(trajectory))
                events=audit_gps_events(gps)
                for row in events:
                    row.update(seed=seed,scenario=case,method=method)
                all_events.extend(events)
                picked={1600000,1800000,40000000,44800000,45000000,46000000,50000000,60000000}
                accepted=[r['timestamp_us'] for r in events if 40000000<=r['timestamp_us']<45000000 and r['status']=='accepted']
                if accepted: picked.add(accepted[0])
                if case=='dropout_offset':
                    selected.extend(r for r in events if r['timestamp_us'] in picked)
                times,states,p,actual=load_states(trajectory,truth_path)
                stats=state_statistics(states,actual,p)
                error=np.linalg.norm(states[:,:2]-actual[:,:2],axis=1)
                comparisons.append(dict(seed=seed,scenario=case,method=method,trajectory_bytes_identical=True,original_gps_columns_identical=True,
                                        rmse_m=float(np.sqrt(np.mean(error**2))),final_error_m=float(error[-1]),
                                        clean_return_rejected=sum(r['timestamp_us']>=45000000 and r['status']=='rejected' for r in events)))
                plot_series[method]=(times,states,p,actual,stats,events)
            if seed==70002 and case=='dropout_offset':
                scenario=next(s for s in protocol['scenarios'] if s['id']==case)
                plot_case(output,scenario,seed,plot_series)
    records(output/'sensor-corrections-after-45s',components)
    records(output/'event-table',selected);records(output/'all-gps-audit',all_events);records(output/'reproduction',comparisons)
    manifest['status']='complete';manifest['artifacts_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diagnose(args.binary,args.output)
