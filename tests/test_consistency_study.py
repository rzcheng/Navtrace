import copy
import json
import os
from pathlib import Path

import pytest

from tools.run_consistency_study import ROOT, run_study, summarize


def small_records():
    seeds={'stress':[], 'matched':[1,2]}
    cases={'stress':[], 'matched':['matched']}
    times={'stress':[], 'matched':[0,10000]}
    runs=[dict(kind='matched',seed=s,scenario='matched',method='ekf',status='complete' if s==1 else 'failed') for s in seeds['matched']]
    points=[dict(kind='matched',seed=s,scenario='matched',method='ekf',timestamp_us=t,
                 **({'full_nees':6.,'position_nees':2.,'position_covered_95':1.} if s==1 else {})) for s in seeds['matched'] for t in times['matched']]
    return points,runs,seeds,cases,times


def test_failed_runs_remain_in_fixed_timestamp_denominator():
    ensemble,summary,paired=summarize(*small_records())
    assert len(ensemble)==4
    for row in ensemble:
        assert row['independent_runs']==2 and row['failed_runs']==1
        assert row['mean_nees'] is None and row['reference_lower'] is None
        assert row['coverage_valid_runs']==1 and row['position_covered_runs']==1


def test_missing_whole_timestamp_and_duplicate_run_are_errors():
    data=list(small_records());data[0]=[p for p in data[0] if p['timestamp_us']!=10000]
    with pytest.raises(ValueError,match='fixed-time matrix'):
        summarize(*data)
    data=list(small_records());data[1].append(data[1][0])
    with pytest.raises(ValueError,match='run matrix'):
        summarize(*data)


def test_missing_helper_fails_before_output_creation(tmp_path):
    fake=tmp_path/'navtrace_replay';fake.write_text('');fake.chmod(0o700)
    with pytest.raises(ValueError,match='linearize'):
        run_study(fake,tmp_path/'output',smoke=True)
    assert not (tmp_path/'output').exists()


def test_actual_three_replay_smoke_has_paired_inputs_and_accounted_outputs(tmp_path):
    binary=Path(os.environ.get('NAVTRACE_BINARY',ROOT/'build/navtrace_replay'))
    if not binary.exists():
        pytest.skip('build C++ first')
    output=tmp_path/'study'
    runs,ensemble=run_study(binary,output,smoke=True)
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='complete' and len(runs)==3
    matched=next(r for r in runs if r['kind']=='matched')
    assert matched['outage_samples']==0
    assert matched['outage_position_rmse_m'] is None
    assert matched['outage_end_evaluation_timestamp_us'] is None
    assert manifest['attempted_replays']==manifest['completed_replays']==3
    assert manifest['failed_replays']==manifest['not_started_replays']==0
    assert manifest['commands'][0][1]==manifest['commands'][1][1]
    assert len(manifest['uncompressed_output_sha256'])==6
    assert len(list(output.rglob('*-gps.csv.gz')))==3
    assert all('truth' not in ' '.join(c) and 'annotations' not in ' '.join(c) for c in manifest['commands'])
    assert all(r['independent_runs']==1 for r in ensemble)
    with pytest.raises(FileExistsError):
        run_study(binary,output,smoke=True)
