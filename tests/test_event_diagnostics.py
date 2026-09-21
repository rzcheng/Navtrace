import csv
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.event_diagnostics import audit_gps_events
from tools.generate_scenario import ROOT, SENSOR_HEADER


@pytest.fixture
def binary():
    path = Path(os.environ.get("NAVTRACE_BINARY", ROOT / "build/navtrace_replay"))
    if not path.exists():
        pytest.skip("build C++ executable first for event diagnostics tests")
    return path.resolve()


def replay(binary, tmp_path, name, events, config=None, method="gated"):
    sensors = tmp_path / f"{name}-input.csv"
    sensors.write_text(",".join(SENSOR_HEADER) + "\n" + "\n".join(events) + "\n")
    trajectory, gps = tmp_path / f"{name}.csv", tmp_path / f"{name}-gps.csv"
    execution = subprocess.run([str(binary), str(sensors), str(config or ROOT / "scenarios/filter.conf"),
                                method, str(trajectory), str(gps)], text=True, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    return trajectory, gps


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def rewrite_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_real_cli_snapshots_match_hand_computable_gps_update(tmp_path, binary):
    config = tmp_path / "hand.conf"
    config.write_text((ROOT / "scenarios/filter.conf").read_text()
                      .replace("initial_px=0", "initial_px=10").replace("variance_px=0.04", "variance_px=4"))
    trajectory, gps = replay(binary, tmp_path, "hand", ["0,gps,12,0,9,0,9"], config, "ekf")
    original = read_rows(gps)[0]
    row = audit_gps_events(gps)[0]
    assert float(original["pre_x0"]) == 10
    assert float(original["pre_p00"]) == 4
    assert float(original["post_x0"]) == pytest.approx(10 + 8 / 13)
    assert float(original["post_p00"]) == pytest.approx(36 / 13)
    assert row["nis"] == pytest.approx(4 / 13)
    assert row["residual_x_m"] == 2 and row["s00"] == 13
    assert row["delta_px"] == pytest.approx(8 / 13)
    assert row["mean_max_abs_error"] < 1e-12 and row["joseph_max_abs_error"] < 1e-12
    assert row["since_available_s"] is None and row["since_accepted_s"] is None
    assert float(read_rows(trajectory)[0]["px_m"]) == float(original["post_x0"])


def test_real_cli_prefix_and_future_suffix_are_causal_and_snapshots_bracket_only_gps(tmp_path, binary):
    prefix = ["0,gyro,0.015,0,0.000064,0,0", "0,wheel,1.5,0,0.0025,0,0",
              "0,gps,0,0.05,0.36,0,0.36", "100000,gyro,0.015,0,0.000064,0,0",
              "200000,wheel,1.45,0,0.0025,0,0", "200000,gps,100,100,0.36,0,0.36"]
    suffix = ["300000,gyro,0.02,0,0.000064,0,0", "400000,wheel,1.5,0,0.0025,0,0",
              "400000,gps,0.6,0,0.36,0,0.36", "600000,gps,100,100,0.36,0,0.36",
              "800000,gps,100,100,0.36,0,0.36"]
    short_state, short_gps = replay(binary, tmp_path, "prefix", prefix)
    long_state, long_gps = replay(binary, tmp_path, "long", prefix + suffix)
    changed_state, changed_gps = replay(binary, tmp_path, "changed", prefix + [s.replace("100,100", "-500,500") for s in suffix])
    # Output bytes for the entire prefix, including all new fields, cannot
    # depend on receiving a suffix or on changes to its measurements.
    for short, full, changed in ((short_state, long_state, changed_state), (short_gps, long_gps, changed_gps)):
        assert full.read_bytes().startswith(short.read_bytes())
        assert changed.read_bytes().startswith(short.read_bytes())
    audit = audit_gps_events(long_gps)
    assert [row["status"] for row in audit] == ["accepted", "rejected", "accepted", "rejected", "rejected"]
    assert audit[-1]["since_available_s"] == pytest.approx(.2)
    assert audit[-1]["since_accepted_s"] == pytest.approx(.4)
    assert audit[-1]["rejection_duration_s"] == pytest.approx(.2)
    states, raw_gps = read_rows(long_state), read_rows(long_gps)
    names = ["px_m", "py_m", "yaw_rad", "speed_m_per_s", "yaw_rate_rad_per_s", "gyro_bias_rad_per_s"]
    for index, state in enumerate(states):
        if state["sensor"] != "gps":
            continue
        event = next(row for row in raw_gps if row["timestamp_us"] == state["timestamp_us"])
        np.testing.assert_array_equal([float(event[f"post_x{i}"]) for i in range(6)], [float(state[k]) for k in names])
        if index > 0 and states[index - 1]["timestamp_us"] == state["timestamp_us"]:
            before = states[index - 1]
            np.testing.assert_array_equal([float(event[f"pre_x{i}"]) for i in range(6)], [float(before[k]) for k in names])
            np.testing.assert_array_equal([float(event[f"pre_p{i}{j}"]) for i in range(6) for j in range(6)],
                                          [float(before[f"p{i}{j}"]) for i in range(6) for j in range(6)])


def test_auditor_rejects_false_nis_and_rejected_corrections(tmp_path, binary):
    _, gps = replay(binary, tmp_path, "rejected", ["0,gps,100,100,0.36,0,0.36"])
    assert audit_gps_events(gps)[0]["status"] == "rejected"
    original = read_rows(gps)
    wrong = [dict(original[0], post_x0="0.1")]
    rewrite_rows(gps, wrong)
    with pytest.raises(ValueError, match="rejected GPS applied"):
        audit_gps_events(gps)
    rewrite_rows(gps, [dict(original[0], nis=str(float(original[0]["nis"]) + 1))])
    with pytest.raises(ValueError, match="pre-update NIS"):
        audit_gps_events(gps)
    rewrite_rows(gps, [dict(original[0], pre_x0="1")])
    with pytest.raises(ValueError, match="pre-update residual"):
        audit_gps_events(gps)
