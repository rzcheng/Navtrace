"""Derive GPS fault variants without changing other sensor realizations."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from tools.generate_scenario import SENSOR_HEADER, generate, validate_settings, write_csv


def validate_protocol(protocol):
    settings = dict(protocol["generator_settings"], seed=protocol["development_seed"])
    validate_settings(settings)
    seeds = protocol["evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("evaluation seeds must be unique nonnegative integers")
    if protocol["development_seed"] in seeds:
        raise ValueError("development and evaluation seeds must be disjoint")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol["frozen_filter_sha256"]):
        raise ValueError("expected frozen SHA-256 for filter configuration")
    duration_us = round(settings["duration_s"] * 1e6)
    gps_step = 1_000_000 // settings["gps_hz"]
    step = 1_000_000 // settings["gyro_hz"]
    def interval(bounds, grid):
        if (len(bounds) != 2 or any(type(t) is not int or t % grid for t in bounds)
                or not 0 <= bounds[0] < bounds[1] <= duration_us):
            raise ValueError("invalid interval or timestamp grid")
    interval(protocol["reference_window_us"], step)
    settling = protocol["settling"]
    if not np.isfinite(settling["position_limit_m"]) or settling["position_limit_m"] <= 0:
        raise ValueError("settling position limit must be positive and finite")
    if type(settling["dwell_us"]) is not int or settling["dwell_us"] <= 0 or settling["dwell_us"] % step:
        raise ValueError("settling dwell must be a positive multiple of the evaluation step")
    ids = []
    for scenario in protocol["scenarios"]:
        ids.append(scenario["id"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scenario["id"]):
            raise ValueError("invalid scenario id")
        fault_intervals = []
        if scenario["outage_us"] is not None:
            interval(scenario["outage_us"], gps_step)
            fault_intervals.append(scenario["outage_us"])
        for offset in scenario["offsets"]:
            bounds = [offset["start_us"], offset["end_us"]]
            interval(bounds, gps_step)
            delta = np.asarray(offset["offset_m"], dtype=float)
            if delta.shape != (2,) or not np.all(np.isfinite(delta)) or np.all(delta == 0):
                raise ValueError("GPS offset must be a nonzero finite 2-vector")
            fault_intervals.append(bounds)
        ordered = sorted(fault_intervals)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("fault intervals cannot overlap")
        windows = scenario["recovery_windows"]
        if len({w["id"] for w in windows}) != len(windows):
            raise ValueError("duplicate recovery window")
        for window in windows:
            start, end = window["reference_us"], window["observation_end_us"]
            interval([start, end], step)
            if start % gps_step or any(a <= end and b > start for a, b in fault_intervals):
                raise ValueError("recovery window must begin on GPS grid and contain no faults")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be nonempty and unique")


def apply_faults(rows, scenario):
    """Preserve input order/strings except GPS values explicitly changed here."""
    output, corrupted, removed = [], [], []
    outage = scenario["outage_us"]
    for original in rows:
        row = list(original)
        time = int(row[0])
        if row[1] == "gps":
            if outage is not None and outage[0] <= time < outage[1]:
                removed.append(time)
                continue
            for fault in scenario["offsets"]:
                if fault["start_us"] <= time < fault["end_us"]:
                    row[2] = repr(float(row[2]) + fault["offset_m"][0])
                    row[3] = repr(float(row[3]) + fault["offset_m"][1])
                    corrupted.append(time)
                    break
        output.append(row)
    annotation = {
        "corrupted_gps_timestamps_us": corrupted, "removed_gps_timestamps_us": removed,
        "gps_timestamps_us": [int(r[0]) for r in output if r[1] == "gps"],
        "sensor_event_count": len(output), "recovery_windows": scenario["recovery_windows"],
    }
    return output, annotation


def generate_suite(output, protocol, seed, seed_role):
    validate_protocol(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(protocol["generator_settings"], seed=seed)
    base = generate(output / "base", settings, seed_role=seed_role)
    with (base / "nominal.csv").open(newline="") as file:
        reader = csv.reader(file)
        if next(reader) != SENSOR_HEADER:
            raise ValueError("unexpected base sensor schema")
        nominal = list(reader)
    annotations = {"seed": seed, "seed_role": seed_role, "protocol_id": protocol["protocol_id"], "scenarios": {}}
    for scenario in protocol["scenarios"]:
        rows, annotation = apply_faults(nominal, scenario)
        write_csv(output / f"{scenario['id']}.csv", SENSOR_HEADER, rows)
        annotations["scenarios"][scenario["id"]] = annotation
    (output / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    return annotations
