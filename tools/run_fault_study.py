"""Execute the frozen M2 protocol using compiled C++ estimates."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fault_scenarios import generate_suite, validate_protocol
from tools.fault_metrics import evaluate_case, summarize
from tools.plot_fault_study import plot_study
from tools.run_demo import git_output, sha256


def save_records(path, rows):
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with path.with_suffix(".csv").open("x", newline="") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def run_study(binary, output, protocol_path, filter_path, development_smoke=False, make_plots=True):
    binary, output = Path(binary).resolve(), Path(output).resolve()
    protocol_path, filter_path = Path(protocol_path).resolve(), Path(filter_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    if sha256(filter_path) != protocol["frozen_filter_sha256"]:
        raise ValueError("filter configuration differs from the frozen protocol")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("build the C++ replay executable first")
    seeds = [protocol["development_seed"]] if development_smoke else protocol["evaluation_seeds"]
    role = "development" if development_smoke else "held_out_m2"
    # Use a new folder so earlier results stay available.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    shutil.copyfile(filter_path, output / "filter.conf")
    status = git_output("status", "--porcelain")
    source_paths = [path for folder in ("include", "src", "apps", "tools", "tests", "scenarios", "docs")
                    for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in (".hpp", ".cpp", ".py", ".json", ".conf", ".md")]
    source_paths += [ROOT / name for name in ("CMakeLists.txt", "requirements-lock.txt", "README.md", "AGENTS.md")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path), "filter_sha256": sha256(filter_path), "seed_role": role, "seeds": seeds,
        "git_commit": git_output("rev-parse", "HEAD"), "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None, "git_status": status,
        "binary": str(binary), "binary_sha256": sha256(binary),
        "build_info": (binary.parent / "build_info.txt").read_text() if (binary.parent / "build_info.txt").exists() else "unavailable",
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("cmake", "numpy", "matplotlib", "pytest")},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(source_paths)}, "commands": [],
        "interpretation": ("Development seed already used in M1; software validation, not held-out evidence." if development_smoke else
                           "Seeds were reserved from M1; no tuning on these recordings. Once inspected they are no longer untouched evaluation data."),
    }
    manifest_path = output / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    save_manifest()
    metrics, episodes, series = [], [], {}
    try:
        with (output / "replay.log").open("x") as log:
            for seed in seeds:
                seed_dir = output / f"seed-{seed}"
                annotations = generate_suite(seed_dir / "inputs", protocol, seed, role)
                for scenario in protocol["scenarios"]:
                    name = scenario["id"]
                    for method in ("dr", "ekf", "gated"):
                        command = [str(binary), str(seed_dir / f"inputs/{name}.csv"), str(output / "filter.conf"), method,
                                   str(seed_dir / f"{name}-{method}.csv"), str(seed_dir / f"{name}-{method}-gps.csv")]
                        manifest["commands"].append(command)
                        save_manifest()
                        replay = subprocess.run(command, text=True, capture_output=True)
                        log.write(f"seed={seed} scenario={name} method={method}\n" + replay.stdout + replay.stderr)
                        log.flush()
                        replay.check_returncode()
                        result, recovery, data = evaluate_case(seed_dir, scenario, method, protocol, seed, annotations["scenarios"][name])
                        metrics.append(result)
                        episodes.extend(recovery)
                        series[(seed, name, method)] = data
                    print(f"Finished seed {seed}: {name}", flush=True)
        summaries, paired = summarize(metrics, episodes, protocol, seeds)
        save_records(output / "metrics", metrics)
        save_records(output / "recovery", episodes)
        save_records(output / "summary", summaries)
        save_records(output / "paired-comparisons", paired)
        if make_plots:
            plot_study(output, protocol, seeds, metrics, series)
        manifest["status"] = "complete"
        manifest["replays"] = len(metrics)
        manifest["artifacts_sha256"] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                                        if path.is_file() and path != manifest_path}
        print("\nScenario         Method  Median RMSE [min, max] m    Fault rejected/considered  Settling not observed")
        for row in summaries:
            print(f"{row['scenario']:16} {row['method']:6} {row['position_rmse_median_m']:7.3f} "
                  f"[{row['position_rmse_min_m']:.3f}, {row['position_rmse_max_m']:.3f}]"
                  f"     {row['fault_gps_rejected']}/{row['fault_gps_considered']}"
                  f"                    {row['settling_not_observed']}")
        print(f"\nResults: {output}\nSeed role: {role}; frozen protocol: {manifest['protocol_sha256']}")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save_manifest()
    return metrics, episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "scenarios/gps_faults.json")
    parser.add_argument("--filter", type=Path, default=ROOT / "scenarios/filter.conf")
    parser.add_argument("--development-smoke", action="store_true", help="use only the previously used development seed")
    args = parser.parse_args()
    run_study(args.binary, args.output, args.protocol, args.filter, args.development_smoke)
