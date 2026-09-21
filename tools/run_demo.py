"""Generate paired data, execute C++, evaluate and plot, recording provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_scenario import generate, validate_settings
from tools.evaluate import evaluate_run
from tools.plot_results import plot_results


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_demo(binary, output, settings_path, filter_path, overwrite=False):
    binary, output = Path(binary).resolve(), Path(output).resolve()
    settings = json.loads(Path(settings_path).read_text())
    validate_settings(settings)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"C++ executable missing or not executable: {binary}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite {output}; choose a new path or use --overwrite")
        if not (output / ".navtrace-run").is_file():
            raise ValueError("--overwrite only replaces a marked NavTrace results directory")
        if output == ROOT or output in binary.parents or output in Path(filter_path).resolve().parents or output in Path(settings_path).resolve().parents:
            raise ValueError("cannot overwrite project, executable, or configuration inputs")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / ".navtrace-run").write_text("NavTrace generated results\n")
    generate(output / "inputs", settings)
    shutil.copyfile(filter_path, output / "filter.conf")
    status = git_output("status", "--porcelain")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": sys.version, "python_executable": sys.executable,
        "dependencies": {package: importlib.metadata.version(package) for package in ("cmake", "numpy", "matplotlib", "pytest")},
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "build_info": (binary.parent / "build_info.txt").read_text() if (binary.parent / "build_info.txt").exists() else "unavailable",
        "git_commit": git_output("rev-parse", "HEAD"), "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None, "git_status": status,
        "seed": settings["seed"], "seed_role": "development",
        "evaluation": "exact gyro timestamps, last output after simultaneous updates; outage [start,end), end error at last sample before returning GPS",
        "position_rmse_definition": "sqrt(mean((estimated_x-true_x)^2 + (estimated_y-true_y)^2))",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for directory in ("include", "src", "apps", "tools", "tests", "scenarios")
                          for path in sorted((ROOT / directory).rglob("*")) if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".cpp", ".hpp", ".py", ".json", ".conf")},
        "commands": [],
    }
    manifest_path = output / "manifest.json"
    for name in ("CMakeLists.txt", "requirements.txt", "requirements-lock.txt"):
        manifest["source_sha256"][name] = sha256(ROOT / name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (output / "replay.log").open("w") as log:
            for scenario in ("nominal", "dropout"):
                for method in ("dr", "ekf", "gated"):
                    command = [str(binary), str(output / f"inputs/{scenario}.csv"), str(output / "filter.conf"), method,
                               str(output / f"{scenario}-{method}.csv"), str(output / f"{scenario}-{method}-gps.csv")]
                    manifest["commands"].append(command)
                    completed = subprocess.run(command, check=True, text=True, capture_output=True)
                    log.write(completed.stdout + completed.stderr)
                    print(completed.stdout.strip(), flush=True)
        metrics, errors = evaluate_run(output)
        plot_results(output, errors, settings["outage_start_s"], settings["outage_end_s"])
        manifest["status"] = "complete"
        manifest["artifacts_sha256"] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                                        if path.is_file() and path != manifest_path}
        print("\nScenario Method   RMSE(m) WindowRMSE(m) End-window(m) Final(m) HeadingRMSE(deg) GPS A/R/I")
        for row in metrics:
            print(f"{row['scenario']:8} {row['method']:6} {row['position_rmse_m']:8.3f} "
                  f"{row['outage_position_rmse_m']:13.3f} {row['outage_end_position_error_m']:13.3f} "
                  f"{row['final_position_error_m']:8.3f} {row['heading_rmse_rad'] * 180 / 3.141592653589793:16.3f} "
                  f"{row['gps_accepted']}/{row['gps_rejected']}/{row['gps_ignored']}")
        print(f"\nArtifacts: {output}\nPlots: {output / 'trajectories.png'}\n       {output / 'errors.png'}")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "scenarios/development.json")
    parser.add_argument("--filter", type=Path, default=ROOT / "scenarios/filter.conf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_demo(args.binary, args.output, args.settings, args.filter, args.overwrite)
