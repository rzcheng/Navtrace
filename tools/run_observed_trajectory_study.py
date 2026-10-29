#!/usr/bin/env python3
"""One frozen observation-conditioned path experiment (issue #10).

Only `study` can open the reserved streams, after the reviewed validation gate.
Numerical imports are lazy; the child is a single serial worker. Wrapper
aggregation also counts toward the final resource assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

if __package__:
    from . import run_trajectory_mixture_study as old
else:
    import run_trajectory_mixture_study as old

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "results/observed-trajectory-study-checks-v1"
DEMO = ROOT / "results/observed-trajectory-study-demo-v1"
STUDY = ROOT / "results/observed-trajectory-study-v1"
GROUPS = (96001, 96002, 96003, 96004)
PREFIXES = (0, 10000, 250000, 500000, 750000, 1000000)
COUNT = 2048
MIB = 1024 * 1024
SECONDS, MEMORY, OUTPUT, RESERVE = 120, 512*MIB, 64*MIB, 4*MIB
EXACT = 1e-10
FROZEN = {
    "docs/observed_trajectory_proposal.md": "37c607341e4b4ceff7bbe550a5bd9b4778c31904985f248baaf934ae91c52491",
    "docs/observed_trajectory_design.md": "7ca53abdf9b7618f56359eaf865f3682b15960d6d7b7e6a2c9c333ee06000ef8",
    "tools/observed_trajectory.py": "1c8d4700b49ba81dfa935127c893b57acdf600d53018eea6a270be33d1ad7899",
    "tests/test_observed_trajectory.py": "35f928b9ed7106b9a100fcd6c8237633c719da8f0ca9d58652a19d0a44fc425a",
    "tools/run_trajectory_mixture_study.py": "60538d532b3bd01b416c5d7f4678d5d18ed4fdae6e89320f27da2c5f146dc674",
    "docs/trajectory_mixture_proposal.md": "a9c23b2b40aaa220c030450e35431fd8a29c08de2f2b6b8bf2bde47b014b70ae",
}
REQUIRED_SOURCE = set(FROZEN) | {"tools/run_observed_trajectory_study.py",
    "tests/test_observed_trajectory_study.py", "tools/trajectory_mixture.py"}
REQUIRED_CHECKS = {"focused", "full-pytest", "ctest", "demo"}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def plain(value):
    """Convert arrays only after the worker imports NumPy."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_bytes(path, data, *, metadata=False):
    path = Path(path)
    reserve = 0 if metadata else RESERVE
    # Charge both old and pending versions during an atomic replacement.
    if old.output_bytes([STUDY]) + len(data) + reserve > OUTPUT:
        raise old.OutputLimit("study output allowance exhausted before artifact write")
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value, *, metadata=False):
    write_bytes(path, (json.dumps(plain(value), indent=2, sort_keys=True,
                                allow_nan=False)+"\n").encode(), metadata=metadata)


def save_inventory(inventory):
    write_json(STUDY / "inventory.json", inventory, metadata=True)


def planned_inventory():
    return {"schema_version": 1, "component_count": COUNT,
        "groups": list(GROUPS), "prefixes_us": list(PREFIXES),
        "global_resource": "unresolved", "global_failures": [],
        "decision": "defer", "aggregates": {},
        "interpretation": "four independent algorithm groups at each prefix; not navigation trials",
        "rows": [{"group": group, "prefix_index": index, "timestamp_us": timestamp,
            "execution": "not-started", "assessment": "inconclusive",
            "reason": "not entered", "metrics": {}, "comparison_status": "unavailable",
            "comparison_reason": "not entered", "exact": {}, "snapshot": None,
            "mean": None, "covariance": None, "dimensions": None,
            "resources": None, "last_valid_timestamp_us": None,
            "position_interpretation": ("independent full-six oracle" if index < 2 else
                "descriptive position and position/latent correlations; no independent oracle")}
            for group in GROUPS for index, timestamp in enumerate(PREFIXES)]}


def precedence(statuses):
    return "fail" if "fail" in statuses else ("inconclusive" if "inconclusive" in statuses else "pass")


def aggregate_inventory(inventory):
    """Missing groups never become a smaller substitute ensemble."""
    import numpy as np
    aggregates = {}
    for timestamp in PREFIXES:
        rows = [row for row in inventory["rows"] if row["timestamp_us"] == timestamp]
        by_group = {row["group"]: row for row in rows}
        expected_support = list(range(6)) if timestamp in PREFIXES[:2] else [2, 3, 4, 5]
        names = [(f"mean_{j}", .1) for j in expected_support]
        names += [(f"cov_{j}_{k}", .2) for i, j in enumerate(expected_support)
                  for k in expected_support[i:]]
        complete = len(rows) == 4 and set(by_group) == set(GROUPS) and all(
            row["execution"] == "completed" and row["comparison_status"] == "available"
            for row in rows)
        entry = {"support": expected_support, "metrics": {}, "assessment": "inconclusive",
                 "available": complete, "reason": None if complete else "required row or comparison unavailable"}
        for name, epsilon in names:
            values = [by_group[group]["metrics"].get(name, {}).get("value")
                      if group in by_group and complete else None for group in GROUPS]
            entry["metrics"][name] = old.assess_four(values, epsilon)
        if complete:
            matrices = np.asarray([by_group[group]["whitened_covariance_error"] for group in GROUPS])
            average = matrices.mean(axis=0)
            individual = [float(np.linalg.norm(matrix, 2)) for matrix in matrices]
            average_norm = float(np.linalg.norm(average, 2))
            entry["covariance_spectral"] = {"individual": individual,
                "average_matrix": average.tolist(), "average_norm": average_norm,
                "assessment": "pass" if max(individual) <= .5 and average_norm <= .3 else "fail"}
            entry["assessment"] = precedence([item["assessment"] for item in entry["metrics"].values()]
                                              +[entry["covariance_spectral"]["assessment"]])
        else:
            entry["covariance_spectral"] = {"assessment": "inconclusive", "reason": entry["reason"]}
        for row in rows:
            individual = row.get("individual_assessment", "inconclusive")
            row["assessment"] = precedence([individual, entry["assessment"]])
        aggregates[str(timestamp)] = entry
    inventory["aggregates"] = aggregates
    assessments = [row["assessment"] for row in inventory["rows"]]
    assessments += [entry["assessment"] for entry in aggregates.values()]
    if inventory["global_failures"] or inventory["global_resource"] == "failed":
        assessments.append("fail")
    elif inventory["global_resource"] != "completed":
        assessments.append("inconclusive")
    if len(inventory["rows"]) != 24:
        assessments.append("inconclusive")
    result = precedence(assessments)
    inventory["decision"] = {"fail": "no-go", "inconclusive": "defer", "pass": "pass"}[result]
    return inventory


def stop_inventory(inventory, reason, *, unresolved=False):
    inventory["global_resource"] = "unresolved" if unresolved else "failed"
    for row in inventory["rows"]:
        if row["execution"] == "running":
            row.update(execution="unresolved" if unresolved else "failed", reason=reason,
                       individual_assessment="inconclusive" if unresolved else "fail")
        elif row["execution"] == "not-started":
            row["reason"] = reason
    aggregate_inventory(inventory)


def frozen_inputs():
    mean, covariance, diffusion, events = old.fixture("O")
    return {"mean": mean, "covariance": covariance, "diffusion": diffusion,
        "events": events, "checkpoint_us": 0, "max_step_s": .01, "max_gap_s": 2.,
        "prefixes_us": PREFIXES, "J": [[.2, 0, 0], [.05, .15, 0], [.04, -.02, .25]],
        "La": [[.3, 0, 0], [.02, .08, 0], [-.01, -.03, .04]],
        "K": [[.4, .2, -.3], [-.2, .5, .1], [.3, -.4, .2]]}


def independent_oracles(inputs):
    # Do not call old.oracle_inventory: L and H are outside this experiment.
    result = old.latent_oracle()
    result[0] = {"mean": inputs["mean"].copy(), "covariance": inputs["covariance"].copy(),
                 "support": list(range(6)), "log_evidence": 0.}
    first_step = old.observed_one_step_oracle()
    if abs(first_step["log_evidence"]-result[10000]["log_evidence"]) > EXACT:
        raise ArithmeticError("independent one-step and latent evidence targets disagree")
    result[10000] = first_step
    if set(result) != set(PREFIXES):
        raise AssertionError("independent oracle prefix set differs from frozen schedule")
    return result


def coordinate_digest(coordinates):
    import numpy as np
    return hashlib.sha256(np.asarray(coordinates, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def covariance_check(covariance):
    import numpy as np
    value = np.asarray(covariance)
    if not np.isfinite(value).all():
        raise ArithmeticError("nonfinite retained covariance")
    scale = np.maximum(1., np.max(np.abs(value), axis=(-2, -1)))
    asymmetry = np.max(np.abs(value-value.swapaxes(-1, -2)), axis=(-2, -1))
    eigenvalues = np.linalg.eigvalsh(.5*value+.5*value.swapaxes(-1, -2))
    if np.any(asymmetry > 1e-10*scale) or np.any(eigenvalues[..., 0] < -1e-10*scale):
        raise ArithmeticError("retained covariance violates unchanged symmetry/PSD policy")
    return {"maximum_asymmetry": float(np.max(asymmetry)),
            "minimum_eigenvalue": float(np.min(eigenvalues[..., 0]))}


def endpoint_audit(snapshot):
    """Reconstruct all correlations using only the retained endpoint mixture."""
    import numpy as np
    weights = np.exp(snapshot["logweights"])
    components = np.empty((len(weights), 6))
    components[:, [2, 4, 5]] = snapshot["angular"]
    components[:, [0, 1, 3]] = snapshot["conditional_mean"]
    mean = np.einsum('n,ni->i', weights, components, optimize=False)
    centered = components-mean
    covariance = np.einsum('n,ni,nj->ij', weights, centered, centered, optimize=False)
    covariance[np.ix_([0, 1, 3], [0, 1, 3])] += np.einsum(
        'n,nij->ij', weights, snapshot["conditional_covariance"], optimize=False)
    return {"mean_max_abs": float(np.max(np.abs(mean-snapshot["mean"]))),
            "covariance_max_abs": float(np.max(np.abs(covariance-snapshot["covariance"])))}


def retained_snapshot(report, terms):
    return {"angular": report["angular_paths"][:, -1, :],
        "conditional_mean": report["conditional_mean"],
        "conditional_covariance": report["conditional_covariance"],
        "logweights": report["logweights"], "mean": report["mean"],
        "covariance": report["covariance"], "log_prior": terms["log_prior"],
        "log_likelihood": terms["log_likelihood"], "log_proposal": terms["log_proposal"],
        "importance_residual": terms["log_importance"]-report["log_evidence"]}


def inspect_row(row, snapshot, report, oracle):
    import numpy as np
    if any(not np.isfinite(value).all() for value in snapshot.values()):
        raise ArithmeticError("nonfinite retained row")
    checks = row["exact"]
    checks["conditional_covariance"] = covariance_check(snapshot["conditional_covariance"])
    checks["represented_covariance"] = covariance_check(snapshot["covariance"])
    checks["importance_max_abs"] = float(np.max(np.abs(snapshot["importance_residual"])))
    checks["evidence_abs_error"] = abs(float(report["log_evidence"])-float(oracle["log_evidence"]))
    weights = np.exp(snapshot["logweights"])
    checks["weight_max_abs_error"] = float(np.max(np.abs(weights-1/len(weights))))
    checks["weight_sum_abs_error"] = abs(float(weights.sum())-1)
    checks["endpoint_audit"] = endpoint_audit(snapshot)
    row["structural_weights"] = {"ess": float(1/np.sum(weights*weights)),
        "maximum_weight": float(np.max(weights)), "interpretation": "structural uniform weights, not an accuracy/calibration test"}
    row["log_evidence"] = float(report["log_evidence"])
    row["mean"], row["covariance"] = snapshot["mean"].tolist(), snapshot["covariance"].tolist()
    exact_values = [checks[key] for key in ("importance_max_abs", "evidence_abs_error",
                                           "weight_max_abs_error", "weight_sum_abs_error")]
    exact_values += list(checks["endpoint_audit"].values())
    exact_failure = any(value > EXACT for value in exact_values)
    try:
        metrics, dp = old.comparison(report, oracle)
    except old.OracleUnresolved as exc:
        row.update(comparison_status="unavailable", comparison_reason=str(exc),
                   individual_assessment="fail" if exact_failure else "inconclusive")
        if exact_failure:
            raise ArithmeticError("retained exact identity failed")
        return
    row.update(metrics=metrics, whitened_covariance_error=dp.tolist(),
               covariance_spectral_norm=float(np.linalg.norm(dp, 2)),
               comparison_status="available", comparison_reason=None,
               oracle_support=oracle["support"], oracle_mean=plain(oracle["mean"]),
               oracle_covariance=plain(oracle["covariance"]))
    failed = exact_failure or row["covariance_spectral_norm"] > .5 or any(
        abs(metric["value"]) > 2*metric["epsilon"] for metric in metrics.values())
    row["individual_assessment"] = "fail" if failed else "pass"
    if exact_failure:
        raise ArithmeticError("retained exact identity failed")


def current_worker_resources(started, row_started):
    return {"worker_elapsed_s": time.monotonic()-started,
            "row_elapsed_s": time.monotonic()-row_started,
            "worker_lifetime_high_water_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "retained_bytes": old.output_bytes([STUDY]),
            "note": "wrapper-plus-worker assessment is in resources.json; Darwin RSS units are bytes"}


def evaluate_row(row, inputs, oracles, candidate, persist):
    """Keep large batch locals scoped to this row, including exception exits."""
    import numpy as np
    events = [event for event in inputs["events"] if event[0] <= row["timestamp_us"]]
    events = [candidate.Observation(*event) for event in events]
    phase_started = time.monotonic()
    posterior = candidate.condition_prefix(inputs["mean"], inputs["covariance"],
        inputs["diffusion"], events, checkpoint_us=0, max_step_s=.01, max_gap_s=2.)
    row["conditioning_s"] = time.monotonic()-phase_started
    row["dimensions"] = {"angular": posterior.angular_dimension,
        "residual": posterior.residual_dimension, "steps": len(posterior.step_sizes_s),
        "events": len(events), "rows": COUNT}
    row["coordinate_layout"] = plain(posterior.coordinate_layout)
    row["step_sizes_s"] = plain(posterior.step_sizes_s)
    row["phase"] = posterior.phase
    stream_key = [row["group"], 9, row["prefix_index"]]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(stream_key)))
    row["rng"] = {"generator": "Generator", "bit_generator": "PCG64",
        "seed_sequence_entropy": stream_key, "initial_state": generator.bit_generator.state}
    phase_started = time.monotonic()
    coordinates = generator.standard_normal((COUNT, posterior.angular_dimension))
    row["rng"]["final_state"] = generator.bit_generator.state
    row["coordinate_sha256"] = coordinate_digest(coordinates)
    row["coordinate_hash_format"] = "C-order little-endian IEEE-754 float64 bytes"
    persist()  # Retain stream provenance before expensive materialization.
    if not np.isfinite(coordinates).all():
        raise ArithmeticError("nonfinite supplied standard coordinates")
    report = posterior.evaluate(coordinates)
    row["materialization_s"] = time.monotonic()-phase_started
    terms = posterior.density_terms(report["angular_coordinates"])
    snapshot = retained_snapshot(report, terms)
    # Retain first, so an exact/accuracy failure cannot erase its evidence.
    buffer = io.BytesIO()
    np.savez(buffer, **snapshot)
    filename = f"group-{row['group']}-prefix-{row['timestamp_us']}.npz"
    write_bytes(STUDY / filename, buffer.getvalue())
    row["snapshot"] = filename
    row["snapshot_sha256"] = old.digest(STUDY / filename)
    if math.isfinite(float(report["log_evidence"])):
        row["log_evidence"] = float(report["log_evidence"])
    persist()  # A resource stop during checks still retains the saved snapshot link.
    phase_started = time.monotonic()
    inspect_row(row, snapshot, report, oracles[row["timestamp_us"]])
    row["checks_s"] = time.monotonic()-phase_started
    row.update(execution="completed", reason=None,
               last_valid_timestamp_us=posterior.timestamp_us)
    # Prefix batches are independent; release all full paths and dense maps.
    del coordinates, report, terms, snapshot, posterior, buffer, generator


def worker():
    if os.environ.get("NAVTRACE_OBSERVED_WORKER_TOKEN") != str(os.getppid()):
        raise RuntimeError("worker requires its owning one-launch wrapper")
    import numpy as np
    if __package__:
        from . import observed_trajectory as candidate
        from .trajectory_mixture import InvalidInput, NumericalFailure, UnresolvedInference
    else:
        import observed_trajectory as candidate
        from trajectory_mixture import InvalidInput, NumericalFailure, UnresolvedInference
    started = time.monotonic()
    inventory = json.loads((STUDY / "inventory.json").read_text())
    if inventory != planned_inventory():
        raise RuntimeError("worker requires complete pristine planned inventory")
    inputs = frozen_inputs()
    write_json(STUDY / "inputs.json", inputs)
    oracles = independent_oracles(inputs)
    write_json(STUDY / "oracles.json", oracles)
    write_json(STUDY / "runtime.json", {"numpy": np.__version__, "python": sys.version,
        "python_executable": sys.executable, "uname": tuple(os.uname()),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ[name] for name in THREAD_ENV}})
    for row in inventory["rows"]:
        row_started = time.monotonic()
        row.update(execution="running", reason=None, started_unix_s=time.time(),
                   resources=current_worker_resources(started, row_started))
        save_inventory(inventory)
        try:
            evaluate_row(row, inputs, oracles, candidate, lambda: save_inventory(inventory))
        except old.OutputLimit:
            raise
        except InvalidInput as exc:
            row.update(execution="invalid", individual_assessment="fail", reason=str(exc))
        except (UnresolvedInference, old.OracleUnresolved) as exc:
            row.update(execution="unresolved", individual_assessment="inconclusive", reason=str(exc))
        except (NumericalFailure, ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            row.update(execution="failed", individual_assessment="fail",
                       reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            row.update(execution="failed", individual_assessment="fail",
                       reason=f"unexpected {type(exc).__name__}: {exc}")
            inventory["global_failures"].append(row["reason"])
            for later in inventory["rows"]:
                if later["execution"] == "not-started":
                    later["reason"] = "worker stopped after unexpected exception"
            row["resources"] = current_worker_resources(started, row_started)
            save_inventory(inventory)
            break
        row["resources"] = current_worker_resources(started, row_started)
        aggregate_inventory(inventory)
        save_inventory(inventory)
    aggregate_inventory(inventory)
    save_inventory(inventory)
    write_json(STUDY / "comparisons.json", {"rows": inventory["rows"], "aggregates": inventory["aggregates"]})
    plot_results(inventory)
    return 0


def plot_results(inventory):
    """Two fixed table-based figures; no RNG or candidate evaluation."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(PREFIXES))
    labels = [f"{timestamp/1e6:g}" for timestamp in PREFIXES]
    by_key = {(row["group"], row["timestamp_us"]): row for row in inventory["rows"]}
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for group, color in zip(GROUPS, colors):
        for metric, ax in zip(("mean", "cov", "spectral"), axes):
            values = []
            for timestamp in PREFIXES:
                row = by_key.get((group, timestamp), {})
                if row.get("comparison_status") != "available":
                    values.append(float("nan"))
                elif metric == "spectral":
                    values.append(row["covariance_spectral_norm"])
                else:
                    selected = [abs(value["value"]) for name, value in row["metrics"].items()
                                if name.startswith(metric+"_")]
                    values.append(max(selected))
            ax.plot(x, values, "o-", label=str(group), color=color)
            missing = ~np.isfinite(values)
            ax.scatter(x[missing], np.zeros(sum(missing)), marker="x", color=color)
    for ax, limit, label in zip(axes, (.1, .2, .5),
            ("max |whitened mean error|", "max |whitened covariance entry|", "covariance error spectral norm")):
        ax.axhline(limit, color="black", ls="--", label="scalar epsilon" if limit != .5 else "individual limit")
        ax.axvspan(-.25, 1.25, color="grey", alpha=.12)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("Independent support: full six states at 0 and 0.01 s; latent four states thereafter\n"
                      "All groups; later position accuracy is not scored")
    axes[-1].set_xticks(x, [label+"\n"+inventory["aggregates"][str(timestamp)]["assessment"]
                                  for label, timestamp in zip(labels, PREFIXES)])
    axes[-1].set_xlabel("received prefix (s); four-group assessment; x marks unavailable")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    write_bytes(STUDY / "supported-errors.png", buffer.getvalue())

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for group, color in zip(GROUPS, colors):
        rows = [by_key.get((group, timestamp), {}) for timestamp in PREFIXES]
        series = ([row.get("resources", {}).get("row_elapsed_s", float("nan"))
                   if row.get("resources") else float("nan") for row in rows],
                  [row.get("dimensions", {}).get("angular", float("nan"))
                   if row.get("dimensions") else float("nan") for row in rows],
                  [row.get("structural_weights", {}).get("ess", float("nan")) for row in rows],
                  [row.get("structural_weights", {}).get("maximum_weight", float("nan")) for row in rows])
        for ax, values in zip(axes, series):
            ax.plot(x, values, "o-", color=color, label=str(group))
            missing = ~np.isfinite(values)
            ax.scatter(x[missing], np.zeros(sum(missing)), marker="x", color=color)
    for ax, label in zip(axes, ("row elapsed (s)", "angular primitive dimension", "structural ESS", "maximum weight")):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title("Prefix cost and algebraically uniform weights\n"
                      "Weights are structural checks, not approximation or calibration evidence")
    axes[2].axhline(COUNT, color="black", ls="--")
    axes[3].axhline(1/COUNT, color="black", ls="--")
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("received prefix (s); x marks unavailable")
    execution_codes = {"completed": "C", "failed": "F", "invalid": "V", "unresolved": "U",
                       "not-started": "N", "running": "R"}
    assessment_codes = {"pass": "P", "fail": "F", "inconclusive": "I"}
    cells = [[execution_codes.get(by_key.get((group, timestamp), {}).get("execution"), "?")+"/"+
              assessment_codes.get(by_key.get((group, timestamp), {}).get("assessment"), "?")
              for timestamp in PREFIXES] for group in GROUPS]
    table_ax = fig.add_axes([.13, .035, .79, .10])
    table_ax.axis("off")
    table = table_ax.table(cellText=cells, rowLabels=[str(group) for group in GROUPS],
                           colLabels=labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table_ax.set_title("Execution C=completed F=failed V=invalid U=unresolved N=not-started; "
                       "assessment P=pass F=fail I=inconclusive", fontsize=7)
    fig.subplots_adjust(left=.13, right=.97, top=.94, bottom=.21, hspace=.24)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    write_bytes(STUDY / "cost-and-weights.png", buffer.getvalue())


def validation_roots():
    return [CHECKS, DEMO, ROOT / "build", ROOT / ".pytest_cache", ROOT / "__pycache__",
            ROOT / "tools", ROOT / "tests"]


def verify_gate(gate):
    if gate.get("decision") != "eligible" or not gate.get("independent_review_accepted"):
        raise RuntimeError("eligible independently reviewed prelaunch gate required")
    if not gate.get("final_required_checks_passed") or not gate.get("clean_committed_source"):
        raise RuntimeError("final checks and clean committed source attestation required")
    sources = gate.get("source_sha256", {})
    if not REQUIRED_SOURCE.issubset(sources):
        raise RuntimeError("required frozen and new source hashes are missing")
    for name, expected in FROZEN.items():
        if sources[name] != expected:
            raise RuntimeError(f"gate disagrees with frozen source: {name}")
    for name, expected in sources.items():
        path = ROOT / name
        if Path(name).is_absolute() or ".." in Path(name).parts or old.digest(path) != expected:
            raise RuntimeError(f"reviewed source changed: {name}")
    initialization = CHECKS / "initialization.json"
    if old.digest(initialization) != gate.get("initialization_sha256"):
        raise RuntimeError("initialization attestation changed")
    initial = json.loads(initialization.read_text())
    if not initial.get("roots_absent") or not initial.get("groups_unopened"):
        raise RuntimeError("fresh roots and unopened groups attestation required")
    ledger_path = CHECKS / "validation.json"
    if old.digest(ledger_path) != gate.get("validation_ledger_sha256"):
        raise RuntimeError("reviewed validation ledger changed")
    ledger = json.loads(ledger_path.read_text())
    attempts = ledger.get("attempts", [])
    if not attempts or not ledger.get("monitor_review_accepted"):
        raise RuntimeError("reviewed validation attempts required")
    if any(attempt["status"] == "running" for attempt in attempts):
        raise RuntimeError("unfinished validation attempt")
    if sum(attempt["charged_s"] for attempt in attempts) > 900:
        raise RuntimeError("aggregate validation time exhausted")
    for attempt in attempts:
        if attempt.get("assessed_peak_bytes") is None or attempt["assessed_peak_bytes"] > 2*1024*MIB:
            raise RuntimeError("validation memory accounting unavailable or exceeded")
    if old.output_bytes(validation_roots()) > 256*MIB:
        raise RuntimeError("retained validation output allowance exceeded")
    required = gate.get("required_attempts", {})
    if set(required) != REQUIRED_CHECKS:
        raise RuntimeError("required final check mapping incomplete")
    for label, path in required.items():
        matches = [attempt for attempt in attempts if attempt.get("attempt_path") == path]
        if len(matches) != 1 or matches[0].get("label") != label or matches[0]["status"] != "completed":
            raise RuntimeError(f"final required check did not complete: {label}")
    cache = CHECKS / "matplotlib-cache"
    expected_cache = gate.get("plotting_cache_sha256", {})
    if not expected_cache or {p.name for p in cache.glob("fontlist-*.json")} != set(expected_cache):
        raise RuntimeError("reviewed font cache missing or changed")
    for name, expected in expected_cache.items():
        if old.digest(cache / name) != expected:
            raise RuntimeError("reviewed font cache changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty or commit != gate.get("git_commit"):
        raise RuntimeError("study requires the clean reviewed source commit")


def artifact_hashes():
    return {str(path.relative_to(STUDY)): old.digest(path)
            for path in sorted(STUDY.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def final_resource_check(result, inventory, started):
    try:
        high = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        worker_high = result.get("worker_job_high_water_bytes")
        if high <= 0 or worker_high is None or worker_high <= 0:
            raise old.AccountingError("required final wrapper/worker high water unavailable")
        result["wrapper_high_water_bytes"] = high
        result["assessed_peak_bytes"] = max(result.get("sampled_aggregate_peak_bytes", 0), high+worker_high)
    except Exception as exc:
        if result["status"] != "failed":
            result.update(status="unresolved", reason=f"final memory accounting: {exc}")
    elapsed, retained = time.monotonic()-started, old.output_bytes([STUDY])
    result.update(elapsed_through_metadata_flush_s=elapsed, retained_bytes=retained)
    if elapsed > SECONDS or retained > OUTPUT or (result.get("assessed_peak_bytes") or 0) > MEMORY:
        result.update(status="failed", reason="final flush exceeded study allowance")
    if result["status"] != "completed":
        stop_inventory(inventory, result["reason"], unresolved=result["status"] == "unresolved")
    return result


def emergency_json(path, value):
    """Preserve terminal failure metadata even after an observed cap violation."""
    old.save_json(path, plain(value))


def finalize(result, inventory, manifest, started):
    """Check time, retained bytes and RSS after terminal artifact writes too."""
    def persist_terminal():
        aggregate_inventory(inventory)
        emergency_json(STUDY / "resources.json", result)
        emergency_json(STUDY / "inventory.json", inventory)
        manifest.update(decision=inventory["decision"], resource_status=result["status"],
                        artifact_sha256=artifact_hashes(),
                        ended_before_final_metadata_unix_s=time.time(),
                        receipt_note="terminal stdout includes accounting after the last metadata flush")
        emergency_json(STUDY / "manifest.json", manifest)

    final_resource_check(result, inventory, started)
    result["charged_through_final_flush_s"] = time.monotonic()-started+1
    result["final_flush_reserve_s"] = 1
    if result["charged_through_final_flush_s"] > SECONDS:
        result.update(status="failed", reason="insufficient time for reserved final flush")
        stop_inventory(inventory, result["reason"])
    persist_terminal()
    # First completed flush is included in both memory and time accounting.
    final_resource_check(result, inventory, started)
    if time.monotonic()-started > result["charged_through_final_flush_s"]:
        result.update(status="failed", reason="final flush exceeded conservative charge")
        stop_inventory(inventory, result["reason"])
    persist_terminal()
    # Probe AFTER the last planned hash/write, not just before metadata allocation.
    previous_status = result["status"]
    final_resource_check(result, inventory, started)
    if time.monotonic()-started > result["charged_through_final_flush_s"]:
        result.update(status="failed", reason="terminal flush exceeded conservative charge")
        stop_inventory(inventory, result["reason"])
    if result["status"] != previous_status:
        persist_terminal()  # Retain a newly discovered late failure/unresolved gate.
        final_resource_check(result, inventory, started)
        # Once a failure is recorded, any still larger terminal cost remains no-go.
        if manifest["resource_status"] != result["status"]:
            persist_terminal()
            final_resource_check(result, inventory, started)
    print(json.dumps({"decision": inventory["decision"], "resource_status": result["status"],
        "elapsed_through_final_flush_s": time.monotonic()-started,
        "retained_bytes": old.output_bytes([STUDY]),
        "wrapper_high_water_bytes": result.get("wrapper_high_water_bytes"),
        "assessed_peak_bytes": result.get("assessed_peak_bytes"),
        "charged_through_final_flush_s": result["charged_through_final_flush_s"]}))
    return 0 if result["status"] == "completed" else 1


def launch(started):
    try:
        gate = json.loads((CHECKS / "gate.json").read_text())
        verify_gate(gate)
    except Exception as exc:
        inventory = planned_inventory()
        for row in inventory["rows"]:
            row["reason"] = f"prelaunch gate blocked: {type(exc).__name__}: {exc}"
        # Does not consume or create the study root; no stream can have opened.
        old.save_json(CHECKS / f"blocked-inventory-{time.time_ns()}.json", inventory)
        raise
    STUDY.mkdir()  # Exclusive one-launch guard; never reuse or choose a suffix.
    inventory = planned_inventory()
    manifest = {}
    try:
        save_inventory(inventory)  # First artifact: all24 rows before any other preparation.
        manifest = {"git_commit": gate["git_commit"], "source_sha256": gate["source_sha256"],
            "frozen_sha256": FROZEN, "component_count": COUNT, "groups": GROUPS, "prefixes_us": PREFIXES,
            "stream": "Generator(PCG64(SeedSequence([group_key, 9, prefix_index])))",
            "draw": "one C-order float64 standard_normal((2048, dU)); no Z draw",
            "coordinate_hash_format": "C-order little-endian IEEE-754 float64 bytes",
            "limits": {"seconds_through_final_flush": SECONDS, "assessed_memory_bytes": MEMORY,
                       "output_bytes": OUTPUT, "metadata_reserve_bytes": RESERVE},
            "started_unix_s": time.time()-(time.monotonic()-started),
            "gate_sha256": old.digest(CHECKS / "gate.json"),
            "validation_ledger_sha256": old.digest(CHECKS / "validation.json"),
            "initialization_sha256": old.digest(CHECKS / "initialization.json"),
            "python_executable_sha256": old.digest(Path(sys.executable).resolve()),
            "python": sys.version, "platform": sys.platform,
            "resource_limitations": ["100 ms polling may miss short-lived peaks; assessed RSS is the larger of live aggregate and sum of wrapper/worker lifetime highs", "Darwin ru_maxrss and libproc RSS are bytes", "no worker descendants allowed"],
            "omitted_by_protocol": ["dense prefix posterior matrices", "full angular histories", "supplied standard-coordinate arrays"]}
        write_json(STUDY / "manifest.json", manifest, metadata=True)
        cache = STUDY / "matplotlib-cache"
        cache.mkdir()
        for name in gate["plotting_cache_sha256"]:
            write_bytes(cache / name, (CHECKS / "matplotlib-cache" / name).read_bytes())
        os.environ["MPLCONFIGDIR"] = str(cache)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["NAVTRACE_OBSERVED_WORKER_TOKEN"] = str(os.getpid())
        result = old.monitor_command([sys.executable, "-W", "error::RuntimeWarning", str(Path(__file__).resolve()), "worker"],
            STUDY / "worker.log", [STUDY], SECONDS, MEMORY, OUTPUT, study=True, started=started)
        inventory = json.loads((STUDY / "inventory.json").read_text())
        if result["status"] == "completed":
            inventory["global_resource"] = "completed"
            if any(row["execution"] == "running" for row in inventory["rows"]):
                inventory["global_failures"].append("worker exited leaving a running row")
        else:
            stop_inventory(inventory, result["reason"], unresolved=result["status"] == "unresolved")
    except Exception as exc:
        reason = f"wrapper failure: {type(exc).__name__}: {exc}"
        path = STUDY / "inventory.json"
        if path.exists():
            inventory = json.loads(path.read_text())
        stop_inventory(inventory, reason)
        result = {"status": "failed", "reason": reason, "worker_job_high_water_bytes": None,
                  "sampled_aggregate_peak_bytes": 0}
        emergency_json(STUDY / "wrapper-failure.json", {"reason": reason})
    return finalize(result, inventory, manifest, started)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("study", "worker"))
    args = parser.parse_args()
    return worker() if args.action == "worker" else launch(started)


if __name__ == "__main__":
    raise SystemExit(main())
