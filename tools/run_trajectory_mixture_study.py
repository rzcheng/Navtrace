#!/usr/bin/env python3
"""Frozen synthetic study and its bounded command monitor (issue #7).

The monitor uses only the standard library. Numerical imports are lazy so the
single-thread environment applies to both wrapper and worker before BLAS loads.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "results/trajectory-mixture-checks-v1"
DEMO = ROOT / "results/trajectory-mixture-demo-v1"
STUDY = ROOT / "results/trajectory-mixture-study-v1"
PROPOSAL_HASH = "a9c23b2b40aaa220c030450e35431fd8a29c08de2f2b6b8bf2bde47b014b70ae"
DESIGN_HASH = "644809e2f91ad5df3c3876b60625595981d87633ba2ddf0475be483daa7f6596"
MIB = 1024 * 1024
SEEDS = (95001, 95002, 95003, 95004)
REPORTS = {"L": (0, 100000, 200000, 300000), "H": (0, 100000),
           "O": (0, 10000, 250000, 500000, 750000, 1000000)}


class AccountingError(RuntimeError):
    pass


class OutputLimit(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_bytes(roots):
    return sum(p.stat().st_size for root in roots if root.exists()
               for p in root.rglob("*") if p.is_file())



def validation_roots():
    # Conservatively include entire source/test trees, including every bytecode
    # cache they can create, plus build products and unique retained test roots.
    return [CHECKS, DEMO, ROOT / "build", ROOT / ".pytest_cache",
            ROOT / "__pycache__", ROOT / "tools", ROOT / "tests"]


def save_json(path, value, roots=None, cap=None, reserve=0):
    """Check retained bytes before each write; never remove earlier artifacts."""
    data = (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    path = Path(path)
    if cap is not None:
        if output_bytes(roots) + len(data) + reserve > cap:
            raise OutputLimit("retained output allowance exceeded before JSON write")
    temporary = path.with_name(path.name + f".pending-{time.time_ns()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class DarwinAccounting:
    """libproc live RSS, in bytes, without creating accounting subprocesses.

    Apple proc_info.h defines six uint64 fields then twelve int32 fields in
    proc_taskinfo. proc_pidinfo returns bytes; proc_listchildpids returns a PID
    COUNT, while its buffer-size argument is bytes. getrusage(2) on Darwin
    defines ru_maxrss in bytes. wait4 usage is a job high-water observation,
    not an exact simultaneous descendant-tree peak.
    """
    class TaskInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "virtual_size", "resident_size", "total_user", "total_system",
            "threads_user", "threads_system")] + [(name, ctypes.c_int32) for name in (
            "policy", "faults", "pageins", "cow_faults", "messages_sent",
            "messages_received", "syscalls_mach", "syscalls_unix", "csw",
            "threadnum", "numrunning", "priority")]

    def __init__(self):
        if sys.platform != "darwin":
            raise AccountingError("this reviewed monitor supports Darwin only")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                         ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_pidinfo.restype = ctypes.c_int
        self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_listchildpids.restype = ctypes.c_int
        self.exit_races = 0

    def children(self, pid):
        capacity = 64
        while capacity <= 65536:
            buffer = (ctypes.c_int * capacity)()
            ctypes.set_errno(0)
            count = self.lib.proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
            error = ctypes.get_errno()
            if count < 0 or (count == 0 and error):
                if error == 3:  # ESRCH: a descendant exited between enumeration calls.
                    self.exit_races += 1
                    return []
                raise AccountingError(f"child enumeration failed pid={pid} errno={error}")
            if count < capacity:
                return [int(buffer[i]) for i in range(count) if buffer[i] > 0]
            capacity *= 2
        raise AccountingError("child enumeration did not fit bounded PID buffer")

    def rss(self, pid, may_exit=False):
        info = self.TaskInfo()
        ctypes.set_errno(0)
        size = self.lib.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if size != ctypes.sizeof(info):
            error = ctypes.get_errno()
            if may_exit and error == 3:
                self.exit_races += 1
                return 0
            raise AccountingError(f"RSS query failed pid={pid} bytes={size} errno={error}")
        return int(info.resident_size)

    def tree(self, root_pid):
        pending, seen = [root_pid], set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            pending.extend(self.children(pid))
        return seen


def monitor_command(command, log_path, roots, seconds, memory_cap, output_cap,
                    study=False, started=None):
    """Poll every 100 ms; retain exit-time high water and all failure evidence.

    os.wait4 owns reaping. Popen.poll/wait must not consume the worker's rusage.
    Validation can have descendants; their short-lived peaks can escape polling.
    Study worker descendants are forbidden, making the sum of lifetime peaks a
    conservative two-process bound rather than a claimed simultaneous peak.
    """
    started = time.monotonic() if started is None else started
    record = {"command": list(command), "poll_seconds": .1, "status": "running",
              "limits": {"seconds": seconds, "memory_bytes": memory_cap,
                         "output_bytes": output_cap}, "samples": [],
              "limitations": ["100 ms polling can miss short-lived descendant peaks; "
                               "job high water is not a simultaneous tree peak"]}
    accounting = None
    proc = None
    usage = None
    reason = None
    peak = 0
    known_pids = set()
    try:
        accounting = DarwinAccounting()
        with Path(log_path).open("xb") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while True:
                pid, status, child_usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    usage = child_usage
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    # An exited root must not leave a live descendant outside accounting.
                    survivors = []
                    for known_pid in known_pids - {proc.pid}:
                        try:
                            if os.getpgid(known_pid) == proc.pid:
                                survivors.append(known_pid)
                        except ProcessLookupError:
                            pass
                    if survivors:
                        reason = ("failed", "worker exited with observed live descendants")
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                try:
                    pids = accounting.tree(proc.pid)
                    known_pids.update(pids)
                    if study and pids != {proc.pid}:
                        reason = ("failed", "study worker created a forbidden descendant")
                    rss = accounting.rss(os.getpid()) + sum(
                        accounting.rss(p, may_exit=True) for p in pids)
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    retained = output_bytes(roots)
                    record["samples"].append({"elapsed_s": elapsed, "rss_bytes": rss,
                                              "retained_bytes": retained, "processes": len(pids) + 1})
                    if elapsed > seconds:
                        reason = ("failed", "elapsed allowance exceeded")
                    elif peak > memory_cap:
                        reason = ("failed", "observed aggregate RSS allowance exceeded")
                    elif retained > output_cap - 4 * MIB:
                        reason = ("failed", "retained output metadata reserve exhausted")
                except AccountingError as exc:
                    record.setdefault("accounting_errors", []).append(str(exc))
                    reason = reason or ("unresolved", str(exc))
                if reason:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    _, status, usage = os.wait4(proc.pid, 0)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(.1)
    except Exception as exc:
        record.setdefault("accounting_errors", []).append(str(exc))
        reason = reason or ("unresolved", f"monitor exception: {type(exc).__name__}: {exc}")
        if proc is not None and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
    wrapper_high = worker_high = assessed = retained = None
    elapsed = time.monotonic() - started
    try:
        wrapper_high = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        worker_high = None if usage is None else int(usage.ru_maxrss)
        if worker_high is None or wrapper_high <= 0 or worker_high <= 0:
            raise AccountingError("required OS high-water measurement unavailable")
        assessed = max(peak, wrapper_high + worker_high) if study else max(peak, wrapper_high, worker_high)
        retained = output_bytes(roots)
        if elapsed > seconds or retained > output_cap or assessed > memory_cap:
            reason = ("failed", "exit-time resource allowance exceeded")
    except Exception as exc:
        reason = reason or ("unresolved", f"final accounting failed: {exc}")
    returncode = None if proc is None else proc.returncode
    record.update(status=reason[0] if reason else ("completed" if returncode == 0 else "failed"),
                  reason=reason[1] if reason else (None if returncode == 0 else "command returned nonzero"),
                  returncode=returncode, elapsed_s=elapsed,
                  wrapper_high_water_bytes=wrapper_high, worker_job_high_water_bytes=worker_high,
                  sampled_aggregate_peak_bytes=peak, assessed_peak_bytes=assessed,
                  retained_bytes=retained, exit_sampling_races=None if accounting is None else accounting.exit_races)
    return record


def validation_command(label, command):
    """Durably debit started commands; failed finalization never erases attempts.

    A one-second conservative charge covers the final metadata flush. A crash
    leaving 'running' blocks further commands until independently reconciled.
    Terminal metadata is retained even after an output overrun; that overrun is
    explicitly a failed resource gate, never hidden by deleting command output.
    """
    os.environ["MPLCONFIGDIR"] = str(CHECKS / "matplotlib-cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    ledger_path = CHECKS / "validation.json"
    ledger = json.loads(ledger_path.read_text())
    if not ledger.get("monitor_review_accepted"):
        raise RuntimeError("independent monitor review is required")
    if any(a["status"] == "running" for a in ledger["attempts"]):
        raise RuntimeError("unfinished attempt requires accounting reconciliation")
    spent = sum(a["charged_s"] for a in ledger["attempts"])
    if spent + 1 >= 900 or output_bytes(validation_roots()) >= 252 * MIB:
        raise RuntimeError("validation allowance exhausted; no further command")
    attempt_dir = CHECKS / f"attempt-{len(ledger['attempts']) + 1:02d}-{label}"
    attempt_dir.mkdir()
    start = time.monotonic()
    result = {"status": "running", "label": label, "command": list(command),
              "attempt_path": str(attempt_dir.relative_to(ROOT)), "started_unix_s": time.time(),
              "monitor_sha256": digest(Path(__file__))}
    ledger["attempts"].append(result)
    save_json(ledger_path, ledger, validation_roots(), 256 * MIB)
    result.update(monitor_command(command, attempt_dir / "output.log", validation_roots(),
                                  900 - spent - 1, 2 * 1024 * MIB, 256 * MIB, started=start))
    # Keep terminal evidence even when the child has already overrun its output cap.
    result["elapsed_s"] = time.monotonic() - start
    result["charged_s"] = result["elapsed_s"] + 1
    save_json(attempt_dir / "resources.json", result)
    save_json(ledger_path, ledger)
    after_flush = time.monotonic() - start
    retained = output_bytes(validation_roots())
    result["elapsed_through_primary_metadata_flush_s"] = after_flush
    result["conservative_final_flush_charge_s"] = 1
    result["retained_bytes"] = retained
    if after_flush > result["charged_s"] or spent + after_flush > 900 or retained > 256 * MIB:
        result.update(status="failed", reason="final metadata flush exceeded accounting allowance",
                      elapsed_s=after_flush, charged_s=after_flush + 1, retained_bytes=retained)
        save_json(attempt_dir / "resources.json", result)
        save_json(ledger_path, ledger)
    save_json(attempt_dir / "resources.json", result)
    save_json(ledger_path, ledger)
    final_elapsed = time.monotonic() - start
    final_bytes = output_bytes(validation_roots())
    if final_elapsed > result["charged_s"] or spent + final_elapsed > 900 or final_bytes > 256*MIB:
        result.update(status="failed", reason="final flush exceeded conservative accounting or observed cap",
                      elapsed_s=final_elapsed, charged_s=final_elapsed + 1, retained_bytes=final_bytes)
        save_json(attempt_dir / "resources.json", result)
        save_json(ledger_path, ledger)
    print(json.dumps({"label": label, "status": result["status"], "reason": result["reason"],
                      "elapsed_through_final_flush_s": time.monotonic() - start,
                      "charged_s": result["charged_s"], "assessed_peak_bytes": result["assessed_peak_bytes"],
                      "retained_bytes": output_bytes(validation_roots())}))
    return 0 if result["status"] == "completed" else 1


def fixture(case):
    """Assemble frozen inputs; never pass these factors to the candidate."""
    import numpy as np
    mean = np.array([.5, -.25, .2, 1.5, .1, -.02])
    covariance = np.zeros((6, 6))
    angular, linear = [2, 4, 5], [0, 1, 3]
    j = np.array([[.2, 0, 0], [.05, .15, 0], [.04, -.02, .25]])
    la = np.array([[.3, 0, 0], [.02, .08, 0], [-.01, -.03, .04]])
    gain = np.array([[.4, .2, -.3], [-.2, .5, .1], [.3, -.4, .2]])
    if case == "H":
        mean = np.array([0., 0., 0., 2., 0., 0.])
        covariance[2, 2] = .4
        diffusion = np.zeros(3)
        events = []
    elif case == "L":
        mean[4:] = 0
        covariance[np.ix_(linear, linear)] = j @ j.T
        diffusion = np.array([.04, 0, 0])
        events = [(t, sensor, value, variance) for t, zg, zw in (
            (100000, .03, 1.4), (200000, -.02, 1.6), (300000, .01, 1.45))
            for sensor, value, variance in (("gyro", zg, .02**2), ("wheel", zw, .1**2))]
    elif case == "O":
        a = la @ la.T
        b = gain @ a
        covariance[np.ix_(angular, angular)] = a
        covariance[np.ix_(linear, angular)] = b
        covariance[np.ix_(angular, linear)] = b.T
        covariance[np.ix_(linear, linear)] = gain @ a @ gain.T + j @ j.T
        diffusion = np.array([.04, .0004, .0001])
        events = [(10000, "gyro", .09, .02**2), (10000, "wheel", 1.4, .1**2)]
        for index in range(1, 21):
            events.append((50000 * index, "gyro", .08 + .01 * (-1)**index, .02**2))
            if index % 2 == 0:
                events.append((50000 * index, "wheel", 1.5 + .1 * (-1)**(index // 2), .1**2))
    else:
        raise ValueError(case)
    return mean, covariance, diffusion, events



class OracleUnresolved(RuntimeError):
    pass


def product(left, right):
    """Direct tensor contraction avoids the observed macOS BLAS status flags.

    The 33D L batch map emitted divide/overflow/invalid warnings despite an
    exactly finite product equal to this contraction (recorded validation #2).
    No warning is suppressed and no mathematical operation or tolerance changes.
    """
    import numpy as np
    signature = {(1, 1): 'i,i->', (2, 1): 'ij,j->i',
                 (1, 2): 'i,ij->j', (2, 2): 'ij,jk->ik'}
    return np.einsum(signature[(np.ndim(left), np.ndim(right))], left, right, optimize=False)


def resolved_oracle_factor(covariance):
    """Independent fixed-unit support gate before any oracle inversion."""
    import numpy as np
    p = np.asarray(covariance, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or not np.isfinite(p).all():
        raise ValueError("oracle covariance is not a finite square matrix")
    tolerance = 1e-10 * max(1., float(np.max(np.abs(p))))
    if np.max(np.abs(p-p.T)) > tolerance:
        raise ValueError("oracle covariance is materially asymmetric")
    p = (p+p.T)/2
    eigenvalues = np.linalg.eigvalsh(p)
    if eigenvalues[0] < -tolerance:
        raise ValueError("oracle covariance is materially indefinite")
    if eigenvalues[0] <= 1e-12 * max(1., float(eigenvalues[-1])):
        raise OracleUnresolved("oracle covariance support is insufficiently resolved")
    return np.linalg.cholesky(p)

def condition_gaussian(mean, covariance, observation_map, values, noise):
    """Independent batch conditioning and normalized log observation density."""
    import numpy as np
    h = np.asarray(observation_map)
    innovation = np.asarray(values) - product(h, mean)
    s = product(product(h, covariance), h.T) + noise
    g = np.linalg.cholesky(s)
    solved = np.linalg.solve(s, product(h, covariance))
    updated_mean = mean + product(solved.T, innovation)
    updated_covariance = covariance - product(product(covariance, h.T), solved)
    standardized = np.linalg.solve(g, innovation)
    log_density = -.5 * (len(innovation) * math.log(2 * math.pi)
                        + 2 * np.log(np.diag(g)).sum() + standardized @ standardized)
    return updated_mean, (updated_covariance + updated_covariance.T) / 2, float(log_density)


def linear_batch_oracle():
    """L: condition independent initial ell and 30 independent speed increments.

    This is a joint Gaussian linear map, not the candidate's recursive update.
    Gyro observations have fixed mean zero and contribute only their density.
    """
    import numpy as np
    mean, covariance, diffusion, events = fixture("L")
    linear = [0, 1, 3]
    base_mean = np.zeros(33)
    base_mean[:3] = mean[linear]
    base_covariance = np.diag([0.] * 3 + [diffusion[0] * .01] * 30)
    base_covariance[:3, :3] = covariance[np.ix_(linear, linear)]
    mapping = np.zeros((3, 33))
    mapping[:, :3] = np.eye(3)
    maps = {0: mapping.copy()}
    f = np.eye(3)
    f[0, 2], f[1, 2] = .01 * math.cos(.2), .01 * math.sin(.2)
    for step in range(1, 31):
        mapping = product(f, mapping)
        mapping[2, 2 + step] += 1
        if step % 10 == 0:
            maps[step * 10000] = mapping.copy()
    result = {}
    for timestamp, mapping in maps.items():
        received = [event for event in events if event[0] <= timestamp]
        wheel = [event for event in received if event[1] == "wheel"]
        if wheel:
            h = np.stack([maps[event[0]][2] for event in wheel])
            m, p, log_density = condition_gaussian(base_mean, base_covariance, h,
                                                   [event[2] for event in wheel],
                                                   np.eye(len(wheel)) * .1**2)
        else:
            m, p, log_density = base_mean, base_covariance, 0.
        for _, sensor, value, variance in received:
            if sensor == "gyro":
                log_density += -.5 * (math.log(2 * math.pi * variance) + value**2 / variance)
        full_mean = mean.copy()
        full_mean[linear] = product(mapping, m)
        full_covariance = np.zeros((6, 6))
        full_covariance[np.ix_(linear, linear)] = product(product(mapping, p), mapping.T)
        result[timestamp] = {"mean": full_mean, "covariance": full_covariance,
                             "support": linear, "log_evidence": log_density}
    return result


def heading_oracle(timestamp):
    import numpy as np
    mean, covariance, _, _ = fixture("H")
    support = [2]
    if timestamp:
        distance, variance = .2, .4
        mean[0] = distance * math.exp(-variance / 2)
        covariance[0, 0] = distance**2 * (1 - math.exp(-variance))**2 / 2
        covariance[1, 1] = distance**2 * (1 - math.exp(-2 * variance)) / 2
        covariance[1, 2] = covariance[2, 1] = distance * variance * math.exp(-variance / 2)
        support = [0, 1, 2]
    return {"mean": mean, "covariance": covariance, "support": support, "log_evidence": 0.}


def heading_empirical(angles, timestamp):
    """Scalar centered sums independent of mixture reconstruction helpers."""
    import numpy as np
    values = [[(.2 * math.cos(theta) if timestamp else 0.),
               (.2 * math.sin(theta) if timestamp else 0.), theta, 2., 0., 0.]
              for theta in angles]
    mean = np.array([math.fsum(row[j] for row in values) / len(values) for j in range(6)])
    covariance = np.array([[math.fsum((row[j] - mean[j]) * (row[k] - mean[k])
                                     for row in values) / len(values)
                            for k in range(6)] for j in range(6)])
    return mean, covariance, np.asarray(values)


def observed_one_step_oracle():
    """O: condition the full 9D Gaussian BEFORE the nonlinear endpoint map."""
    import numpy as np
    mean, covariance, diffusion, _ = fixture("O")
    m = np.r_[mean, [0., 0., 0.]]
    p = np.zeros((9, 9))
    p[:6, :6], p[6:, 6:] = covariance, np.diag(diffusion * .01)
    observation = np.zeros((2, 9))
    observation[0, [4, 5, 7, 8]] = 1
    observation[1, [3, 6]] = 1
    m, p, log_density = condition_gaussian(m, p, observation, [.09, 1.4],
                                           np.diag([.02**2, .1**2]))
    f = np.zeros((6, 9))
    f[:, :6] = np.eye(6)
    f[2, 4] = .01
    f[3, 6] = f[4, 7] = f[5, 8] = 1
    my, py = f @ m, f @ p @ f.T
    theta, speed = 2, 3
    c = p[speed, theta]
    psi = lambda k: np.exp(1j * k * m[theta] - .5 * k*k * p[theta, theta])
    first = (m[speed] + 1j * c) * psi(1)
    second = (p[speed, speed] + (m[speed] + 2j * c)**2) * psi(2)
    speed_second = p[speed, speed] + m[speed]**2
    md = np.array([first.real, first.imag])
    raw_dd = np.array([[(speed_second + second.real)/2, second.imag/2],
                       [second.imag/2, (speed_second - second.real)/2]])
    cross = (f @ p[:, speed] + (my + 1j * (f @ p[:, theta]))
             * (m[speed] + 1j*c)) * psi(1)
    cyd = np.column_stack((cross.real, cross.imag)) - np.outer(my, md)
    ep = np.eye(6)[:, :2]
    full_mean = my + .01 * ep @ md
    full_covariance = (py + .01 * (cyd @ ep.T + ep @ cyd.T)
                       + .01**2 * ep @ (raw_dd - np.outer(md, md)) @ ep.T)
    return {"mean": full_mean, "covariance": full_covariance, "support": list(range(6)),
            "log_evidence": log_density, "conditioned_mean9": m, "conditioned_covariance9": p}


def latent_oracle():
    """Independent exact 4D linear Gaussian filter; no later position oracle."""
    import numpy as np
    mean, covariance, diffusion, events = fixture("O")
    indices = [2, 3, 4, 5]
    m, p = mean[indices], covariance[np.ix_(indices, indices)]
    result = {0: {"mean": m.copy(), "covariance": p.copy(), "support": indices,
                  "log_evidence": 0.}}
    timestamp, log_evidence = 0, 0.
    for index, (time_us, sensor, value, variance) in enumerate(events):
        gap = time_us - timestamp
        while gap:
            step_us = min(gap, 10000)
            h = step_us / 1e6
            f = np.eye(4)
            f[0, 2] = h
            m = f @ m
            p = f @ p @ f.T + np.diag([0., *(diffusion * h)])
            gap -= step_us
        hobs = np.array([[0., 0., 1., 1.]]) if sensor == "gyro" else np.array([[0., 1., 0., 0.]])
        m, p, log_density = condition_gaussian(m, p, hobs, [value], np.array([[variance]]))
        log_evidence += log_density
        timestamp = time_us
        if timestamp in REPORTS["O"] and (index + 1 == len(events) or events[index+1][0] != timestamp):
            result[timestamp] = {"mean": m.copy(), "covariance": p.copy(), "support": indices,
                                 "log_evidence": log_evidence}
    return result


def oracle_inventory():
    mean, covariance, _, _ = fixture("O")
    observed = latent_oracle()
    observed[0] = {"mean": mean, "covariance": covariance, "support": list(range(6)),
                   "log_evidence": 0.}
    observed[10000] = observed_one_step_oracle()
    return {"L": linear_batch_oracle(), "H": {t: heading_oracle(t) for t in REPORTS["H"]},
            "O": observed}


def comparison(snapshot, oracle):
    import numpy as np
    indices = oracle["support"]
    expected_mean, expected_cov = oracle["mean"], oracle["covariance"]
    if len(expected_mean) == 6:
        expected_mean = expected_mean[indices]
        expected_cov = expected_cov[np.ix_(indices, indices)]
    g = resolved_oracle_factor(expected_cov)
    dm = np.linalg.solve(g, snapshot["mean"][indices] - expected_mean)
    left = np.linalg.solve(g, snapshot["covariance"][np.ix_(indices, indices)])
    dp = np.linalg.solve(g, left.T).T - np.eye(len(indices))
    metrics = {f"mean_{j}": {"value": float(dm[i]), "epsilon": .1} for i, j in enumerate(indices)}
    for i, j in enumerate(indices):
        for k in range(i, len(indices)):
            metrics[f"cov_{j}_{indices[k]}"] = {"value": float(dp[i, k]), "epsilon": .2}
    if len(snapshot["mean"]) != 6:
        raise AssertionError("candidate reports must retain full six-state moments")
    return metrics, dp


def assess_four(values, epsilon):
    import numpy as np
    if len(values) != 4 or any(value is None for value in values):
        return {"assessment": "inconclusive", "reason": "one or more independent groups unavailable"}
    values = np.asarray(values, dtype=float)
    average = float(values.mean())
    se = float(np.sqrt(np.sum((values - average)**2) / 12))
    precise = 4 * se <= epsilon / 2
    if np.any(np.abs(values) > 2 * epsilon) or abs(average) - 4 * se > epsilon:
        status = "fail"
    elif precise and abs(average) + 4 * se <= epsilon:
        status = "pass"
    else:
        status = "inconclusive"
    return {"assessment": status, "mean": average, "se": se, "four_se": 4*se,
            "epsilon": epsilon, "precision_pass": precise, "individual": values.tolist(),
            "interpretation": "descriptive four-standard-error rule, not calibrated coverage"}


def planned_inventory():
    rows = []
    for case, seed, branch in [("L", 0, "L")] + [
        ("H", seed, branch) for seed in SEEDS for branch in ("H1", "H8")] + [
            ("O", seed, "O") for seed in SEEDS]:
        reports = {}
        for timestamp in REPORTS[case]:
            support = ([0, 1, 3] if case == "L" else ([2] if timestamp == 0 else [0, 1, 2]))
            if case == "O":
                support = list(range(6)) if timestamp <= 10000 else [2, 3, 4, 5]
            quantities = [f"mean_{j}" for j in support]
            quantities += [f"cov_{j}_{k}" for i, j in enumerate(support) for k in support[i:]]
            if case == "H" and timestamp:
                quantities += ["relative_var_x", "relative_var_y"]
            if case == "O":
                quantities += ["log_evidence"]
            reports[str(timestamp)] = {"execution": "not-started", "assessment": "inconclusive",
                "support": support, "metrics": {q: {"status": "not-started", "value": None,
                "reason": "row has not reached this report"} for q in quantities}}
        rows.append({"id": f"{case}/{seed}/{branch}", "case": case, "seed": seed, "branch": branch,
                     "execution": "not-started", "assessment": "inconclusive", "reports": reports,
                     "last_valid_timestamp_us": None, "last_valid_phase": None, "events": [],
                     "exact_checks": [], "reason": "not started", "elapsed_s": None})
    return {"schema": 1, "rows": rows, "aggregates": {}, "global_resource": "not-started",
            "decision": "defer", "independent_stochastic_groups": 4,
            "interpretation": "fixed-input algorithm randomizations, not navigation trials"}


def json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def artifact_bytes(path, data, root=None):
    root = STUDY if root is None else Path(root)
    if output_bytes([root]) + len(data) + 4 * MIB > 64 * MIB:
        raise OutputLimit("artifact would consume reserved metadata/failure allowance")
    Path(path).write_bytes(data)


def write_snapshot(path, snapshot):
    import io
    import numpy as np
    buffer = io.BytesIO()
    keys = ("angular", "conditional_mean", "conditional_covariance", "logweights", "ids",
            "mean", "covariance")
    np.savez(buffer, **{key: snapshot[key] for key in keys})
    artifact_bytes(path, buffer.getvalue())


def exact_check(row, name, actual, expected, tolerance=1e-10):
    import numpy as np
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    row["exact_checks"].append({"name": name, "max_absolute_error": error,
                                "tolerance": tolerance, "passed": error <= tolerance})
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(f"{name}: error {error} exceeds {tolerance}")


def retain_report(row, timestamp, snapshot, oracle, initial_angles=None):
    import numpy as np
    report = row["reports"][str(timestamp)]
    filename = f"{row['case']}-{row['seed']}-{row['branch']}-{timestamp}.npz"
    write_snapshot(STUDY / filename, snapshot)
    metrics, dp = comparison(snapshot, oracle)
    if row["case"] == "H" and timestamp:
        for index, name in enumerate(("relative_var_x", "relative_var_y")):
            metrics[name] = {"value": float(snapshot["covariance"][index, index]
                                            / oracle["covariance"][index, index] - 1), "epsilon": .2}
    if row["case"] == "O":
        metrics["log_evidence"] = {"value": float(snapshot["log_evidence"] - oracle["log_evidence"]),
                                   "epsilon": .5}
    for metric in metrics.values():
        metric["status"] = "available"
    report.update(execution="completed", metrics=metrics, snapshot=filename,
                  mean=json_value(snapshot["mean"]), covariance=json_value(snapshot["covariance"]),
                  covariance_whitened_error=json_value(dp), covariance_spectral_norm=float(np.linalg.norm(dp, 2)),
                  clock_kind="primitive elapsed time; timestamp_us is checkpoint clock" if row["case"] == "H" else "received event timestamp",
                  log_evidence=float(snapshot["log_evidence"]),
                  underflow_count=int(snapshot["underflow_count"]),
                  diagnostics=json_value({k: snapshot[k] for k in (
                      "ess", "max_weight", "logweight_range", "underflow_count", "observation_count",
                      "substeps", "timestamp_us", "phase", "primitive_elapsed_s")}))
    row["last_valid_timestamp_us"], row["last_valid_phase"] = timestamp, snapshot["phase"]
    if row["case"] in ("L", "H"):
        exact_check(row, f"equal weights at {timestamp}", snapshot["logweights"], -math.log(2048))
        exact_check(row, f"ESS at {timestamp}", snapshot["ess"] / 2048, 1.)
    if row["case"] == "L":
        exact_check(row, f"L mean {timestamp}", snapshot["mean"], oracle["mean"])
        exact_check(row, f"L covariance {timestamp}", snapshot["covariance"], oracle["covariance"])
        exact_check(row, f"L whitened covariance {timestamp}", dp, 0.)
        exact_check(row, f"L log evidence {timestamp}", snapshot["log_evidence"], oracle["log_evidence"])
        exact_check(row, f"L component means {timestamp}", snapshot["conditional_mean"],
                    oracle["mean"][[0, 1, 3]])
        exact_check(row, f"L component covariances {timestamp}", snapshot["conditional_covariance"],
                    oracle["covariance"][np.ix_([0, 1, 3], [0, 1, 3])])
        exact_check(row, f"L observation count {timestamp}", snapshot["observation_count"], 2 * (timestamp // 100000), 0)
        exact_check(row, f"L substeps {timestamp}", snapshot["substeps"], timestamp // 10000, 0)
    if row["case"] == "H":
        empirical_mean, empirical_covariance, states = heading_empirical(initial_angles, timestamp)
        exact_check(row, f"H empirical mean {timestamp}", snapshot["mean"], empirical_mean)
        exact_check(row, f"H empirical covariance {timestamp}", snapshot["covariance"], empirical_covariance)
        exact_check(row, f"H angular states {timestamp}", snapshot["angular"], states[:, [2, 4, 5]])
        exact_check(row, f"H conditional means {timestamp}", snapshot["conditional_mean"], states[:, [0, 1, 3]])
        exact_check(row, f"H deterministic residual {timestamp}", snapshot["conditional_covariance"], 0., 0)


def mark_unavailable(row, reason):
    for report in row["reports"].values():
        if report["execution"] == "not-started":
            report["reason"] = reason
            for metric in report["metrics"].values():
                metric["reason"] = reason


def mark_resource_stop(inventory, reason):
    """Keep every planned row, including the unstarted partner of an H row."""
    inventory["global_resource"] = "failed"
    inventory["resource_reason"] = reason
    for row in inventory["rows"]:
        if row["execution"] == "running":
            row.update(execution="failed", assessment="fail", reason=reason)
        if row["execution"] in ("failed", "not-started"):
            mark_unavailable(row, reason)
    inventory["decision"] = "no-go"


def aggregate_inventory(inventory):
    import numpy as np
    all_assessments = []
    for row in inventory["rows"]:
        if row["execution"] in ("failed", "invalid"):
            row["assessment"] = "fail"
        elif row["execution"] != "completed":
            row["assessment"] = "inconclusive"
        else:
            underflow = any(r.get("underflow_count", 0) for r in row["reports"].values())
            underflow |= any(e.get("underflow_count", 0) for e in row["events"])
            row["assessment"] = "inconclusive" if underflow else "pass"
            if row["case"] == "O" and any(e["ess"] < .1 * 2048 or e["max_weight"] > .05 for e in row["events"]):
                row["assessment"] = "fail"
                row["concentration_requirement"] = "fail"
            elif row["case"] == "O":
                row["concentration_requirement"] = "pass"
            if row["case"] != "L":
                for report in row["reports"].values():
                    if report.get("covariance_spectral_norm", 0) > .5:
                        row["assessment"] = "fail"
                    if any(abs(metric["value"]) > 2 * metric["epsilon"]
                           for metric in report["metrics"].values() if metric["value"] is not None):
                        row["assessment"] = "fail"
        # Failure evidence survives a later incomplete or unresolved run.
        if row["case"] == "O" and any(e["ess"] < .1*2048 or e["max_weight"] > .05 for e in row["events"]):
            row["assessment"] = "fail"
            row["concentration_requirement"] = "fail"
        for report in row["reports"].values():
            if report["execution"] == "completed":
                report["assessment"] = "inconclusive" if report.get("underflow_count", 0) else "pass"
                if report.get("covariance_spectral_norm", 0) > .5 or any(
                    abs(m["value"]) > 2*m["epsilon"] for m in report["metrics"].values()
                    if m["value"] is not None):
                    report["assessment"] = "fail"
                    row["assessment"] = "fail"
        all_assessments.append(row["assessment"])
    aggregates = {}
    for case in ("H", "O"):
        rows = [r for r in inventory["rows"] if r["case"] == case and r["branch"] != "H8"]
        required = [r for r in inventory["rows"] if r["case"] == case]
        complete_groups = all(r["execution"] == "completed" for r in required)
        for timestamp in REPORTS[case]:
            reports = [r["reports"][str(timestamp)] for r in rows]
            key = f"{case}/{timestamp}"
            group = {"metrics": {}, "independent_groups": 4,
                     "support": reports[0]["support"], "covariance_guard": "inconclusive"}
            for name in reports[0]["metrics"]:
                values = [report["metrics"][name]["value"] for report in reports]
                epsilon = reports[0]["metrics"][name].get("epsilon", .5 if name == "log_evidence" else (
                    .1 if name.startswith("mean_") else .2))
                assessed = assess_four(values if complete_groups else [None]*4, epsilon)
                group["metrics"][name] = assessed
                all_assessments.append(assessed["assessment"])
            if complete_groups and all("covariance_whitened_error" in report for report in reports):
                errors = np.asarray([r["covariance_whitened_error"] for r in reports])
                norm = float(np.linalg.norm(errors.mean(axis=0), 2))
                group["mean_covariance_error_norm"] = norm
                group["covariance_guard"] = "pass" if norm <= .3 else "fail"
                all_assessments.append(group["covariance_guard"])
            else:
                all_assessments.append("inconclusive")
            aggregates[key] = group
    # A per-seed row must not read "pass" when its required four-group accuracy
    # assessment is unresolved. Keep individual evidence and group evidence apart.
    priority = {"pass": 0, "inconclusive": 1, "fail": 2}
    for row in inventory["rows"]:
        row["individual_assessment"] = row["assessment"]
        for timestamp, report in row["reports"].items():
            report["individual_assessment"] = report["assessment"]
            if row["case"] in ("H", "O"):
                group = aggregates[f"{row['case']}/{timestamp}"]
                statuses = [m["assessment"] for m in group["metrics"].values()]
                statuses.append(group["covariance_guard"])
                assessment = max(statuses, key=priority.get)
                report["group_assessment"] = assessment
                report["group_assessment_reason"] = "required four-group rule; H1 only supplies independent H errors"
                report["assessment"] = max([report["assessment"], assessment], key=priority.get)
                row["assessment"] = max([row["assessment"], assessment], key=priority.get)
    inventory["aggregates"] = aggregates
    if inventory["global_resource"] == "failed" or "fail" in all_assessments:
        inventory["decision"] = "no-go"
    elif inventory["global_resource"] != "completed" or "inconclusive" in all_assessments:
        inventory["decision"] = "defer"
    else:
        inventory["decision"] = "pass"
    return inventory


def new_candidate(case, seed):
    import numpy as np
    from trajectory_mixture import TrajectoryMixture
    mean, covariance, diffusion, _ = fixture(case)
    case_id = {"L": 1, "H": 2, "O": 3}[case]
    streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261018, case_id, seed, phase])))
               for phase in (0, 1)]
    return TrajectoryMixture.initialize(mean, covariance, 2048, *streams, diffusion,
                                        checkpoint_us=0, max_step_s=.01, max_gap_s=2.)


def study_worker():
    import numpy as np
    from trajectory_mixture import InvalidInput, NumericalFailure, Observation, UnresolvedInference
    inventory = json.loads((STUDY / "inventory.json").read_text())
    oracles = oracle_inventory()
    save_json(STUDY / "oracles.json", json_value(oracles), [STUDY], 64 * MIB, 4 * MIB)
    paired = {}
    paired_endpoints = {}
    for row in inventory["rows"]:
        start = time.monotonic()
        row.update(execution="running", reason=None)
        save_json(STUDY / "inventory.json", inventory, [STUDY], 64 * MIB)
        try:
            case, seed = row["case"], row["seed"]
            if case == "H":
                if row["branch"] == "H1":
                    candidate = new_candidate(case, seed)
                    paired[seed] = candidate.clone()
                else:
                    candidate = paired[seed]
                angles = candidate.angular[:, 0].copy()
                retain_report(row, 0, candidate.snapshot(), oracles[case][0], angles)
                steps = 1 if row["branch"] == "H1" else 8
                for _ in range(steps):
                    candidate.predict_step(.1 / steps)
                endpoint = candidate.snapshot()
                retain_report(row, 100000, endpoint, oracles[case][100000], angles)
                if row["branch"] == "H1":
                    paired_endpoints[seed] = endpoint
                elif seed in paired_endpoints:
                    for key in ("angular", "conditional_mean", "conditional_covariance", "logweights", "mean", "covariance", "ids"):
                        exact_check(row, f"paired H1/H8 {key}", endpoint[key], paired_endpoints[seed][key])
            else:
                candidate = new_candidate(case, seed)
                retain_report(row, 0, candidate.snapshot(), oracles[case][0])
                events = fixture(case)[3]
                for index, event in enumerate(events):
                    diagnostic = candidate.process(Observation(*event))
                    row["events"].append(json_value(diagnostic))
                    row["last_valid_timestamp_us"], row["last_valid_phase"] = event[0], event[1]
                    final_at_time = index + 1 == len(events) or events[index + 1][0] != event[0]
                    if final_at_time and event[0] in REPORTS[case]:
                        retain_report(row, event[0], candidate.snapshot(), oracles[case][event[0]])
                    save_json(STUDY / "inventory.json", inventory, [STUDY], 64 * MIB)
            row["execution"] = "completed"
        except OutputLimit as exc:
            row.update(execution="failed", assessment="fail", reason=str(exc))
            mark_resource_stop(inventory, str(exc))
            row["elapsed_s"] = time.monotonic() - start
            save_json(STUDY / "inventory.json", inventory)
            return 1
        except Exception as exc:
            status = "invalid" if isinstance(exc, InvalidInput) else (
                "unresolved" if isinstance(exc, (UnresolvedInference, OracleUnresolved)) else "failed")
            row.update(execution=status, reason=f"{type(exc).__name__}: {exc}")
            mark_unavailable(row, row["reason"])
        row["elapsed_s"] = time.monotonic() - start
        save_json(STUDY / "inventory.json", inventory, [STUDY], 64 * MIB)
    inventory["global_resource"] = "completed"  # Wrapper replaces this after independent accounting.
    aggregate_inventory(inventory)
    save_json(STUDY / "inventory.json", inventory, [STUDY], 64 * MIB)
    plot_results(inventory)
    return 0


def observed_accuracy_series(row, prefix):
    """Pair saved errors with prescribed times, independent of JSON key order."""
    times, values = [], []
    for timestamp in REPORTS["O"]:
        report = row["reports"][str(timestamp)]
        errors = [abs(metric["value"]) for name, metric in report["metrics"].items()
                  if name.startswith(prefix) and metric["value"] is not None]
        times.append(timestamp / 1e6)
        values.append(max(errors) if errors else math.nan)
    return times, values


def plot_results(inventory, destination=None):
    """Render saved tables; an explicit correction destination must be fresh."""
    output = STUDY if destination is None else Path(destination)
    if destination is not None:
        original, target = STUDY.resolve(), output.resolve()
        if target == original or original in target.parents:
            raise ValueError("correction destination must be outside the original study")
        output.mkdir(exist_ok=False)
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, quantity in zip(axes, ("relative_var_x", "relative_var_y")):
        for branch, marker in (("H1", "o"), ("H8", "x")):
            rows = [r for r in inventory["rows"] if r["branch"] == branch]
            values = [r["reports"]["100000"]["metrics"][quantity]["value"] for r in rows]
            axis.plot(SEEDS, [np.nan if v is None else 1+v for v in values], marker, label=branch)
            for index, row in enumerate(rows):
                axis.text(.02, .98-.055*(index + (0 if branch == "H1" else 4)),
                          f"{row['seed']} {branch}: {row['assessment']}", transform=axis.transAxes, fontsize=7, va="top")
            for seed, value in zip(SEEDS, values):
                if value is None:
                    axis.annotate("unavailable", (seed, 1), rotation=90)
        axis.axhline(1, color="black", linewidth=1)
        axis.set(title=quantity.replace("relative_var_", "Position variance ") + " / exact",
                 xlabel="paired algorithm seed", ylabel="variance ratio")
        axis.ticklabel_format(useOffset=False, style="plain", axis="x")
        axis.legend()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    artifact_bytes(output / "heading-variance.png", buffer.getvalue(), root=output)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row in [r for r in inventory["rows"] if r["case"] == "O"]:
        events = row["events"]
        if events:
            ts = [e["timestamp_us"] / 1e6 for e in events]
            axes[0, 0].plot(ts, [e["ess"] / 2048 for e in events], label=str(row["seed"]))
            axes[0, 1].plot(ts, [e["max_weight"] for e in events], label=str(row["seed"]))
        for axis, prefix in zip(axes[1], ("mean_", "cov_")):
            times, values = observed_accuracy_series(row, prefix)
            axis.plot(times, values, "o-", label=str(row["seed"]))
        if row["execution"] != "completed" or row["assessment"] != "pass":
            axes[0, 0].text(.02, .95-.07*list(SEEDS).index(row["seed"]),
                f"{row['seed']}: {row['execution']} / {row['assessment']}", transform=axes[0, 0].transAxes)
    axes[0, 0].axhline(.1, color="black", linestyle="--")
    axes[0, 1].axhline(.05, color="black", linestyle="--")
    for axis, title in zip(axes.flat, ("ESS / M, every sensor phase", "Maximum weight, every sensor phase",
                                      "Maximum |whitened mean error|", "Maximum |whitened covariance entry error|")):
        axis.set(title=title, xlabel="time (s)")
        axis.legend()
    fig.suptitle("O: full six-state accuracy at 0 and 0.01 s; latent four-state only thereafter")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    artifact_bytes(output / "observed-diagnostics.png", buffer.getvalue(), root=output)



def verify_plotting_cache(expected):
    if not expected:
        raise RuntimeError("reviewed plotting font cache is missing")
    cache = CHECKS / "matplotlib-cache"
    actual = {path.name for path in cache.glob("fontlist-*.json")}
    if actual != set(expected):
        raise RuntimeError("plotting cache file set changed")
    for name, value in expected.items():
        path = cache / name
        if not path.stat().st_size or digest(path) != value:
            raise RuntimeError("plotting cache hash changed")
    return cache

def verify_gate(gate):
    verify_plotting_cache(gate.get("plotting_cache_sha256"))
    if gate.get("decision") != "eligible" or not gate.get("independent_review_accepted"):
        raise RuntimeError("final independent review and eligible prelaunch gate are required")
    if digest(ROOT / "docs/trajectory_mixture_proposal.md") != PROPOSAL_HASH:
        raise RuntimeError("frozen proposal changed")
    if digest(ROOT / "docs/trajectory_mixture_design.md") != DESIGN_HASH:
        raise RuntimeError("frozen design changed")
    for name, expected in gate["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"reviewed source changed: {name}")
    ledger = json.loads((CHECKS / "validation.json").read_text())
    if any(a["status"] == "running" for a in ledger["attempts"]):
        raise RuntimeError("validation attempt is unfinished")
    if sum(a["charged_s"] for a in ledger["attempts"]) > 900:
        raise RuntimeError("validation allowance exhausted")
    if output_bytes(validation_roots()) > 256 * MIB:
        raise RuntimeError("validation output allowance exceeded")
    if not gate.get("final_required_checks_passed") or not gate.get("clean_committed_source"):
        raise RuntimeError("required final checks and clean source attestation missing")



def final_memory_accounting(result, inventory):
    """A missing final measurement cannot inherit a previous completed gate."""
    try:
        high = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if high <= 0:
            raise AccountingError("missing final wrapper high water")
        result["wrapper_high_water_bytes"] = high
        if result["worker_job_high_water_bytes"] is not None:
            result["assessed_peak_bytes"] = max(result["sampled_aggregate_peak_bytes"],
                                                 high + result["worker_job_high_water_bytes"])
    except Exception as exc:
        if result["status"] != "failed":
            result.update(status="unresolved", reason=f"final memory accounting failed: {exc}")
            inventory["global_resource"] = "unresolved"
            aggregate_inventory(inventory)

def launch_study(started, owned_root):
    gate = json.loads((CHECKS / "gate.json").read_text())
    verify_gate(gate)
    # Exclusive creation is the one-launch guard; no retry directory is supported.
    STUDY.mkdir()
    owned_root["created"] = True
    inventory = planned_inventory()
    save_json(STUDY / "inventory.json", inventory, [STUDY], 64 * MIB)
    # Reuse the development demo's font index, keeping all study cache files in
    # the assessed output root and avoiding a font-discovery child process.
    cache_source = CHECKS / "matplotlib-cache"
    cache_target = STUDY / "matplotlib-cache"
    cache_target.mkdir()
    for path in cache_source.glob("fontlist-*.json"):
        artifact_bytes(cache_target / path.name, path.read_bytes())
    os.environ["MPLCONFIGDIR"] = str(cache_target)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    save_json(STUDY / "manifest.json", {
        "git_commit": gate["git_commit"], "source_sha256": gate["source_sha256"],
        "proposal_sha256": PROPOSAL_HASH, "design_sha256": DESIGN_HASH,
        "python_version": sys.version, "python_executable": sys.executable,
        "platform": sys.platform, "thread_environment": {k: os.environ[k] for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "stream": "NumPy PCG64(SeedSequence([20261018,case_id,replicate_seed,phase]))",
        "case_ids": {"L":1,"H":2,"O":3}, "phases": {"initialization":0,"transition":1},
        "draw_order": "component-major; active theta,omega,b initialization; nonzero omega,b transition; zero coordinates draw nothing",
        "started_unix_s": time.time()-(time.monotonic()-started),
        "component_count": 2048, "seed_groups": list(SEEDS), "output_reserve_bytes": 4*MIB,
        "gate_sha256": digest(CHECKS / "gate.json"), "validation_ledger_sha256": digest(CHECKS / "validation.json")
    }, [STUDY], 64 * MIB)
    # Fixture serialization is evaluation metadata; no random streams are opened.
    save_json(STUDY / "checkpoint-factors.json", {"J":[[.2,0,0],[.05,.15,0],[.04,-.02,.25]],
        "La":[[.3,0,0],[.02,.08,0],[-.01,-.03,.04]],
        "K":[[.4,.2,-.3],[-.2,.5,.1],[.3,-.4,.2]],
        "use":"frozen L/O evaluation metadata only; candidate receives assembled full mean/covariance"}, [STUDY],64*MIB)
    save_json(STUDY / "inputs.json", json_value({case: {"mean": fixture(case)[0],
        "covariance": fixture(case)[1], "diffusion": fixture(case)[2], "events": fixture(case)[3],
        "reports_us": REPORTS[case]} for case in ("L", "H", "O")}), [STUDY], 64*MIB)
    os.environ["NAVTRACE_STUDY_WORKER_TOKEN"] = str(os.getpid())
    result = monitor_command([sys.executable, str(Path(__file__).resolve()), "worker"],
                             STUDY / "worker.log", [STUDY], 120, 512*MIB, 64*MIB,
                             study=True, started=started)
    try:
        inventory = json.loads((STUDY / "inventory.json").read_text())
    except Exception as exc:
        inventory = planned_inventory()
        result.update(status="failed", reason=f"worker inventory unreadable: {exc}")
    if result["status"] == "failed":
        mark_resource_stop(inventory, result["reason"])
    elif result["status"] == "unresolved":
        inventory["global_resource"] = "unresolved"
        for row in inventory["rows"]:
            if row["execution"] == "running":
                row.update(execution="unresolved", reason=result["reason"])
            mark_unavailable(row, result["reason"])
    else:
        inventory["global_resource"] = "completed"
    aggregate_inventory(inventory)
    result["elapsed_s"] = time.monotonic() - started
    save_json(STUDY / "resources.json", result)
    save_json(STUDY / "inventory.json", inventory)
    manifest = json.loads((STUDY / "manifest.json").read_text())
    import numpy as np
    manifest["numpy_version"] = np.__version__
    manifest["artifact_sha256"] = {str(p.relative_to(STUDY)): digest(p) for p in sorted(STUDY.rglob("*"))
                                     if p.is_file() and p.name != "manifest.json"}
    manifest["decision"] = inventory["decision"]
    manifest["finalization_unix_s"] = time.time()
    manifest["finalization_timestamp_note"] = "recorded immediately before terminal metadata flush"
    save_json(STUDY / "manifest.json", manifest)
    elapsed = time.monotonic() - started
    retained = output_bytes([STUDY])
    final_memory_accounting(result, inventory)
    result["elapsed_through_primary_metadata_flush_s"] = elapsed
    result["retained_bytes"] = retained
    if elapsed > 120 or retained > 64*MIB or (result["assessed_peak_bytes"] or 0) > 512*MIB:
        result.update(status="failed", reason="final artifact flush exceeded study resource allowance")
        mark_resource_stop(inventory, result["reason"])
    save_json(STUDY / "resources.json", result)
    save_json(STUDY / "inventory.json", inventory)
    manifest["artifact_sha256"] = {str(p.relative_to(STUDY)): digest(p) for p in sorted(STUDY.rglob("*"))
                                     if p.is_file() and p.name != "manifest.json"}
    manifest["decision"] = inventory["decision"]
    manifest["finalization_unix_s"] = time.time()
    manifest["finalization_timestamp_note"] = "recorded immediately before terminal metadata flush"
    save_json(STUDY / "manifest.json", manifest)
    final_elapsed, final_bytes = time.monotonic()-started, output_bytes([STUDY])
    if final_elapsed > 120 or final_bytes > 64*MIB:
        result.update(status="failed", reason="final metadata flush exceeded study allowance",
                      elapsed_through_final_flush_s=final_elapsed, retained_bytes=final_bytes)
        mark_resource_stop(inventory, result["reason"])
        save_json(STUDY / "resources.json", result)
        save_json(STUDY / "inventory.json", inventory)
        manifest["decision"] = inventory["decision"]
        manifest["artifact_sha256"] = {str(p.relative_to(STUDY)): digest(p) for p in sorted(STUDY.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}
        save_json(STUDY / "manifest.json", manifest)
    print(json.dumps({"decision": inventory["decision"], "resource_status": result["status"],
                      "elapsed_through_final_flush_s": time.monotonic()-started,
                      "retained_bytes": output_bytes([STUDY])}))
    return 0 if result["status"] == "completed" else 1


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    validation = sub.add_parser("validate")
    validation.add_argument("--label", required=True)
    validation.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("study")
    sub.add_parser("worker")
    args = parser.parse_args()
    if args.action == "validate":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return validation_command(args.label, command)
    if args.action == "study":
        owned_root = {"created": False}
        try:
            return launch_study(started, owned_root)
        except Exception as exc:
            if owned_root["created"]:
                path = STUDY / "inventory.json"
                inventory = json.loads(path.read_text()) if path.exists() else planned_inventory()
                reason = f"study wrapper exception: {type(exc).__name__}: {exc}"
                mark_resource_stop(inventory, reason)
                save_json(path, inventory)
                save_json(STUDY / "wrapper-failure.json", {"reason": reason,
                    "elapsed_s": time.monotonic()-started, "finalization_unix_s": time.time(),
                    "measurement_status": "incomplete; no compliance claim"})
            raise
    if os.environ.get("NAVTRACE_STUDY_WORKER_TOKEN") != str(os.getppid()):
        raise RuntimeError("worker must be the direct child of the exclusive study wrapper")
    if not (STUDY / "manifest.json").exists():
        raise RuntimeError("worker requires the parent-created one-launch manifest")
    return study_worker()


if __name__ == "__main__":
    sys.exit(main())
