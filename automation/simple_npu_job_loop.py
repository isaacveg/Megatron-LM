#!/usr/bin/env python3
"""Simple sequential NPU job loop.

This script intentionally avoids the full supervisor design. It does one thing:
wait for a fixed Ascend Phy-ID set to become idle, then run shell scripts from a
container directory one by one. Failed scripts are retried; repeatedly failing
scripts are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_devices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sh_join(args: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in args)


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        line = f"[{now()}] {message}"
        with self._lock:
            print(line, flush=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def raw(self, message: str) -> None:
        with self._lock:
            print(message, end="", flush=True)
            with self.path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(message)


def run_cmd(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def docker_exec(container: str, command: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return run_cmd(["docker", "exec", container, "bash", "-lc", command], timeout=timeout)


def docker_is_running(container: str) -> bool:
    result = run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", container], timeout=20)
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_container(container: str, log: Log) -> bool:
    result = run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", container], timeout=20)
    if result.returncode != 0:
        log.write(f"container does not exist: {container}")
        return False
    if result.stdout.strip() == "true":
        return True
    log.write(f"starting stopped container: {container}")
    start = run_cmd(["docker", "start", container], timeout=60)
    if start.returncode != 0:
        log.write(f"docker start failed: {start.stderr.strip()}")
        return False
    return True


def list_container_scripts(container: str, script_dir: str, pattern: str) -> list[str]:
    command = (
        f"find {shlex.quote(script_dir)} -maxdepth 1 -type f -name {shlex.quote(pattern)} "
        "| sort"
    )
    result = docker_exec(container, command, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to list scripts in {script_dir}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_npu_smi_info(output: str) -> dict[int, dict[str, Any]]:
    """Parse `npu-smi info` by Ascend Phy-ID.

    The sample output groups two chips under one NPU id. For scheduling an
    8-device slice out of 16 devices, Phy-ID is the useful index.
    """

    text = output.replace("\xa0", " ")
    devices: dict[int, dict[str, Any]] = {}
    chip_to_phy: dict[tuple[int, int], int] = {}
    current_npu: int | None = None
    current_health = "unknown"
    process_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "Process id" in line and "Process name" in line:
            process_section = True
            current_npu = None
            continue

        if process_section:
            if not line.startswith("|"):
                continue
            cols = [col.strip() for col in line.split("|")[1:-1]]
            if len(cols) < 4:
                continue
            npu_chip = re.fullmatch(r"(\d+)\s+(\d+)", cols[0])
            if not npu_chip or not cols[1].isdigit():
                continue
            npu_id = int(npu_chip.group(1))
            chip_id = int(npu_chip.group(2))
            phy_id = chip_to_phy.get((npu_id, chip_id))
            if phy_id is None:
                continue
            entry = devices.setdefault(
                phy_id,
                {
                    "aicore": 0,
                    "hbm": 0,
                    "hbm_total": 0,
                    "health": "unknown",
                    "process_count": 0,
                    "processes": [],
                },
            )
            entry["process_count"] = parse_int(entry.get("process_count"), 0) + 1
            entry.setdefault("processes", []).append(
                {
                    "npu": npu_id,
                    "chip": chip_id,
                    "pid": int(cols[1]),
                    "name": cols[2],
                    "memory": parse_int(cols[3], 0),
                }
            )
            continue

        if not line.startswith("|"):
            continue
        cols = [col.strip() for col in line.split("|")[1:-1]]
        if len(cols) < 3:
            continue

        header = re.match(r"^(\d+)\s+Ascend", cols[0])
        if header:
            current_npu = int(header.group(1))
            current_health = cols[1].split()[0] if cols[1] else "unknown"
            continue

        chip_line = re.fullmatch(r"(\d+)\s+(\d+)", cols[0])
        if current_npu is None or not chip_line or ":" not in cols[1]:
            continue

        metrics = re.match(r"^(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", cols[2])
        if not metrics:
            continue

        chip_id = int(chip_line.group(1))
        phy_id = int(chip_line.group(2))
        aicore = int(metrics.group(1))
        hbm = int(metrics.group(2))
        hbm_total = int(metrics.group(3))
        chip_to_phy[(current_npu, chip_id)] = phy_id
        devices[phy_id] = {
            "npu": current_npu,
            "chip": chip_id,
            "aicore": aicore,
            "hbm": hbm,
            "hbm_total": hbm_total,
            "health": current_health,
            "process_count": 0,
            "processes": [],
        }

    return devices


def npu_snapshot(command: str) -> dict[int, dict[str, Any]]:
    result = run_cmd(["bash", "-lc", command], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "npu-smi failed")
    snapshot = parse_npu_smi_info(result.stdout)
    if not snapshot:
        raise RuntimeError("could not parse any NPU devices from npu-smi output")
    return snapshot


def devices_idle(
    snapshot: dict[int, dict[str, Any]],
    devices: list[int],
    *,
    max_hbm_mb: int,
    max_aicore_pct: int,
    require_no_processes: bool,
) -> tuple[bool, str]:
    for device in devices:
        item = snapshot.get(device)
        if item is None:
            return False, f"Phy-ID {device}: missing in npu-smi output"
        health = str(item.get("health") or "unknown")
        aicore = parse_int(item.get("aicore"), 0)
        hbm = parse_int(item.get("hbm"), 0)
        process_count = parse_int(item.get("process_count"), 0)
        if health != "OK":
            return False, f"Phy-ID {device}: health={health}"
        if aicore > max_aicore_pct:
            return False, f"Phy-ID {device}: AICore={aicore}%"
        if hbm > max_hbm_mb:
            return False, f"Phy-ID {device}: HBM={hbm}MB"
        if require_no_processes and process_count > 0:
            return False, f"Phy-ID {device}: processes={process_count}"
    return True, "idle"


def wait_for_idle(args: argparse.Namespace, log: Log, stop_event: threading.Event) -> bool:
    passed = 0
    while not stop_event.is_set():
        try:
            snapshot = npu_snapshot(args.npu_smi_command)
            ok, reason = devices_idle(
                snapshot,
                args.devices,
                max_hbm_mb=args.idle_hbm_mb,
                max_aicore_pct=args.idle_aicore_pct,
                require_no_processes=not args.allow_existing_processes,
            )
        except Exception as exc:
            ok = False
            reason = f"npu-smi error: {exc}"

        if ok:
            passed += 1
            log.write(f"idle check passed ({passed}/{args.idle_checks})")
            if passed >= args.idle_checks:
                if args.cooldown_sec > 0:
                    log.write(f"cooldown {args.cooldown_sec}s before launch")
                    if stop_event.wait(args.cooldown_sec):
                        return False
                return True
        else:
            if passed:
                log.write("idle check reset")
            passed = 0
            log.write(f"devices busy: {reason}")

        stop_event.wait(args.check_interval_sec)
    return False


def shell_assignment(key: str, value: str) -> str:
    return f"{key}={shlex.quote(value)}"


def make_container_run_command(args: argparse.Namespace, script: str, pid_file: str) -> str:
    env = {
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(x) for x in args.devices),
        "ASCEND_VISIBLE_DEVICES": ",".join(str(x) for x in args.devices),
        "NPU_VISIBLE_DEVICES": ",".join(str(x) for x in args.devices),
        "NPUS_PER_NODE": str(len(args.devices)),
        "TRAIN_SCRIPT": script,
        "RUNNER_PID_FILE": pid_file,
    }
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value

    env_cmd = " ".join(shell_assignment(k, v) for k, v in env.items())
    workdir = shlex.quote(args.workdir)
    runtime_dir = shlex.quote(args.runtime_dir)
    pid_file_q = shlex.quote(pid_file)
    inner = (
        "echo $$ > \"$RUNNER_PID_FILE\"; "
        "trap 'rm -f \"$RUNNER_PID_FILE\"' EXIT; "
        "exec bash \"$TRAIN_SCRIPT\""
    )
    return (
        f"mkdir -p {runtime_dir} && rm -f {pid_file_q} && cd {workdir} && "
        f"{env_cmd} setsid bash -lc {shlex.quote(inner)}"
    )


def stop_container_job(container: str, pid_file: str, log: Log) -> None:
    command = (
        f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        "if [ -n \"$pid\" ]; then "
        "echo stopping container process group $pid; "
        "kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true; "
        "sleep 5; "
        "kill -KILL -$pid 2>/dev/null || true; "
        "fi; "
        f"rm -f {shlex.quote(pid_file)}"
    )
    result = docker_exec(container, command, timeout=20)
    if result.stdout.strip():
        log.write(result.stdout.strip())
    if result.returncode != 0:
        log.write(f"stop command failed: {result.stderr.strip()}")


def run_script_once(args: argparse.Namespace, script: str, log: Log, stop_event: threading.Event) -> int:
    pid_file = f"{args.runtime_dir.rstrip('/')}/{args.name}.current.pid"
    command = make_container_run_command(args, script, pid_file)
    full = ["docker", "exec", args.container, "bash", "-lc", command]
    log.write(f"launching script: {script}")
    log.write(f"docker exec command: {sh_join(full)}")

    proc = subprocess.Popen(
        full,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    assert proc.stdout is not None
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            log.raw(line)
        if stop_event.is_set():
            log.write("stop requested; terminating current container job")
            stop_container_job(args.container, pid_file, log)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return 130

    for line in proc.stdout:
        if line:
            log.raw(line)
    rc = proc.returncode if proc.returncode is not None else 1
    log.write(f"script exited: script={script} rc={rc}")
    return rc


def read_container_file(container: str, path: str) -> str | None:
    result = docker_exec(container, f"cat {shlex.quote(path)}", timeout=20)
    if result.returncode != 0:
        return None
    return result.stdout


def simple_expand(value: str, variables: dict[str, str]) -> str:
    previous = None
    current = value.strip().strip('"').strip("'")
    while previous != current:
        previous = current
        for key, replacement in variables.items():
            current = current.replace("${" + key + "}", replacement)
            current = current.replace("$" + key, replacement)
    return current


def detect_checkpoint_dir(container: str, script: str, default_ckpt_root: str | None, log: Log) -> str | None:
    content = read_container_file(container, script)
    if not content:
        return None

    values: dict[str, str] = {}
    patterns = {
        "RUN_NAME": r"^\s*(?:export\s+)?RUN_NAME=(?:\"\$\{RUN_NAME:-([^\"}]+)\}\"|\"([^\"]+)\"|'([^']+)'|([^\s#]+))",
        "CKPT_ROOT": r"^\s*(?:export\s+)?CKPT_ROOT=(?:\"\$\{CKPT_ROOT:-([^\"}]+)\}\"|\"([^\"]+)\"|'([^']+)'|([^\s#]+))",
        "CKPT_SAVE_DIR": r"^\s*(?:export\s+)?CKPT_SAVE_DIR=(?:\"\$\{CKPT_SAVE_DIR:-([^\"}]+)\}\"|\"([^\"]+)\"|'([^']+)'|([^\s#]+))",
    }
    for line in content.splitlines():
        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                value = next((group for group in match.groups() if group), "")
                if value:
                    values[key] = simple_expand(value, values)

    if "CKPT_SAVE_DIR" in values:
        return simple_expand(values["CKPT_SAVE_DIR"], values)
    if default_ckpt_root and "RUN_NAME" in values:
        return f"{default_ckpt_root.rstrip('/')}/{values['RUN_NAME']}"
    log.write(f"could not detect checkpoint dir from script: {script}")
    return None


def cleanup_checkpoints(args: argparse.Namespace, script: str, log: Log) -> None:
    if args.keep_last_checkpoints < 0:
        return
    ckpt_dir = args.ckpt_dir or detect_checkpoint_dir(args.container, script, args.ckpt_root, log)
    if not ckpt_dir:
        log.write(f"checkpoint cleanup skipped: no checkpoint dir for {script}")
        return

    find_cmd = f"find {shlex.quote(ckpt_dir)} -maxdepth 1 -type d -name 'iter_*' -printf '%f\\n' 2>/dev/null"
    result = docker_exec(args.container, find_cmd, timeout=60)
    if result.returncode != 0:
        log.write(f"checkpoint cleanup skipped: cannot list {ckpt_dir}")
        return

    entries: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        name = line.strip()
        match = re.fullmatch(r"iter_(\d+)", name)
        if match:
            entries.append((int(match.group(1)), name))
    entries.sort()

    keep = max(0, args.keep_last_checkpoints)
    if len(entries) <= keep:
        log.write(f"checkpoint cleanup: nothing to remove in {ckpt_dir}")
        return

    remove = entries[:-keep] if keep else entries
    remove_paths = [f"{ckpt_dir.rstrip('/')}/{name}" for _, name in remove]
    rm_cmd = "rm -rf " + " ".join(shlex.quote(path) for path in remove_paths)
    log.write(f"checkpoint cleanup: removing {len(remove_paths)} old checkpoint(s) from {ckpt_dir}")
    rm = docker_exec(args.container, rm_cmd, timeout=300)
    if rm.returncode != 0:
        log.write(f"checkpoint cleanup failed: {rm.stderr.strip()}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"scripts": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"scripts": {}}
    data.setdefault("scripts", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state["updated_at"] = now()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def start_quit_watcher(stop_event: threading.Event, log: Log) -> None:
    if not sys.stdin.isatty():
        log.write("stdin is not a TTY; press Ctrl-C or kill the runner process to stop")
        return

    def watch() -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            log.write("press q to stop this runner")
            while not stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    log.write("q pressed; stop requested")
                    stop_event.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()


def run_loop(args: argparse.Namespace) -> int:
    log = Log(args.log_file)
    state = load_state(args.state_file)
    stop_event = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        log.write(f"signal {signum} received; stop requested")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    start_quit_watcher(stop_event, log)

    log.write(f"runner started: name={args.name} devices={','.join(str(x) for x in args.devices)}")
    if not ensure_container(args.container, log):
        return 1

    while not stop_event.is_set():
        try:
            scripts = list_container_scripts(args.container, args.script_dir, args.pattern)
        except Exception as exc:
            log.write(f"failed to list scripts: {exc}")
            if args.once:
                return 1
            stop_event.wait(args.rescan_interval_sec)
            continue

        if not scripts:
            log.write(f"no scripts found in {args.script_dir} matching {args.pattern}")
            if args.once:
                break
            stop_event.wait(args.rescan_interval_sec)
            continue

        progressed = False
        for script in scripts:
            if stop_event.is_set():
                break
            item = state["scripts"].setdefault(script, {"status": "pending", "consecutive_failures": 0})
            if item.get("status") in {"completed", "skipped"}:
                continue

            while not stop_event.is_set():
                failures = parse_int(item.get("consecutive_failures"), 0)
                if failures >= args.max_failures:
                    item["status"] = "skipped"
                    item["skipped_at"] = now()
                    log.write(f"skip script after {failures} consecutive failure(s): {script}")
                    save_state(args.state_file, state)
                    break

                item["status"] = "waiting_idle"
                save_state(args.state_file, state)
                if not wait_for_idle(args, log, stop_event):
                    break

                item["status"] = "running"
                item["attempts"] = parse_int(item.get("attempts"), 0) + 1
                item["last_start"] = now()
                save_state(args.state_file, state)

                rc = run_script_once(args, script, log, stop_event)
                item["last_rc"] = rc
                item["last_end"] = now()
                if stop_event.is_set():
                    item["status"] = "stopped"
                    save_state(args.state_file, state)
                    break
                if rc == 0:
                    item["status"] = "completed"
                    item["completed_at"] = now()
                    item["consecutive_failures"] = 0
                    save_state(args.state_file, state)
                    log.write(f"completed script: {script}")
                    cleanup_checkpoints(args, script, log)
                    progressed = True
                    break

                item["status"] = "failed"
                item["consecutive_failures"] = failures + 1
                save_state(args.state_file, state)
                log.write(
                    f"script failed: {script} rc={rc} "
                    f"consecutive_failures={item['consecutive_failures']}/{args.max_failures}"
                )
                if args.retry_sleep_sec > 0:
                    stop_event.wait(args.retry_sleep_sec)

        if args.once:
            break
        if not progressed:
            log.write(f"cycle finished; sleeping {args.rescan_interval_sec}s before rescan")
        stop_event.wait(args.rescan_interval_sec)

    save_state(args.state_file, state)
    log.write("runner stopped")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple sequential Ascend NPU job runner")
    parser.add_argument("--name", default="npu-runner", help="runner name used in logs and pid files")
    parser.add_argument("--container", required=True, help="Docker container name")
    parser.add_argument("--workdir", required=True, help="container workdir for running training scripts")
    parser.add_argument("--script-dir", required=True, help="container directory containing experiment scripts")
    parser.add_argument("--pattern", default="*.sh", help="script filename pattern")
    parser.add_argument("--devices", required=True, type=parse_devices, help="Ascend Phy-ID list, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--log-file", type=Path, required=True, help="host log file")
    parser.add_argument("--state-file", type=Path, required=True, help="host state file")
    parser.add_argument("--runtime-dir", default="/tmp/simple-npu-job-loop", help="container runtime dir for pid file")
    parser.add_argument("--npu-smi-command", default="npu-smi info")
    parser.add_argument("--idle-hbm-mb", type=int, default=8000)
    parser.add_argument("--idle-aicore-pct", type=int, default=10)
    parser.add_argument("--allow-existing-processes", action="store_true")
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--check-interval-sec", type=int, default=60)
    parser.add_argument("--cooldown-sec", type=int, default=300)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=int, default=60)
    parser.add_argument("--rescan-interval-sec", type=int, default=300)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2, help="set -1 to disable cleanup")
    parser.add_argument("--ckpt-root", help="fallback container checkpoint root if script only defines RUN_NAME")
    parser.add_argument("--ckpt-dir", help="force one container checkpoint dir for cleanup")
    parser.add_argument("--env", action="append", default=[], help="extra env KEY=VALUE passed to scripts")
    parser.add_argument("--once", action="store_true", help="run one pass over current scripts and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
