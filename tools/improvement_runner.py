"""Sequential, resumable CLI supervisor for the improvement programme (stdlib only)."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"DONE", "EVALUATED_NO_GAIN", "DEFERRED_DATA"}
_ADDITIONAL_SECRETS = set()


class RunnerError(RuntimeError):
    pass


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RunLock:
    """OS-owned byte/flock lock; a stale PID file never grants ownership."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise RunnerError("Another runner owns this repository. Use status/watch or stop.") from exc
        return self

    def __exit__(self, *unused):
        if self.file:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None


def lock_is_held(path):
    try:
        with RunLock(path):
            return False
    except RunnerError:
        return True


def ledger(root):
    text = (root / "memory/improvement/STATE.md").read_text(encoding="utf-8-sig")
    next_match = re.search(r"^Next step:\s*(\d{2}|none)\s*$", text, re.M)
    rows = dict(re.findall(r"^\|\s*(\d{2})\s*\|[^|]*\|\s*([A-Z_]+)\s*\|", text, re.M))
    if not next_match or not rows:
        raise RunnerError("STATE.md has no valid next step or stage ledger.")
    return next_match.group(1), rows


def next_stage(root, manifest):
    next_id, rows = ledger(root)
    ids = [step["id"] for step in manifest["steps"]]
    if list(rows) != ids:
        raise RunnerError("STATE stage order differs from the manifest.")
    optional = {sid for stage in manifest["steps"] for sid in stage.get("review_of", [])}
    invalid_deferred = [sid for sid, value in rows.items() if value == "DEFERRED_DATA" and sid not in optional]
    if invalid_deferred:
        raise RunnerError(f"Required stages cannot be skipped as data-deferred: {invalid_deferred}.")
    pending = [sid for sid in ids if rows[sid] not in TERMINAL]
    expected = pending[0] if pending else "none"
    if next_id != expected:
        raise RunnerError(f"Refusing skipped stages: STATE queues {next_id}; first unfinished stage is {expected}.")
    if expected != "none" and rows[expected] == "BLOCKED":
        raise RunnerError(f"Stage {expected} is BLOCKED. Resolve its recorded dependency before restarting.")
    return next((s for s in manifest["steps"] if s["id"] == expected), None)


def check_memory(root):
    result = subprocess.run([sys.executable, str(root / "tools/improvement_memory.py"), "check"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode:
        # This utility prints only known document paths and validation errors.
        detail = "\n".join(line for line in result.stdout.splitlines() if line.startswith("FAIL:"))
        raise RunnerError("Memory validation failed. " + (detail or "Run tools/improvement_memory.py check for details."))


def evidence_hash(root, sid):
    path = root / "memory/improvement/stages" / (sid + ".md")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def validate_completion(root, manifest, sid, before, previous_evidence, exit_code, stream_error):
    if exit_code != 0 or stream_error:
        raise RunnerError(f"Stage {sid} CLI failed (exit {exit_code}); " + (stream_error or "see stage events"))
    _, after = ledger(root)
    changed_others = [key for key in set(before) | set(after) if key != sid and before.get(key) != after.get(key)]
    if changed_others:
        raise RunnerError(f"Stage {sid} changed other stage statuses: {changed_others}. Manual review required.")
    if after.get(sid) not in TERMINAL:
        raise RunnerError(f"Stage {sid} has no accepted disposition (status {after.get(sid)}). Exit zero is not completion.")
    current = evidence_hash(root, sid)
    if not current or current == previous_evidence:
        raise RunnerError(f"Stage {sid} needs fresh stage evidence; exit zero is not completion.")
    if after[sid] == "DEFERRED_DATA":
        note = (root / "memory/improvement/stages" / (sid + ".md")).read_text(encoding="utf-8-sig")
        if not re.search(r"^Candidate enabled:\s*no\s*$", note, re.M | re.I) or not re.search(r"^Resume condition:\s*\S.+$", note, re.M):
            raise RunnerError(f"Deferred stage {sid} lacks disabled-candidate/resume evidence.")
    check_memory(root)
    next_stage(root, manifest)  # Refuse even a memory-valid leap past a pending independent step.
    return after[sid]


def config_load(path):
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise RunnerError("runner.json schema_version must be 1.")
    if config["codex"].get("approval_policy") != "never" or config["codex"].get("sandbox") != "workspace-write":
        raise RunnerError("Codex must use workspace-write and approval_policy=never.")
    if config["claude"].get("permission_mode") != "acceptEdits":
        raise RunnerError("Claude must use permission_mode=acceptEdits.")
    for key in ("max_stage_minutes", "max_run_hours", "max_attempts_per_stage", "heartbeat_seconds"):
        if float(config["limits"][key]) <= 0:
            raise RunnerError(f"Invalid positive limit: {key}")
    attempts = config["limits"]["max_attempts_per_stage"]
    if type(attempts) is not int or attempts < 1:
        raise RunnerError("max_attempts_per_stage must be an integer >= 1.")
    performance = config.get("performance", {})
    workers = performance.get("cpu_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise RunnerError("performance.cpu_workers must be an integer from 1 to 32.")
    if not isinstance(performance.get("parallel_tests", False), bool):
        raise RunnerError("performance.parallel_tests must be true or false.")
    for assigned, override in config.get("runtime_overrides", {}).items():
        target = override.get("model")
        if not assigned or target not in config["claude"]["models"]:
            raise RunnerError(f"Invalid explicit runtime model override: {assigned} -> {target}")
        if not str(override.get("reason", "")).strip():
            raise RunnerError(f"Runtime model override for {assigned} needs a recorded reason.")
    return config


def executable(value):
    found = shutil.which(value)
    if not found:
        raise RunnerError(f"Executable not found: {value}. Install/sign in, then retry; no model substitution.")
    if os.name == "nt" and Path(found).suffix.lower() in {".cmd", ".bat", ".ps1"}:
        raise RunnerError("Use the native CLI executable path; shell wrapper executables are unsupported.")
    return found


def effective_assignment(step, config):
    override = config.get("runtime_overrides", {}).get(step["model"])
    if override:
        return override["model"], override["reason"]
    return step["model"], None


def command_for(step, config, root):
    effort = step["effort"].lower().replace("extra high (xhigh)", "xhigh")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise RunnerError(f"Unsupported effort: {effort}")
    assigned_model, _ = effective_assignment(step, config)
    if assigned_model.startswith("Codex"):
        cli = config["codex"]
        isolation = ["--ignore-user-config"] if cli.get("ignore_user_config", True) else []
        settings = []
        if os.name == "nt" and cli.get("windows_sandbox"):
            if cli["windows_sandbox"] not in {"elevated", "unelevated"}:
                raise RunnerError("Invalid configured Windows sandbox mode.")
            settings += ["-c", 'windows.sandbox="' + cli["windows_sandbox"] + '"']
        if cli.get("network_access") is True:
            settings += ["-c", "sandbox_workspace_write.network_access=true"]
        return [executable(cli["executable"]), "exec", *isolation, "--json", "-m", cli["model"], "-c",
                f'model_reasoning_effort="{effort}"', "-c", 'approval_policy="never"',
                *settings, "-s", "workspace-write", "-C", str(root), "-"]
    cli = config["claude"]
    model = cli["models"].get(assigned_model)
    if not model:
        raise RunnerError(f"No exact configured model for {assigned_model}; no fallback permitted.")
    return [executable(cli["executable"]), "--print", "--output-format", "stream-json", "--verbose",
            "--model", model, "--effort", effort, "--permission-mode", "acceptEdits",
            "--allowedTools", ",".join(cli["allowed_tools"])]


def prompt_for(root, step, config, attempt, repair_reason=None):
    source = (root / step["file"]).read_text(encoding="utf-8-sig")
    trial = config.get("hosted_trial", {})
    actual_model, override_reason = effective_assignment(step, config)
    assignment_note = f"Runtime model: {actual_model}."
    if override_reason:
        assignment_note += f" This is an explicit supervisor override of {step['model']}: {override_reason}"
    performance = config.get("performance", {})
    workers = int(performance.get("cpu_workers", 1))
    if performance.get("parallel_tests", False) and workers > 1:
        performance_note = (
            f"Performance budget: up to {workers} local CPU workers. For independent targeted test directories use "
            f"`python -m pytest -n auto --dist loadscope -q <paths>`; PYTEST_XDIST_AUTO_NUM_WORKERS is capped at {workers}. "
            "Run the complete repository suite serially because its process, port and shared-file tests are not xdist-safe. "
            "Keep small targeted test commands serial. You may run independent read-only checks concurrently, "
            "but wait for every child and never overlap commands that write the same file, database, cache or artifact.\n"
        )
    else:
        performance_note = "Performance budget: run verification serially.\n"
    envelope = (
        f"Automatic supervisor work order: stage {step['id']}, attempt {attempt}.\n"
        + assignment_note + "\n"
        + performance_note +
        "The user authorized the supervisor to execute the work orders in the manifest sequentially, "
        "including the daily-paper follow-up stages 19-24. "
        "You execute ONLY this numbered stage; the supervisor starts a fresh CLI session for the next one.\n"
        "Preserve all unrelated dirty changes. Do not start background agents/jobs that outlive this session. "
        "This is a one-shot CLI session: a final response ends it and the supervisor terminates its child jobs. "
        "Do not use ScheduleWakeup or end your response to await a background-task notification. "
        "For long commands, keep the session active and wait using blocking tool calls or the tool's task-output "
        "wait facility until every job has exited. Then inspect exit codes and saved outputs before writing "
        "completion evidence. On resume, reuse completed cached records and persistent budget accounting; "
        "never reset spend or duplicate an already-running job.\n"
        "Do not alter the runner, its configuration, manifest, other stage statuses, or acceptance gates.\n"
        "Read the current shared memory and prerequisites, inspect existing stage artifacts before continuing. "
        "Repair only this stage's own failed acceptance. Record genuine BLOCKED/DEFERRED_DATA states honestly.\n"
        "Emit brief plain-language progress at meaningful milestones. Never print secrets, credentials, "
        "environment dumps, or config.local.yaml contents. Keep application operation paper-only.\n"
        "Do not change models or permission modes. A CLI/subscription/access failure must be reported.\n"
        "For hosted text extraction only (including shadow integration), the configured model is " + str(trial.get("model", "unspecified")) +
        f"; total spend cap USD {trial.get('max_cost_usd', 0)}, total request cap {trial.get('max_requests', 0)}. "
        "These are hard ceilings across retries, not targets. Inspect persistent trial accounting before requests. "
        "Do not make paid requests without an implemented durable cap and confirmed applicable pricing. "
        "If OPENROUTER_API_KEY is absent, do not print/search for credentials; record the hosted trial unavailable.\n"
        "Before finishing, write fresh stages/NN.md evidence, update only this stage's ledger disposition, "
        "set Active step: none and queue the first unfinished numbered stage (or none if every stage has a "
        "valid disposition). Run improvement_memory.py export and check. Do not skip a pending stage.\n\n"
    )
    if repair_reason:
        envelope += ("\nThis stage's previous session was NOT accepted. Resume/revalidate only this stage; "
                     "its recorded DONE status alone is insufficient. Save fresh evidence of the current checks. "
                     "Specific supervisor finding: " + str(repair_reason) + "\n\n")
    return envelope + source


def performance_environment(config):
    """Expose a bounded parallel-test budget without oversubscribing numerical libraries."""
    performance = config.get("performance", {})
    workers = int(performance.get("cpu_workers", 1))
    return {
        "RP_CPU_WORKERS": str(workers),
        "PYTEST_XDIST_AUTO_NUM_WORKERS": str(workers),
    }


def protected_hashes(root):
    names = ("tools/improvement_runner.py", "tools/improvement_memory.py", "docs/improvement/runner.json", "docs/improvement/manifest.json")
    names += tuple(p.relative_to(root).as_posix() for p in sorted((root / "docs/improvement/prompts").glob("*.md")))
    names += ("docs/improvement/DAILY_PAPER_FOLLOWUP.md",)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() if (root / name).exists() else None for name in names}


def verify_protected(root, expected):
    actual = protected_hashes(root)
    changed = [name for name, digest in expected.items() if actual.get(name) != digest]
    if changed:
        raise RunnerError(f"Protected supervisor/validation files changed during the stage: {changed}. Review before resuming.")


def redact(text):
    # Never log tool inputs/outputs. Also redact known inherited secret values from prose/errors.
    for key, value in os.environ.items():
        if len(value) >= 8 and any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            text = text.replace(value, "[REDACTED]")
    for value in _ADDITIONAL_SECRETS:
        text = text.replace(value, "[REDACTED]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def stream_summary(line):
    """Return safe progress only: deliberately omit tool arguments, outputs, and reasoning."""
    try:
        event = json.loads(line)
    except (ValueError, TypeError):
        return [], None
    if not isinstance(event, dict):
        return [], None
    kind = event.get("type", "")
    failure = None
    messages = []
    if kind in {"error", "turn.failed"} or (kind == "result" and event.get("is_error")):
        failure = "CLI reported an error: " + redact(str(event.get("message") or event.get("error") or event.get("errors") or event.get("subtype") or "unknown"))[:1600]
    if kind == "assistant":
        for part in event.get("message", {}).get("content", []):
            if part.get("type") == "text":
                messages.append(redact(part.get("text", ""))[:2500])
            elif part.get("type") == "tool_use":
                messages.append("Tool: " + str(part.get("name", "unknown")))
    elif kind in {"item.started", "item.completed"}:
        item = event.get("item", {})
        if item.get("type") == "agent_message":
            messages.append(redact(item.get("text", ""))[:2500])
        elif kind == "item.started" and item.get("type") not in {"reasoning", "agent_message"}:
            messages.append("Tool: " + str(item.get("type", "operation")))
    elif kind == "result" and not event.get("is_error") and isinstance(event.get("result"), str):
        messages.append(redact(event["result"])[:2500])
    return messages, failure


class WindowsJob:
    """Kill-on-close job. The gated helper cannot spawn its CLI until assignment succeeds."""
    def __init__(self, process):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64), ("Flags", w.DWORD),
                        ("MinWorking", ctypes.c_size_t), ("MaxWorking", ctypes.c_size_t),
                        ("Active", w.DWORD), ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD), ("Scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t), ("PeakJob", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise RunnerError("Could not create a Windows process-ownership job.")
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not self.kernel.AssignProcessToJobObject(self.handle, w.HANDLE(int(process._handle))):
            self.close()
            raise RunnerError("Could not own the CLI process tree; refusing to launch it.")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def gated_child(command):
    # Unbuffered one-byte read: never consume any of the actual UTF-8 prompt.
    if os.read(sys.stdin.fileno(), 1) != b"\n":
        return 125
    return subprocess.call(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, shell=False)


def run_process(command, prompt, root, timeout, heartbeat, emit, env_overrides=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", RP_DISABLE_NOTIFICATIONS="1")
    env.update({str(key): str(value) for key, value in (env_overrides or {}).items()})
    if os.name == "nt" and not env.get("OPENROUTER_API_KEY"):
        import winreg
        with contextlib.suppress(OSError):
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "OPENROUTER_API_KEY")
                if value:
                    env["OPENROUTER_API_KEY"] = str(value)
                    _ADDITIONAL_SECRETS.add(str(value))
    wrapper = [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(command)]
    kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(wrapper, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, shell=False, **kwargs)
    job = None
    lines = queue.Queue()
    threads = []
    stream_error = None
    started = last_heartbeat = time.monotonic()
    try:
        if os.name == "nt":
            job = WindowsJob(process)
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            def reader(name=name, stream=stream):
                for line in iter(stream.readline, b""):
                    lines.put((name, line.decode("utf-8", errors="replace")))
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            threads.append(thread)
        def writer():
            try:
                process.stdin.write(b"\n" + prompt.encode("utf-8"))
                process.stdin.close()
            except (OSError, ValueError):
                pass  # Exit-code and completion validation decide whether the work succeeded.
        threading.Thread(target=writer, daemon=True).start()
        emit("process", f"CLI started (owned process {process.pid}); waiting for progress.")
        while process.poll() is None or any(t.is_alive() for t in threads) or not lines.empty():
            now = time.monotonic()
            if process.poll() is not None:
                if job:
                    job.close()
                elif os.name != "nt":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            if now - started >= timeout:
                raise RunnerError(f"Stage time limit reached after {timeout / 60:.1f} minutes. Owned process tree stopped.")
            try:
                channel, line = lines.get(timeout=0.2)
                messages, error = stream_summary(line)
                for message in messages:
                    emit("progress", message)
                if error:
                    stream_error = error
                    emit("cli_error", error)
                # Plain stderr may contain credentials or raw tool output; retain neither.
                if channel == "stderr" and not messages and not error:
                    if "model" in line.lower() and any(v in line.lower() for v in ("not found", "unavailable", "invalid", "not supported")):
                        stream_error = "The requested model is unavailable or unsupported; no fallback was attempted."
                        emit("cli_error", stream_error)
                    elif re.match(r"^\s*(error:|authentication (failed|required)|permission denied|unknown (option|argument)|usage:)", line, re.I):
                        stream_error = "CLI diagnostic: " + redact(line.strip())[:1000]
                        emit("cli_error", stream_error)
            except queue.Empty:
                pass
            if now - last_heartbeat >= heartbeat:
                emit("heartbeat", f"Still running; elapsed {(now - started) / 60:.1f} min. Terminal output can be quiet during work.")
                last_heartbeat = now
        return process.wait(), stream_error
    finally:
        if job:
            job.close()  # Also removes any grandchildren after a nominally successful CLI exit.
        elif os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()  # Gate was never released if job assignment failed.
        with contextlib.suppress(Exception):
            process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(Exception):
                stream.close()


class Supervisor:
    def __init__(self, root, config):
        self.root, self.config = Path(root).resolve(), config
        self.base = self.root / "logs/improvement-runner"
        self.status_path = self.base / "status.json"
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        self.run_dir = self.base / self.run_id
        self.state = {"run_id": self.run_id, "pid": os.getpid(), "status": "starting", "started_at": timestamp(),
                      "stage": None, "attempt": 0, "completed_this_run": [], "run_directory": str(self.run_dir)}

    def emit(self, kind, message):
        message = redact(str(message))
        event = {"time": timestamp(), "run_id": self.run_id, "stage": self.state["stage"], "kind": kind, "message": message}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        display = f"[{event['time']}] [{self.state['stage'] or '--'}] {message}"
        with (self.run_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(display + "\n")
        print(display, flush=True)
        self.state.update(updated_at=timestamp(), message=message)
        atomic_json(self.status_path, self.state)

    def stop_requested(self):
        path = self.base / "stop.json"
        return path.exists() and read_json(path).get("run_id") == self.run_id

    def run(self, max_steps=None, dry_run=False):
        with RunLock(self.base / "runner.lock"):
            old = read_json(self.status_path) if self.status_path.exists() else {}
            recovery = old.get("in_flight")
            if recovery:
                self.state["in_flight"] = recovery
                _, current_rows = ledger(self.root)
                changed = [sid for sid, value in recovery["before"].items() if sid != recovery["stage"] and current_rows.get(sid) != value]
                if changed:
                    raise RunnerError(f"Unverified session changed other stage statuses: {changed}. Resolve before resuming.")
                verify_protected(self.root, recovery["protected"])
            if old.get("status") == "running":
                self.emit("recovery", "Previous runner no longer owns the OS lock; inspecting recorded memory before resuming.")
            manifest = read_json(self.root / "docs/improvement/manifest.json")
            check_memory(self.root)
            self.state["status"] = "running"
            self.emit("start", "Sequential programme started. Each stage uses a fresh session and verified memory handoff.")
            started = time.monotonic()
            count = 0
            try:
                while True:
                    if self.stop_requested():
                        self.state["status"] = "stopped"
                        self.emit("stop", "Clean stop requested. No next stage started; resume with run.")
                        return 0
                    if recovery:
                        stage = next((s for s in manifest["steps"] if s["id"] == recovery["stage"]), None)
                        if stage is None:
                            raise RunnerError("The interrupted stage is missing from the manifest.")
                        if ledger(self.root)[1].get(stage["id"]) == "BLOCKED":
                            raise RunnerError(f"Interrupted stage {stage['id']} is BLOCKED. Resolve its recorded dependency first.")
                    else:
                        stage = next_stage(self.root, manifest)
                    if stage is None:
                        self.state["status"] = "complete"
                        self.emit("complete", "Every numbered stage has a verified disposition. Check separate model and forward-validation verdicts.")
                        return 0
                    if max_steps is not None and count >= max_steps:
                        self.state["status"] = "paused"
                        self.emit("pause", f"Requested step limit reached. Next stage: {stage['id']}.")
                        return 0
                    remaining = float(self.config["limits"]["max_run_hours"]) * 3600 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RunnerError("Run time limit reached; resume with run after inspecting status.")
                    self.state.update(stage=stage["id"], attempt=1)
                    actual_model, override_reason = effective_assignment(stage, self.config)
                    command = command_for(stage, self.config, self.root)
                    model_display = actual_model if not override_reason else f"{actual_model} (override of {stage['model']})"
                    self.emit("stage", f"{stage['title']} | {model_display} | {stage['effort']}")
                    if dry_run:
                        self.state["status"] = "dry_run"
                        self.emit("dry_run", f"Ready to launch stage {stage['id']}; no model called and no ledger changed.")
                        return 0
                    before = recovery["before"] if recovery else ledger(self.root)[1]
                    previous = recovery["previous_evidence"] if recovery else evidence_hash(self.root, stage["id"])
                    protected = recovery["protected"] if recovery else protected_hashes(self.root)
                    self.state["in_flight"] = {"stage": stage["id"], "before": before,
                                               "previous_evidence": previous, "protected": protected}
                    repair_reason = "Prior runner session ended without a verified successful handoff." if recovery else None
                    self.emit("checkpoint", "Saved stage baseline and ownership checkpoint before CLI launch.")
                    disposition = None
                    for attempt in range(1, int(self.config["limits"]["max_attempts_per_stage"]) + 1):
                        self.state["attempt"] = attempt
                        attempt_evidence = evidence_hash(self.root, stage["id"])
                        timeout = min(float(self.config["limits"]["max_stage_minutes"]) * 60,
                                      float(self.config["limits"]["max_run_hours"]) * 3600 - (time.monotonic() - started))
                        if timeout <= 0:
                            raise RunnerError("Run time limit reached.")
                        code, error = run_process(
                            command,
                            prompt_for(self.root, stage, self.config, attempt, repair_reason),
                            self.root,
                            timeout,
                            float(self.config["limits"]["heartbeat_seconds"]),
                            self.emit,
                            performance_environment(self.config),
                        )
                        if code or error:
                            raise RunnerError(f"CLI failed for stage {stage['id']} (exit {code}). {error or 'Inspect stage progress; no automatic retry or substitution.'}")
                        verify_protected(self.root, protected)
                        try:
                            disposition = validate_completion(self.root, manifest, stage["id"], before, attempt_evidence, code, error)
                            break
                        except RunnerError as exc:
                            _, current = ledger(self.root)
                            changed = any(current.get(sid) != value for sid, value in before.items() if sid != stage["id"])
                            if current.get(stage["id"]) in {"BLOCKED", "DEFERRED_DATA"} or changed or attempt >= int(self.config["limits"]["max_attempts_per_stage"]) or self.stop_requested():
                                raise
                            repair_reason = str(exc)
                            self.emit("repair", f"Own-stage verification failed; bounded repair attempt follows: {exc}")
                    self.state.pop("in_flight", None)
                    recovery = None
                    self.state["completed_this_run"].append({"stage": stage["id"], "disposition": disposition})
                    self.emit("verified", f"Stage {stage['id']} verified: {disposition}.")
                    count += 1
            except KeyboardInterrupt:
                self.state["status"] = "interrupted"
                self.emit("interrupt", "Interrupted; owned CLI process tree stopped. Inspect memory, then run to resume.")
                return 130
            except (RunnerError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                self.state["status"] = "failed"
                self.emit("failure", str(exc))
                return 1


def show_status(base):
    path = base / "status.json"
    if not path.exists():
        print("No automatic run recorded.")
        return None
    state = read_json(path)
    state["runner_lock_held"] = lock_is_held(base / "runner.lock")
    if state["status"] == "running" and not state["runner_lock_held"]:
        state["effective_status"] = "interrupted (no active runner lock)"
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return state


def adopt_recovery_configuration(root, base):
    """Adopt reviewed supervisor/config changes only when stage evidence is untouched."""
    status_path = base / "status.json"
    if lock_is_held(base / "runner.lock"):
        raise RunnerError("Cannot reconfigure while a runner owns the project.")
    if not status_path.exists():
        raise RunnerError("No failed run is available for recovery reconfiguration.")
    state = read_json(status_path)
    recovery = state.get("in_flight")
    if state.get("status") != "failed" or not recovery:
        raise RunnerError("Recovery reconfiguration requires a failed run with an in-flight stage.")
    _, rows = ledger(root)
    if rows != recovery["before"] or evidence_hash(root, recovery["stage"]) != recovery["previous_evidence"]:
        raise RunnerError("Stage state or evidence changed; refusing to adopt a new supervisor configuration.")
    check_memory(root)
    recovery["protected"] = protected_hashes(root)
    state.update(updated_at=timestamp(), message="Reviewed runtime configuration adopted; in-flight stage baseline is unchanged.")
    atomic_json(status_path, state)
    print(state["message"])


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_child":
        return gated_child(json.loads(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop", "watch", "doctor", "reconfigure"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = root / "logs/improvement-runner"
    try:
        if args.action == "status":
            show_status(base)
        elif args.action == "reconfigure":
            config_load(args.config or root / "docs/improvement/runner.json")
            adopt_recovery_configuration(root, base)
        elif args.action == "stop":
            path = base / "status.json"
            if not path.exists() or not lock_is_held(base / "runner.lock"):
                print("No active runner. Nothing to stop.")
                return 0
            atomic_json(base / "stop.json", {"run_id": read_json(path)["run_id"], "requested_at": timestamp()})
            print("Stop requested: the current stage will finish, then no next stage will launch.")
        elif args.action == "watch":
            position, run_id = 0, None
            while True:
                path = base / "status.json"
                if not path.exists():
                    print("No automatic run recorded.")
                    return 0
                state = read_json(path)
                if state["run_id"] != run_id:
                    position, run_id = 0, state["run_id"]
                log = Path(state["run_directory"]) / "progress.log"
                if log.exists():
                    with log.open(encoding="utf-8") as stream:
                        stream.seek(position)
                        print(stream.read(), end="", flush=True)
                        position = stream.tell()
                if state["status"] != "running" or not lock_is_held(base / "runner.lock"):
                    return 0
                time.sleep(1)
        else:
            config = config_load(args.config or root / "docs/improvement/runner.json")
            if args.action == "doctor":
                manifest = read_json(root / "docs/improvement/manifest.json")
                for step in manifest["steps"]:
                    command_for(step, config, root)
                check_memory(root)
                next_stage(root, manifest)
                print("PASS: native CLI paths, configuration, stage order and shared memory. Model access requires a live CLI call.")
                print("OpenRouter key: " + ("present" if os.environ.get("OPENROUTER_API_KEY") else "absent; optional hosted trial unavailable"))
            else:
                if args.max_steps is not None and args.max_steps < 1:
                    raise RunnerError("--max-steps must be positive.")
                return Supervisor(root, config).run(args.max_steps, args.dry_run)
    except KeyboardInterrupt:
        return 130
    except (RunnerError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print("Runner error: " + redact(str(exc)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
