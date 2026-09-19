"""Offline supervisor checks. Run directly with unittest to avoid application fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/improvement_runner.py"
SPEC = importlib.util.spec_from_file_location("improvement_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return {
        "schema_version": 1,
        "codex": {"executable": sys.executable, "model": "gpt-6-astra", "sandbox": "workspace-write", "approval_policy": "never"},
        "claude": {"executable": sys.executable, "models": {"Claude Sonnet 5": "claude-sonnet-5"},
                   "permission_mode": "acceptEdits", "allowed_tools": ["Bash", "Read", "Edit", "Write", "Glob", "Grep"]},
        "limits": {"max_stage_minutes": 1, "max_run_hours": 1, "max_attempts_per_stage": 2, "heartbeat_seconds": 0.1},
        "hosted_trial": {"model": "deepseek/example", "max_cost_usd": 5, "max_requests": 1200},
    }


def step(sid="01", model="Codex — GPT-6 Astra", effort="High"):
    return {"id": sid, "title": "A bounded test stage", "model": model, "effort": effort,
            "depends": [], "file": f"docs/improvement/prompts/{sid}.md"}


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "memory/improvement/stages").mkdir(parents=True)
        (self.root / "docs/improvement/prompts").mkdir(parents=True)
        self.manifest = {"steps": [step("01"), step("02")]}
        runner.atomic_json(self.root / "docs/improvement/manifest.json", self.manifest)
        for s in self.manifest["steps"]:
            (self.root / s["file"]).write_text("Stage instructions", encoding="utf-8")
        self.write_state("01", "PENDING", "PENDING")

    def write_state(self, next_id, first, second):
        (self.root / "memory/improvement/STATE.md").write_text(
            f"Next step: {next_id}\nActive step: none\n| 01 | First | {first} | - |\n| 02 | Second | {second} | - |\n", encoding="utf-8")

    def evidence(self, sid, text="Objective: actual work\nChecks: verified\n"):
        (self.root / "memory/improvement/stages" / f"{sid}.md").write_text(text, encoding="utf-8")

    def test_exit_zero_is_not_completion(self):
        with self.assertRaisesRegex(runner.RunnerError, "Exit zero is not completion"):
            runner.validate_completion(self.root, self.manifest, "01", {"01": "PENDING", "02": "PENDING"}, None, 0, None)

    def test_complete_requires_fresh_evidence_and_memory_check(self):
        before = {"01": "PENDING", "02": "PENDING"}
        self.write_state("02", "DONE", "PENDING")
        with self.assertRaisesRegex(runner.RunnerError, "fresh stage evidence"):
            runner.validate_completion(self.root, self.manifest, "01", before, None, 0, None)
        self.evidence("01")
        with patch.object(runner, "check_memory") as check:
            self.assertEqual(runner.validate_completion(self.root, self.manifest, "01", before, None, 0, None), "DONE")
            check.assert_called_once_with(self.root)
        with self.assertRaisesRegex(runner.RunnerError, "fresh stage evidence"):
            runner.validate_completion(self.root, self.manifest, "01", before, runner.evidence_hash(self.root, "01"), 0, None)

    def test_other_stage_changes_and_pending_leaps_rejected(self):
        self.write_state("none", "DONE", "DONE")
        with self.assertRaisesRegex(runner.RunnerError, "other stage statuses"):
            runner.validate_completion(self.root, self.manifest, "01", {"01": "PENDING", "02": "PENDING"}, None, 0, None)
        self.write_state("none", "DONE", "PENDING")
        with self.assertRaisesRegex(runner.RunnerError, "Refusing skipped stages"):
            runner.next_stage(self.root, self.manifest)

    def test_valid_deferred_optional_disposition(self):
        self.manifest["steps"][1]["review_of"] = ["01"]
        self.write_state("02", "DEFERRED_DATA", "PENDING")
        self.evidence("01", "Candidate enabled: no\nResume condition: obtain dated data.\n")
        with patch.object(runner, "check_memory"):
            self.assertEqual(runner.validate_completion(self.root, self.manifest, "01", {"01": "PENDING", "02": "PENDING"}, None, 0, None), "DEFERRED_DATA")

    def test_duplicate_process_rejected_then_lock_reusable(self):
        path = self.root / "runner.lock"
        probe = (
            "import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('runner',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "lock=m.RunLock(sys.argv[1]);lock.__enter__();lock.__exit__()"
        )
        with runner.RunLock(path):
            result = subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Another runner", result.stderr)
        result = subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)

    def test_exact_model_effort_and_permission_arguments(self):
        settings = config()
        settings["codex"].update(windows_sandbox="elevated", network_access=True)
        command = runner.command_for(step(), settings, self.root)
        self.assertIn("--ignore-user-config", command)
        self.assertGreater(command.index("--ignore-user-config"), command.index("exec"))
        self.assertIn("sandbox_workspace_write.network_access=true", command)
        if os.name == "nt":
            self.assertIn('windows.sandbox="elevated"', command)
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertIn('approval_policy="never"', command)
        self.assertIn("workspace-write", command)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        claude = runner.command_for(step(model="Claude Sonnet 5", effort="Extra high (xhigh)"), config(), self.root)
        self.assertEqual(claude[claude.index("--model") + 1], "claude-sonnet-5")
        self.assertEqual(claude[claude.index("--effort") + 1], "xhigh")
        self.assertNotIn("--dangerously-skip-permissions", claude)
        with self.assertRaisesRegex(runner.RunnerError, "No exact configured model"):
            runner.command_for(step(model="Claude Unknown"), config(), self.root)

    def test_explicit_runtime_override_uses_and_discloses_claude(self):
        settings = config()
        settings["runtime_overrides"] = {
            "Codex — GPT-6 Astra": {"model": "Claude Sonnet 5", "reason": "quota unavailable"}
        }
        command = runner.command_for(step(), settings, self.root)
        self.assertEqual(command[command.index("--model") + 1], "claude-sonnet-5")
        prompt = runner.prompt_for(self.root, self.manifest["steps"][0], settings, 1)
        self.assertIn("explicit supervisor override", prompt)
        self.assertIn("quota unavailable", prompt)

    def test_parallel_test_budget_is_bounded_and_disclosed(self):
        settings = config()
        settings["performance"] = {"cpu_workers": 2, "parallel_tests": True}
        prompt = runner.prompt_for(self.root, self.manifest["steps"][0], settings, 1)
        self.assertIn("up to 2 local CPU workers", prompt)
        self.assertIn("pytest -n auto --dist loadscope", prompt)
        self.assertIn("complete repository suite serially", prompt)
        self.assertEqual(runner.performance_environment(settings)["PYTEST_XDIST_AUTO_NUM_WORKERS"], "2")

    def test_invalid_parallel_test_budget_is_rejected(self):
        for bad in (0, 33, 2.5, "6"):
            settings = config()
            settings["performance"] = {"cpu_workers": bad, "parallel_tests": True}
            path = self.root / "config.json"
            runner.atomic_json(path, settings)
            with self.assertRaisesRegex(runner.RunnerError, "cpu_workers"):
                runner.config_load(path)

    def test_reconfigure_failed_untouched_stage_updates_protected_fence(self):
        base = self.root / "logs/improvement-runner"
        before = {"01": "PENDING", "02": "PENDING"}
        runner.atomic_json(base / "status.json", {
            "status": "failed", "in_flight": {"stage": "01", "before": before,
            "previous_evidence": None, "protected": {"old": "hash"}}}
        )
        with patch.object(runner, "check_memory"):
            runner.adopt_recovery_configuration(self.root, base)
        adopted = runner.read_json(base / "status.json")
        self.assertEqual(adopted["in_flight"]["protected"], runner.protected_hashes(self.root))

    def test_json_errors_and_tool_redaction(self):
        text = json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "SECRET"}},
            {"type": "text", "text": "Checking evidence"}]}})
        messages, error = runner.stream_summary(text)
        self.assertEqual(messages, ["Tool: Bash", "Checking evidence"])
        self.assertIsNone(error)
        for event in ({"type": "error", "message": "model unavailable"},
                      {"type": "turn.failed", "error": {"message": "quota exceeded"}},
                      {"type": "result", "is_error": True, "subtype": "error_max_turns"}):
            self.assertIsNotNone(runner.stream_summary(json.dumps(event))[1])

    def test_process_streams_unicode_prompt_and_reports_error_even_exit_zero(self):
        script = "import sys,json; p=sys.stdin.read(); print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':p}})); print(json.dumps({'type':'error','message':'test failure'}))"
        events = []
        code, error = runner.run_process([sys.executable, "-c", script], "Race € — ✓", self.root, 10, .1,
                                         lambda kind, message: events.append((kind, message)))
        self.assertEqual(code, 0)
        self.assertIn("test failure", error)
        self.assertTrue(any("Race € — ✓" in message for _, message in events))

    def test_timeout_terminates_child_tree_and_never_hangs_on_stdin(self):
        pid_file = self.root / "child.pid"
        script = (
            "import subprocess,sys,time,pathlib;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));time.sleep(60)"
        )
        started = time.monotonic()
        with self.assertRaisesRegex(runner.RunnerError, "time limit"):
            runner.run_process([sys.executable, "-c", script], "x" * 200000, self.root, 4, .1, lambda *args: None)
        self.assertLess(time.monotonic() - started, 12)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                code = wintypes.DWORD()
                kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                kernel.CloseHandle(handle)
                self.assertNotEqual(code.value, 259, "grandchild remains active")

    def test_stop_after_verified_stage_never_starts_next(self):
        supervisor = runner.Supervisor(self.root, config())
        calls = []
        def fake_run(*args):
            calls.append(True)
            self.write_state("02", "DONE", "PENDING")
            self.evidence("01")
            runner.atomic_json(supervisor.base / "stop.json", {"run_id": supervisor.run_id})
            return 0, None
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", side_effect=fake_run):
            self.assertEqual(supervisor.run(), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(runner.read_json(supervisor.status_path)["status"], "stopped")
        self.assertEqual(runner.ledger(self.root)[1]["02"], "PENDING")

    def test_cli_failure_stops_without_repair_or_fallback(self):
        supervisor = runner.Supervisor(self.root, config())
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", return_value=(1, "model unavailable")) as execute:
            self.assertEqual(supervisor.run(), 1)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(runner.read_json(supervisor.status_path)["status"], "failed")

    def test_dry_run_does_not_execute_or_change_ledger(self):
        original = runner.ledger(self.root)
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process") as execute:
            self.assertEqual(runner.Supervisor(self.root, config()).run(dry_run=True), 0)
            execute.assert_not_called()
        self.assertEqual(runner.ledger(self.root), original)

    def test_required_stage_cannot_defer(self):
        self.write_state("02", "DEFERRED_DATA", "PENDING")
        with self.assertRaisesRegex(runner.RunnerError, "Required stages cannot be skipped"):
            runner.next_stage(self.root, self.manifest)

    def test_completed_programme_resumes_at_appended_stage_without_replaying(self):
        self.manifest = {"steps": [step(f"{i:02d}") for i in range(1, 25)]}
        self.manifest["steps"][16]["review_of"] = ["16"]
        text = "Next step: 19\nActive step: none\n"
        for i in range(1, 25):
            status = "DEFERRED_DATA" if i == 16 else "DONE" if i <= 18 else "PENDING"
            text += f"| {i:02d} | Scope | {status} | - |\n"
        (self.root / "memory/improvement/STATE.md").write_text(text, encoding="utf-8")
        self.assertEqual(runner.next_stage(self.root, self.manifest)["id"], "19")
        (self.root / "memory/improvement/STATE.md").write_text(text.replace("Next step: 19", "Next step: 20"), encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "Refusing skipped stages"):
            runner.next_stage(self.root, self.manifest)

    def test_worker_cannot_rewrite_protected_acceptance_prompt(self):
        expected = runner.protected_hashes(self.root)
        (self.root / self.manifest["steps"][0]["file"]).write_text("weakened acceptance", encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "Protected supervisor/validation"):
            runner.verify_protected(self.root, expected)

    def test_followup_envelope_preserves_shared_hosted_budget(self):
        prompt = runner.prompt_for(self.root, self.manifest["steps"][0], config(), 1)
        self.assertIn("follow-up stages 19-24", prompt)
        self.assertIn("including shadow integration", prompt)
        self.assertIn("never reset spend", prompt)
        self.assertIn("total request cap 1200", prompt)

    def test_fractional_attempt_limit_rejected(self):
        value = config()
        value["limits"]["max_attempts_per_stage"] = .5
        path = self.root / "config.json"
        runner.atomic_json(path, value)
        with self.assertRaisesRegex(runner.RunnerError, "integer"):
            runner.config_load(path)

    def test_failed_cli_cannot_advance_on_restart_despite_done_ledger(self):
        def failed_run(*args):
            self.write_state("02", "DONE", "PENDING")
            self.evidence("01", "Unverified earlier evidence")
            return 1, "API error"
        first = runner.Supervisor(self.root, config())
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", side_effect=failed_run):
            self.assertEqual(first.run(), 1)
        self.assertEqual(runner.read_json(first.status_path)["in_flight"]["stage"], "01")
        calls = []
        def recovered_run(command, prompt, *args):
            calls.append(prompt)
            self.evidence("01", "Fresh recovery checks were run")
            return 0, None
        second = runner.Supervisor(self.root, config())
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", side_effect=recovered_run):
            self.assertEqual(second.run(max_steps=1), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("stage 01", calls[0])
        self.assertIn("Prior runner session", calls[0])
        self.assertNotIn("in_flight", runner.read_json(second.status_path))
        self.assertEqual(runner.ledger(self.root)[1]["02"], "PENDING")

    def test_worker_cannot_weaken_validator(self):
        (self.root / "tools").mkdir()
        target = self.root / "tools/improvement_memory.py"
        target.write_text("honest validator", encoding="utf-8")
        def fake_run(*args):
            target.write_text("always pass", encoding="utf-8")
            self.write_state("02", "DONE", "PENDING")
            self.evidence("01")
            return 0, None
        supervisor = runner.Supervisor(self.root, config())
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", side_effect=fake_run) as execute:
            self.assertEqual(supervisor.run(), 1)
            self.assertEqual(execute.call_count, 1)
        self.assertIn("Protected supervisor/validation", runner.read_json(supervisor.status_path)["message"])

    def test_repair_prompt_contains_actual_validation_failure(self):
        prompts = []
        def fake_run(command, prompt, *args):
            prompts.append(prompt)
            if len(prompts) == 2:
                self.write_state("02", "DONE", "PENDING")
                self.evidence("01")
            return 0, None
        with patch.object(runner, "check_memory"), patch.object(runner, "run_process", side_effect=fake_run):
            self.assertEqual(runner.Supervisor(self.root, config()).run(max_steps=1), 0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Exit zero is not completion", prompts[1])


if __name__ == "__main__":
    unittest.main()
