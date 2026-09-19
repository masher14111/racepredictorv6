# Automatic improvement runner
Updated: 2026-09-19. This runs24 stages through your installed CLIs: original01-18 plus the
authorized [daily paper follow-up19-24](DAILY_PAPER_FOLLOWUP.md). Completed stages are not rerun.

## Start and resume
Double-click START_AUTOMATIC_IMPROVEMENTS.cmd in the outer Race Predictor v4 folder.
The terminal shows the current step, selected model/effort, agent progress and elapsed-time updates.
It uses your existing signed-in Codex and Claude subscriptions; those account usage limits still apply.
Each stage starts a fresh CLI session and reads the current shared memory.
The runner continues automatically after a successful, verified handoff.
A restart resumes from the saved stage evidence, rather than beginning the plan again.

## Controls
- WATCH_IMPROVEMENTS.cmd shows progress without starting a second worker.
- STOP_AFTER_CURRENT_STEP.cmd requests a pause after the current stage has finished.
- Ctrl+C in the running terminal interrupts immediately; inspect the saved stage before resuming.
- CHECK_IMPROVEMENTS.cmd checks the local runner environment and reports readiness.
- SET_OPENROUTER_KEY.cmd accepts the key privately in a local terminal; never paste a key into chat.
The computer must stay awake. This is a local sequential job, not a cloud service or a scheduled daily task.
No recurring schedule, startup service or real-money execution is created.

## What runs
The original18 assignments remain intact; stages19-23 use Sonnet5 High and24 uses Opus5 xhigh.
Exact identifiers: gpt-6-astra, claude-sonnet-5, claude-opus-5 and claude-fable-5-1.
Each invocation sets its model and thinking effort; the header text is not the mechanism.
Unavailable models, account limits and authentication failures stop with a readable error.
The runner does not silently switch to another model.
Current runtime override: Codex-assigned stages use Claude Sonnet 5 because Codex reported its usage limit before step01.
The terminal and worker prompt disclose this override; original recommended assignments remain in the prompt manifest.
Remove runtime_overrides from runner.json after Codex access returns if you want later Codex-assigned stages to use Astra.
Original NASC files are unchanged; this runner uses the predictor's own manifest and memory checks.

## Completion checks
A successful process exit alone cannot complete a stage.
The runner checks the memory validator, the executed stage's disposition and its evidence file.
It rejects skipped stages and unrelated stage-status changes.
A deferred optional experiment stays disabled and needs a concrete resume condition.
Required data/correctness blockers stop the job; a pending scientific result is not a passed gate.
An audit finding needing substantial earlier-stage repairs pauses at the audit and identifies the owner.
It does not silently alter completed stages; supervised repair is followed by a fresh audit.
Only one runner may own this project at a time. Do not manually run another writer in the same files.
No automatic commits, pushes, resets, stashes or staging of your existing dirty changes.

## Limits and permissions
Settings live in [runner.json](runner.json).
Defaults: 12 hours per run, 3 hours per stage and at most two attempts at a stage within a run.
A limit or failed check preserves evidence and stops or bounds repair attempts; restarting is explicit.
Codex uses the workspace-write sandbox with no interactive escalation.
It explicitly selects the configured Windows sandbox and allows networking for authorized source/model access.
The sandbox setting matters: omitting it under isolated CLI configuration produced a read-only smoke run.
Claude pre-allows the listed local coding tools with acceptEdits; it does not use bypassPermissions.
Claude's tool allowances are not an operating-system sandbox.
Workers are instructed to preserve unrelated work and the paper-only model/forward gates.
The child process tree is owned by the runner and terminated on interruption or timeout.
Workers must wait for long-running tools inside their CLI session. A final response ends that
session and terminates its remaining child processes; scheduled wakeups cannot resume it.
The supervisor prompt explicitly forbids yielding a final response while a background job runs.

## Performance
The runner exposes a two-worker CPU budget on this 6-core/12-thread computer. Six workers were
benchmarked and rejected because each heavy test worker loaded its own multi-gigabyte race matrix,
causing memory pressure and slower completion. Two workers made the verified feature/tools test group
1.51x faster (21.86s to 14.50s). Workers use pytest-xdist for independent targeted test directories;
the complete repository suite remains serial because its process, port and shared-file tests are not
xdist-safe. Small targeted tests also remain serial.
Independent read-only checks may run concurrently, but operations that write the same file,
database, cache or artifact remain serialized. Coding stages still run one at a time because
they share the checkout and stage ledger; parallel writers could corrupt evidence or overwrite edits.
Set `performance.cpu_workers` in runner.json to tune the cap.

## Hosted DeepSeek trial
Step14 fixes extraction and preserves dated source text.
Step15 benchmarks repaired regex, local Qwen and deepseek/deepseek-v4.1-flash through OpenRouter.
Hermes remains optional. Step16 measures whether validated text features improve forecasts.
Stage19 now integrates this existing backend into daily shadow extraction; total programme length is24.
The same persistent spending/request cap includes the earlier benchmark and all follow-up cycles.
The hosted benchmark must implement and test a persistent US$5 / 1,200 request cap before live calls.
Caps include retries and output/reasoning-token reservations, and survive restarts.
The coding CLIs' subscription usage is separate from that hosted inference cap.
A missing key, credit or reviewed dataset yields an explicit deferred result, never invented scores.
Your key is stored in the Windows user environment and is read locally; the setup helper makes no API call.
Current saved model and forward-validation verdicts remain separate from engineering completion.

## Files and manual commands
Runtime state, progress and per-stage traces are stored under logs/improvement-runner/ (ignored by Git).
Treat local CLI traces as private: they can contain source excerpts and model output.
Shared project progress remains in memory/improvement/STATE.md and HANDOFF.md.
Generated CHAT_CONTEXT.md and NEXT_PROMPT.md remain available for a manual fresh chat.
Run from the nested Git repository:
```powershell
.\.venv\Scripts\python.exe tools/improvement_runner.py doctor
.\.venv\Scripts\python.exe tools/improvement_runner.py run --dry-run
.\.venv\Scripts\python.exe tools/improvement_runner.py run
.\.venv\Scripts\python.exe tools/improvement_runner.py status
.\.venv\Scripts\python.exe tools/improvement_runner.py watch
.\.venv\Scripts\python.exe tools/improvement_runner.py stop
```

## References
- [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex Windows sandbox](https://learn.chatgpt.com/docs/windows/windows-sandbox)
- [Claude programmatic execution](https://code.claude.com/docs/en/headless)
- [DeepSeek V4.1 Flash model/providers](https://openrouter.ai/deepseek/deepseek-v4.1-flash)
- [Existing prompt sequence](START_HERE.md)
- [Runner verification](RUNNER_VERIFICATION.md)
