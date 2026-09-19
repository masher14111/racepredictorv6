# Automatic runner verification
Date: 2026-09-19 (Europe/Dublin). Scope: runner, launchers and expanded hosted-text work orders.

## Verified
- Codex CLI 0.155.0 and Claude Code 2.1.268 are installed and signed in using existing subscriptions.
- Live response probes succeeded for gpt-6-astra, claude-sonnet-5, claude-opus-5 and claude-fable-5-1.
- A real Codex shell write/read passed through the owned Windows process wrapper in workspace-write mode.
- A real Claude Sonnet shell write/read passed through the same wrapper with configured tool permissions.
- Smoke files were confined to ignored logs/improvement-runner/setup/.
- 17 focused runner tests passed in the final run (5.54 seconds).
- Tests cover duplicate locks, child/grandchild timeout cleanup, blocked stdin, structured errors and exit-zero false completion.
- Tests cover restart after a failed CLI writes DONE, skipped stages, protected-file changes, repair limits and clean stop.
- Both PowerShell files parsed without errors; the actual PowerShell dry-run selected step01 / Astra / Medium.
- Runner doctor, compact-memory checker, prompt line limits, local documentation links and new utility whitespace passed.
- An independent review found no remaining launch blocker after the identified fixes.
- All active memory files remain under 200 lines; generated portable context is 156 lines at setup completion.

## Corrections made during verification
- Codex --ignore-user-config belongs after exec in this installed CLI.
- Isolated Windows CLI configuration needs explicit windows.sandbox=elevated to permit the intended workspace write.
- Audit steps now pause on substantial earlier-stage defects instead of contradicting the supervisor's status rules.
- Failed/interrupted workers cannot advance the programme solely by writing DONE into shared memory.
- Worker changes to the supervisor, validator, manifest or active runner configuration prevent acceptance.

## Limits
- Runner completion records engineering dispositions, not predictive profitability or completed future observations.
- No hosted DeepSeek inference was run during setup; no OpenRouter key was found in the inspected local settings.
- Step15 must implement and test its persistent US$5 / 1,200 request budget before making hosted calls.
- Full predictor tests/training are delegated to the appropriate numbered stages, not claimed by these runner checks.
- Live runner status and run IDs are in logs/improvement-runner/status.json; stage evidence is in memory/improvement/.

## Reproduce offline checks
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/tools -p test_improvement_runner.py -v
.\.venv\Scripts\python.exe tools/improvement_runner.py doctor
.\.venv\Scripts\python.exe tools/improvement_memory.py check
```
