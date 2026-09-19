# GitHub checkpoint — 2026-09-19

Destination: private repository https://github.com/masher14111/racepredictorv6.

This is a snapshot of the user's current project, including existing UI/application changes,
the original improvement programme, and the automatic daily-paper follow-up.
At snapshot preparation, stages 19 and 20 were verified DONE; stage 21 was still running
its implementation/checks. The user revoked the earlier safe-stop request and instructed
the coder to keep running; that stop request was cleared before this upload.
Stages 22-24 remain pending. Do not interpret this upload as completed implementation,
model acceptance or prospective validation. Consult current STATE.md for later progress.

The GitHub publishing history starts at this snapshot because old local Git history contained
a proxy credential. Original local history remains on the pre-upload branch and is not pushed.
Do not merge or push that legacy history to this repository without sanitising it first.

Source code, tests, documentation, memory and report evidence are included. Secrets, local
environments/caches, database backups and generated model binaries remain local according
to .gitignore. A clone needs its own environment, data/models and privately supplied credentials.

Portable START_AUTOMATIC_IMPROVEMENTS.cmd, WATCH_IMPROVEMENTS.cmd,
STOP_AFTER_CURRENT_STEP.cmd, CHECK_IMPROVEMENTS.cmd and SET_OPENROUTER_KEY.cmd
are included in the repository root. The existing outer-folder launchers remain available locally.
Do not start another coder while the local supervisor still owns its run lock.
