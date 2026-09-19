# GitHub checkpoint — 2026-09-19

Destination: private repository https://github.com/masher14111/racepredictorv6.

This is a snapshot of the user's current project, including existing UI/application changes,
the original improvement programme, and the automatic daily-paper follow-up.
At snapshot preparation, stages 19 and 20 were verified DONE; stage 21 was still running
its implementation/checks. A safe stop after the current stage has been requested.
The user's instruction to keep it working refers to preserving local app credentials;
credentials and local data have not been deleted by the upload.
Stages 22-24 remain pending. Do not interpret this upload as completed implementation,
model acceptance or prospective validation. Consult current STATE.md for later progress.

The GitHub publishing history starts at this snapshot because old local Git history contained
a proxy credential. Original local history remains on the pre-upload branch and is not pushed.
Do not merge or push that legacy history to this repository without sanitising it first.

Source code, tests, documentation, memory and report evidence are included. At the user's explicit
request, this PRIVATE repository also includes config.local.yaml and .env with the current app
credentials. Do not make this repository public. Local environments/caches, database backups and
generated model binaries remain local according to .gitignore; a clone still needs its data/models.

On a new Windows installation, run RESTORE_PRIVATE_CREDENTIALS.cmd once to install the two
predictor API keys from .env into that Windows user's environment, then reopen the application
terminal. config.local.yaml is already read automatically for proxy/service configuration.
The helper does not display keys or contact a provider. Existing local credentials remain unchanged.
Credential inclusion does not create a new hosted allowance: the same cumulative $5/1,200-request
cap still applies. Preserve the existing budget ledger when moving the application to another PC;
do not start a new empty ledger or run concurrent independent copies against that allowance.

Portable START_AUTOMATIC_IMPROVEMENTS.cmd, WATCH_IMPROVEMENTS.cmd,
STOP_AFTER_CURRENT_STEP.cmd, CHECK_IMPROVEMENTS.cmd and SET_OPENROUTER_KEY.cmd
are included in the repository root. The existing outer-folder launchers remain available locally.
Do not start another coder while the local supervisor still owns its run lock.
