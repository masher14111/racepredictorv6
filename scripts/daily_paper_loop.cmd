@echo off
REM Daily paper-betting loop (Stage 6): refresh today's card (scrape -> normalize
REM -> predict, which also takes the day's first live price snapshot), then
REM capture -> PASS-by-default gate -> issue paper tickets -> settle open
REM tickets -> recompute forward metrics -> reports. `--no-scrape` on the second
REM step avoids re-scraping a card `scripts.refresh` just captured.
REM Idempotent: safe to re-run any number of times the same day. Intended to be
REM registered with Windows Task Scheduler (see the schtasks command in
REM HANDOFF.md's Stage 6 section) - not installed automatically by this repo.
REM Output is appended to logs/daily_paper_loop.log.

set ROOT=%~dp0..
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d "%ROOT%"

echo ============================================================ >> logs\daily_paper_loop.log
echo [%DATE% %TIME%] daily_paper_loop start >> logs\daily_paper_loop.log

"%PY%" -m scripts.refresh                       >> logs\daily_paper_loop.log 2>&1
"%PY%" -m scripts.daily_paper_loop --no-scrape   >> logs\daily_paper_loop.log 2>&1

echo [%DATE% %TIME%] daily_paper_loop done >> logs\daily_paper_loop.log
