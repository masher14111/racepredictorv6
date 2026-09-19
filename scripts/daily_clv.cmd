@echo off
REM Daily live-CLV job: refresh predictions (scrape -> normalize -> build -> predict
REM -> settle), then log today's value picks at the best board price and settle/report
REM matured ones. Registered with Windows Task Scheduler (see scripts/install_schedule.cmd).
REM Output is appended to logs/daily_clv.log.

set ROOT=%~dp0..
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d "%ROOT%"

echo ============================================================ >> logs\daily_clv.log
echo [%DATE% %TIME%] daily_clv start >> logs\daily_clv.log

"%PY%" -m scripts.refresh        >> logs\daily_clv.log 2>&1
"%PY%" -m scripts.clv_live run   >> logs\daily_clv.log 2>&1

echo [%DATE% %TIME%] daily_clv done >> logs\daily_clv.log
