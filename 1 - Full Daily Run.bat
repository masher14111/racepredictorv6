@echo off
REM ============================================================================
REM  Race Predictor - Full Daily Run   (the everyday button)
REM  Exactly the three QUICKSTART.md section 2 commands, in order:
REM    1. python -m scripts.refresh                      scrape, normalize, predict
REM    2. python -m scripts.daily_paper_loop --no-scrape  gate, tickets, settle, reports
REM    3. the dashboard                                  via "2 - Open Dashboard Only.bat"
REM  BoyleSports is scraped by an anti-detect browser tier (Botasaurus) that
REM  opens a real browser window for ~1 min. Firecrawl is wired but OFF - it
REM  cannot reach either book. See config.yaml.
REM  Idempotent: safe to run again later the same day for another price poll.
REM  PAPER ONLY - nothing here can stake real money.
REM  Add --dry-run to print the commands without running them.
REM ============================================================================
setlocal
title Race Predictor - Full Daily Run
set "HERE=%~dp0"
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
set "RUN="
if /i "%~1"=="--dry-run" set "RUN=echo [dry-run]"
set "RC=0"

echo ============================================================
echo  Race Predictor - Full Daily Run
echo  Paper only - nothing here can stake real money.
echo  The model is rated NO-GO, so "0 candidates" is the expected,
echo  correct result - not a fault.
echo ============================================================
if not defined FIRECRAWL_API_KEY goto :fc_ready
echo  Firecrawl tier: key found, but the tier is OFF in config.yaml -
echo  both books returned 403 to Firecrawl when tested on 2026-09-18.
:fc_ready
echo.

if not exist "%PY%" goto :no_venv

echo [1/3] Update - scrape today's racing, normalize, predict. Usually 3-6 minutes.
echo       A BROWSER WINDOW WILL OPEN for about a minute - that is the BoyleSports
echo       anti-detect scraper clearing Cloudflare. Leave it alone, it closes itself.
echo       Early Cloudflare warnings before it opens are normal - the quick HTTP
echo       tries are expected to fail first. The "predict" step then goes
echo       QUIET for about 2 minutes
echo       while it scores every runner. That is not a hang - please leave it running.
%RUN% "%PY%" -m scripts.refresh
if errorlevel 1 goto :refresh_failed

echo.
echo [2/3] Daily paper loop - gate, paper tickets, settle, reports.
%RUN% "%PY%" -m scripts.daily_paper_loop --no-scrape
if errorlevel 1 echo   WARNING: the paper loop reported a problem - see above. Continuing to the dashboard.

echo.
echo [3/3] Dashboard
call "%HERE%2 - Open Dashboard Only.bat" %*
exit /b %errorlevel%

:refresh_failed
set "RC=1"
echo.
echo   [ERROR] The update step failed - read the messages above.
echo   If the bookmaker sites are blocking the scrapers, try again later.
echo   If you switched the Firecrawl tier on and see 402 or budget exhausted,
echo   you are out of Firecrawl credits - top up or lower its page budget.
echo   See QUICKSTART.md section 6 for common fixes.
goto :end

:no_venv
set "RC=1"
echo   [ERROR] Python environment not found: "%PY%"
echo   Double-click "0 - First-Time Setup.bat" first.

:end
echo.
if not defined RP_NOPAUSE pause
exit /b %RC%
