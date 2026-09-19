@echo off
REM ============================================================================
REM  Race Predictor - First-Time Setup
REM  Run once on a new PC, or after moving/copying the project folder.
REM  Safe to re-run: every step is skipped or is a no-op when already done.
REM    1. Python virtual environment (.venv) - built from requirements.lock.txt
REM    2. Playwright Chromium - the headless browser the scrapers drive
REM    3. SQLite schema migration (data\races.db)
REM    4. Config check (config.yaml + config.local.yaml + Firecrawl key)
REM  Add --dry-run to print the commands without running them.
REM ============================================================================
setlocal
title Race Predictor - First-Time Setup
set "HERE=%~dp0"
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
set "RUN="
if /i "%~1"=="--dry-run" set "RUN=echo [dry-run]"
set "RC=0"

echo ============================================================
echo  Race Predictor - First-Time Setup
echo  Run once on a new PC, or after moving the project folder.
echo  Safe to re-run at any time.
echo ============================================================
echo.

echo [1/4] Python environment
if not exist "%PY%" goto :build_venv
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto :broken_venv
echo   Virtual environment found and working.
goto :step2

:build_venv
echo   No virtual environment found - building one from requirements.lock.txt.
echo   This downloads about 1 GB of packages and takes several minutes.
where py >nul 2>&1
if errorlevel 1 goto :venv_with_python
%RUN% py -3.14 -m venv "%HERE%.venv"
goto :venv_created
:venv_with_python
REM "where py" left ERRORLEVEL=1; clear it so only the venv step is judged.
ver >nul
%RUN% python -m venv "%HERE%.venv"
:venv_created
if errorlevel 1 goto :no_python
%RUN% "%PY%" -m pip install --disable-pip-version-check -r "%HERE%requirements.lock.txt"
if errorlevel 1 goto :failed

:step2
echo.
echo [2/4] Browser for the scrapers - Playwright Chromium, one-time download
%RUN% "%PY%" -m playwright install chromium
if errorlevel 1 goto :failed

echo.
echo [3/4] Database schema
%RUN% "%PY%" -m utils.storage migrate
if errorlevel 1 goto :failed
%RUN% "%PY%" -m utils.storage version
if errorlevel 1 goto :failed

echo.
echo [4/4] Configuration
if not exist "%HERE%config.local.yaml" echo   NOTE: config.local.yaml is missing. Proxy, Timeform and Telegram secrets go there - see FINALSETUP.md section 3.
%RUN% "%PY%" -c "from utils.config_loader import get_config; print('config OK', bool(get_config()))"
if errorlevel 1 goto :failed
if defined FIRECRAWL_API_KEY goto :fc_ok
echo   Firecrawl tier: not configured - and nothing to do. It is disabled in
echo   config.yaml because both books returned 403 to Firecrawl on 2026-09-18.
goto :fc_done
:fc_ok
echo   Firecrawl tier: key found, but disabled in config.yaml - both books
echo   returned 403 to Firecrawl when it was tested on 2026-09-18.
:fc_done

echo.
echo ============================================================
echo  Setup complete.
echo  Next: double-click "1 - Full Daily Run.bat"
echo ============================================================
goto :end

:broken_venv
set "RC=1"
echo.
echo   [ERROR] The .venv folder exists but its Python will not start.
echo   This happens when the project folder is moved or Python is reinstalled,
echo   because virtual environments cannot be relocated.
echo   Fix: delete the ".venv" folder, then run this setup again.
goto :end

:no_python
set "RC=1"
echo.
echo   [ERROR] Could not create the virtual environment.
echo   Install Python 3.14 from python.org, then run this setup again.
goto :end

:failed
set "RC=1"
echo.
echo   [ERROR] A setup step failed - read the messages above.
echo   See QUICKSTART.md section 6 for common fixes.

:end
echo.
if not defined RP_NOPAUSE pause
exit /b %RC%
