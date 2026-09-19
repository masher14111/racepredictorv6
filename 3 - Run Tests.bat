@echo off
REM ============================================================================
REM  Race Predictor - Run Tests   (health check)
REM  Runs the full test suite: python -m pytest -q
REM  Expect about 2000 tests, all passing, in 2-3 minutes. The tests never touch
REM  the network, never send Telegram alerts, and never write to the real data\.
REM  Add --dry-run to print the command without running it.
REM ============================================================================
setlocal
title Race Predictor - Run Tests
set "HERE=%~dp0"
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
set "RUN="
if /i "%~1"=="--dry-run" set "RUN=echo [dry-run]"
set "RC=0"

if not exist "%PY%" goto :no_venv

echo Running the full test suite - about 2000 tests, usually 2-3 minutes.
echo.
%RUN% "%PY%" -m pytest -q
if errorlevel 1 goto :tests_failed

echo.
echo ============================================================
echo  RESULT: ALL TESTS PASSED
echo ============================================================
goto :end

:tests_failed
set "RC=1"
echo.
echo ============================================================
echo  RESULT: SOME TESTS FAILED - scroll up for the details.
echo  After a "git pull", run "0 - First-Time Setup.bat" first.
echo ============================================================
goto :end

:no_venv
set "RC=1"
echo   [ERROR] Python environment not found: "%PY%"
echo   Double-click "0 - First-Time Setup.bat" first.

:end
echo.
if not defined RP_NOPAUSE pause
exit /b %RC%
