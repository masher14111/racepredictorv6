@echo off
REM ============================================================================
REM  Race Predictor - Open Dashboard Only
REM  Starts the Streamlit dashboard (ui/app.py) at http://localhost:8501 and
REM  opens it in your browser. No scraping, no updates - it shows whatever the
REM  last daily run produced. If it is already running, just opens the browser.
REM  Keep this window open while you use the dashboard; close it to stop.
REM  Add --dry-run to print the command without running it.
REM  Set RP_NO_BROWSER=1 to start the server without opening a browser.
REM ============================================================================
setlocal
title Race Predictor - Dashboard
set "HERE=%~dp0"
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
set "URL=http://localhost:8501"
set "RUN="
if /i "%~1"=="--dry-run" set "RUN=echo [dry-run]"
set "RC=0"

if not exist "%PY%" goto :no_venv

REM Already running? Then just show it instead of starting a second copy.
powershell -NoProfile -Command "try{(New-Object Net.Sockets.TcpClient('127.0.0.1',8501)).Close();exit 0}catch{exit 1}" >nul 2>&1
if not errorlevel 1 goto :already_running
REM The probe leaves ERRORLEVEL=1 when nothing is listening; clear it so it
REM cannot be mistaken for a dashboard failure below.
ver >nul

echo Starting the dashboard at %URL%
echo Your browser opens automatically as soon as it is ready.
echo Keep this window open while you use it. Close it, or press Ctrl+C, to stop.
echo.

REM --server.headless stops Streamlit's first-run e-mail prompt from blocking the
REM window, so the browser is opened here instead, once the port answers.
if defined RUN goto :launch
if defined RP_NO_BROWSER goto :launch
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 90;$i++){try{(New-Object Net.Sockets.TcpClient('127.0.0.1',8501)).Close();Start-Process '%URL%';break}catch{Start-Sleep -Seconds 1}}"

REM --server.address 127.0.0.1 keeps the dashboard private to this PC (its
REM Settings page shows your config) and avoids a Windows Firewall prompt.
REM To reach it from another device on your network, delete that one option.
:launch
%RUN% "%PY%" -m streamlit run ui/app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 set "RC=1"
echo.
echo Dashboard stopped.
goto :end

:already_running
echo The dashboard is already running - opening %URL%
if defined RUN goto :end
if defined RP_NO_BROWSER goto :end
start "" "%URL%"
exit /b 0

:no_venv
set "RC=1"
echo   [ERROR] Python environment not found: "%PY%"
echo   Double-click "0 - First-Time Setup.bat" first.

:end
echo.
if not defined RP_NOPAUSE pause
exit /b %RC%
