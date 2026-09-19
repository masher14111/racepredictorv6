@echo off
setlocal
pwsh.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0tools\start_improvements.ps1" -Action stop
set "runner_exit=%errorlevel%"
echo.
pause
exit /b %runner_exit%
