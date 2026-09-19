@echo off
setlocal
pwsh.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0tools\set_openrouter_key.ps1" 
set "runner_exit=%errorlevel%"
echo.
pause
exit /b %runner_exit%
