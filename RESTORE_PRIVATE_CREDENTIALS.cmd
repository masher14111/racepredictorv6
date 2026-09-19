@echo off
setlocal
pwsh.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0tools\restore_private_credentials.ps1"
set "restore_exit=%errorlevel%"
echo.
pause
exit /b %restore_exit%
