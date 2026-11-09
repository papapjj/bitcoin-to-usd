@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-ui.ps1" %*
exit /b %ERRORLEVEL%
