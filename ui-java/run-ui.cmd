@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
java -jar "%~dp0build\SelfGrowingTraderControlV193.jar" "%PROJECT_ROOT%"
exit /b %ERRORLEVEL%
