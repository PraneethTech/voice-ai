@echo off
setlocal
cd /d %~dp0
echo ========================================================
echo        tobi v2 - Personal Offline Voice Assistant
echo ========================================================
echo.

if not exist v2 (
  echo [ERROR] 'v2' folder not found. Please run this script from the repository root.
  pause
  exit /b 1
)

cd v2
call setup.bat
