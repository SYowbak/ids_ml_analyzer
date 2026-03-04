@echo off
setlocal
title IDS Launcher
cd /d "%~dp0"

echo [1/2] Preparing environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

echo [2/2] Starting application...
".venv\Scripts\python.exe" start_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Startup failed.
    pause
)
endlocal
