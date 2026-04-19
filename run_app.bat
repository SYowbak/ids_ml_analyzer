@echo off
setlocal
chcp 65001 >nul
title IDS Launcher
cd /d "%~dp0"

echo [1/2] Підготовка середовища...
if not exist ".venv\Scripts\python.exe" (
    echo [ПОМИЛКА] Віртуальне середовище не знайдено!
    pause
    exit /b 1
)

echo [2/2] Запуск застосунку...
".venv\Scripts\python.exe" start_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ПОМИЛКА] Помилка запуску.
    pause
)
endlocal
