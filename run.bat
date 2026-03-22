@echo off
title LivingRPS
color 0A

echo ============================================================
echo  LivingRPS Launcher
echo ============================================================
echo.

:: Check Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found. Please install Python and add it to your PATH.
    echo         https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Check dependencies
echo [*] Checking dependencies...
python -c "import pygame, torch, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Missing dependencies detected. Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    echo.
)

:: Check trained model
if not exist "models\rps_agent.pth" (
    echo [*] No trained model found. Running training first...
    echo     This may take a minute...
    echo.
    python train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Training failed.
        pause
        exit /b 1
    )
    echo.
)

echo [*] Starting LivingRPS...
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The game exited with an error (code %errorlevel%).
    pause
)