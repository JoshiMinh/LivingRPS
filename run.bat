@echo off
setlocal enabledelayedexpansion

:menu
cls
echo ==========================================
echo           LivingRPS CLI Menu
echo ==========================================
echo [1] Run Game
echo [2] Train Model
echo [3] Install Dependencies
echo [4] Exit
echo ==========================================
set /p opt="Pick an option (1-4): "

if "%opt%"=="1" goto run_game
if "%opt%"=="2" goto train
if "%opt%"=="3" goto install
if "%opt%"=="4" goto exit
echo Invalid option, please try again.
pause
goto menu

:run_game
echo Starting LivingRPS...
if not exist "src\main.py" (
    echo [ERROR] src\main.py not found.
    pause
    goto menu
)
cd src
python main.py
cd ..
pause
goto menu

:train
echo Starting Training...
if not exist "src\train.py" (
    echo [ERROR] src\train.py not found.
    pause
    goto menu
)
cd src
python train.py
cd ..
pause
goto menu

:install
echo Installing Dependencies...
pip install -r requirements.txt
pause
goto menu

:exit
echo Exiting...
exit /b