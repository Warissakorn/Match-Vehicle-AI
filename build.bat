@echo off
REM Package Match-Vehicle-AI into a standalone Windows .exe (PyInstaller).
REM
REM Run run.bat at least once first -- this reuses that same .venv (with
REM torch/ultralytics/easyocr/opencv already installed, GPU build detected
REM and all) rather than resolving dependencies a second time.
REM
REM Output lands in dist\MatchVehicleAI\MatchVehicleAI.exe. That whole
REM dist\MatchVehicleAI\ folder is the deliverable -- it is not one file,
REM everything next to the .exe is required. Copy the folder, not just the
REM .exe, when moving it to another machine.

setlocal enableextensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VPY%" (
    echo Error: %VENV_DIR% not found. Run run.bat once first to create it
    echo and install dependencies, then run build.bat again.
    exit /b 1
)

echo Installing PyInstaller ...
"%VPY%" -m pip install -r requirements-build.txt
if errorlevel 1 (
    echo Error: could not install PyInstaller.
    exit /b 1
)

echo Building MatchVehicleAI.exe ...
"%VPY%" -m PyInstaller --noconfirm packaging\match_vehicle_ai.spec
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo ---------------------------------------------------------------
    echo Build failed (code %EXITCODE%^). Common causes:
    echo   - a dependency changed since the last "pip install -r requirements.txt"
    echo   - antivirus quarantining a file mid-build -- add an exclusion
    echo     for this folder and retry
    echo See PyInstaller's own output above for the actual error.
    echo ---------------------------------------------------------------
    pause
    exit /b %EXITCODE%
)

echo.
echo Done: dist\MatchVehicleAI\MatchVehicleAI.exe
echo Copy the whole dist\MatchVehicleAI\ folder to distribute it.
endlocal
