@echo off
REM Launcher for the Match-Vehicle-AI vehicle Re-ID GUI (Windows).
REM
REM On first run it creates a local virtual environment, installs the
REM dependencies, then opens the Tkinter desktop app. Later runs reuse the
REM environment and only reinstall when requirements.txt has changed.
REM Just double-click run.bat, or run it from a command prompt.

setlocal enableextensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "STAMP=%VENV_DIR%\.requirements.sha256"

REM Pick a Python launcher: prefer the 'py' launcher, fall back to python.
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    set "PYLAUNCH=py -3"
) else (
    set "PYLAUNCH=python"
)

REM Create the virtual environment if missing.
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR% ...
    %PYLAUNCH% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: could not create virtual environment. Install Python 3.10+ and retry.
        exit /b 1
    )
)

set "VPY=%VENV_DIR%\Scripts\python.exe"

REM Compute the current requirements.txt hash.
for /f "usebackq delims=" %%H in (`certutil -hashfile requirements.txt SHA256 ^| findstr /r "^[0-9a-f]"`) do set "REQ_HASH=%%H"

REM Read the stored hash, if any.
set "OLD_HASH="
if exist "%STAMP%" set /p OLD_HASH=<"%STAMP%"

REM Install / update dependencies only when requirements changed.
REM A plain "pip install torch" commonly resolves to a CPU-only wheel on
REM Windows even with a real NVIDIA GPU present -- the #1 cause of "GPU not
REM detected" reports, see the README's GPU section. tools\install_torch.py
REM detects the driver, installs the matching CUDA build first so that
REM requirements.txt's torch>=2.0.0 is already satisfied, then verifies and
REM reports the result. It never fails: no GPU, an unsupported driver, or a
REM failed download all fall through to the CPU build installed below.
REM NOTE: REM comments must stay outside the parenthesized IF block below --
REM a REM line containing unbalanced parentheses inside a block confuses
REM cmd.exe's parser.
if not "%REQ_HASH%"=="%OLD_HASH%" (
    echo Installing dependencies ^(first run or requirements changed^) ...
    "%VPY%" -m pip install --upgrade pip

    "%VPY%" tools\install_torch.py

    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: dependency installation failed.
        exit /b 1
    )
    > "%STAMP%" echo %REQ_HASH%
)

echo Launching Match-Vehicle-AI GUI ...
"%VPY%" app\gui.py %*
endlocal
