@echo off
REM Build the installed app's Python environment. Run by Setup (and by the
REM workflow's install test) with the install directory as %1.
REM
REM Exists as its own script rather than living in the .iss because it needs
REM environment variables set around the uv calls, which Inno's Exec() cannot
REM do, and because keeping it here means the exact sequence Setup performs
REM can be run and debugged by hand on a developer machine.

setlocal enableextensions
set "APP=%~1"
if "%APP%"=="" set "APP=%~dp0..\.."

REM Same pin as bootstrap.py and the build workflow. It cannot be read out of
REM bootstrap.py here, because the interpreter that would read it is the very
REM thing this line is about to download; tests/test_installer_pins.py fails
REM the build if the three ever disagree.
set "PYTHON_VERSION=3.14"

set "UV=%APP%\uv.exe"
set "ENV_DIR=%APP%\env"

REM Keep uv's downloaded interpreters inside the install directory. uv's
REM default is a shared per-user location (%LOCALAPPDATA%\uv\python), which
REM the uninstaller has no knowledge of and would leave behind as a few
REM hundred megabytes of orphaned CPython after a clean removal.
set "UV_PYTHON_INSTALL_DIR=%APP%\python"

echo Downloading Python %PYTHON_VERSION% ...
"%UV%" python install %PYTHON_VERSION%
if errorlevel 1 (
    echo Error: could not download Python. Check the internet connection and retry.
    exit /b 1
)

echo Creating the environment ...
"%UV%" venv --python %PYTHON_VERSION% "%ENV_DIR%"
if errorlevel 1 (
    echo Error: could not create the Python environment.
    exit /b 1
)

REM From here the work is Python's: a real interpreter now exists, so the GPU
REM probing, the pinned installs and the post-install self-test all live in
REM bootstrap.py where they can be unit-tested instead of in batch.
"%ENV_DIR%\Scripts\python.exe" "%APP%\tools\installer\bootstrap.py" ^
    --uv "%UV%" --python "%ENV_DIR%\Scripts\python.exe" --root "%APP%"
if errorlevel 1 (
    echo Error: setting up the dependencies failed.
    exit /b 1
)

echo Done.
endlocal
exit /b 0
