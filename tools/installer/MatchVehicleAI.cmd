@echo off
REM Launcher for an installed (thin-installer) Match-Vehicle-AI.
REM
REM Installed under Program Files, the app runs from *source* rather than
REM from a PyInstaller bundle, so app_paths.is_frozen() is False and every
REM user-writable default falls back to its source-checkout location --
REM models/ beside the code, settings.json next to config.py, a CWD-relative
REM logs/. All three sit inside the install directory, which a normal user
REM cannot write to, so they would fail silently and two users on one machine
REM would share one settings file.
REM
REM MASH_DATA_DIR is checked ahead of both the frozen and the source defaults
REM (see app_paths._user_data_root_or_none), so pointing it at the per-user
REM location here fixes all of them at once with no change to the app. The
REM path is deliberately the exact one a frozen build picks for itself, which
REM also means an installed build and a downloaded .zip build share their
REM downloaded model weights instead of fetching ~110 MB twice.

setlocal enableextensions
cd /d "%~dp0"

if not defined MASH_DATA_DIR set "MASH_DATA_DIR=%LOCALAPPDATA%\MatchVehicleAI"

set "VPY_CONSOLE=%~dp0env\Scripts\python.exe"

if not exist "%VPY_CONSOLE%" (
    echo Match-Vehicle-AI is not fully installed -- its Python environment is missing.
    echo Re-run the installer to repair it.
    pause
    exit /b 1
)

REM Start on the console interpreter so a startup traceback has somewhere to
REM go. run.bat holds its window open on a non-zero exit for exactly this
REM reason, and the frozen build keeps console=True for it too -- a launcher
REM that hid the error would make the one failure users actually report
REM ("the window flashes and disappears") unreportable again.
"%VPY_CONSOLE%" app\gui.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo ---------------------------------------------------------------
    echo Match-Vehicle-AI exited with an error ^(code %EXITCODE%^).
    echo The details are printed above. A copy is also in
    echo   %MASH_DATA_DIR%\logs
    echo ---------------------------------------------------------------
    echo.
    pause
)
endlocal
exit /b %EXITCODE%
