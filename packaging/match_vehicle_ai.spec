# PyInstaller build spec for Match-Vehicle-AI. Build with:
#     pyinstaller --noconfirm packaging/match_vehicle_ai.spec
# (or just run build.bat, which does that inside the project's venv).
#
# --onedir, not --onefile: --onefile re-extracts the whole bundle (torch +
# ultralytics + easyocr + opencv is several hundred MB) into a temp folder on
# every single launch, which is both slow and a common false-positive trigger
# for antivirus heuristics. --onedir starts instantly and lets Windows build
# per-file trust instead of rescanning one giant blob each run.
#
# console=True on purpose, not --windowed: this project's app/gui.py already
# had one bug where a startup crash's traceback vanished with the console
# that printed it (see run.bat and "When the app won't start" in the
# README). A windowed build with no console reintroduces exactly that
# failure mode for anything that dies before logging is set up. Switch to
# windowed only once you're confident startup is solid on the target
# machine -- logs/ still captures a traceback either way, but only a console
# shows it immediately.
#
# ultralytics and easyocr both ship non-Python data files (ultralytics's
# cfg/*.yaml defaults, easyocr's character-set tables) that PyInstaller's
# import scanner won't find on its own since nothing "imports" a yaml file --
# collect_data_files pulls those in explicitly. This does NOT include the
# downloaded model weights (yolov8n.pt, the OCR recognition network, ...);
# those still fetch over the network into models/ on first run, same as
# running from source -- see model_manager.py.

import os

from PyInstaller.utils.hooks import collect_data_files

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))

datas = [
    (os.path.join(PROJECT_ROOT, "assets"), "assets"),
]
datas += collect_data_files("ultralytics")
datas += collect_data_files("easyocr")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "app", "gui.py")],
    pathex=[PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MatchVehicleAI",
    console=True,
    icon=os.path.join(PROJECT_ROOT, "assets", "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="MatchVehicleAI",
)
