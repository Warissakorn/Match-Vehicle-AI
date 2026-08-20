# PyInstaller build spec for the Windows executable.
#
# Build with (from the project root):
#   pyinstaller tools/windows.spec
#
# See tools/pyinstaller_entry.py for why the entry point imports app.gui
# as a package rather than running app/gui.py directly.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

ROOT = Path(SPECPATH).parent

# torch / ultralytics / easyocr all ship non-code assets (config yaml files,
# character sets, etc.) that plain import-graph analysis misses -- pull each
# in wholesale via its PyInstaller hook rather than guessing at file lists.
datas = [(str(ROOT / "assets"), "assets")]
binaries = []
hiddenimports = []
for pkg in ("torch", "torchvision", "ultralytics", "easyocr", "cv2", "scipy"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [str(ROOT / "tools" / "pyinstaller_entry.py")],
    pathex=[str(ROOT), str(ROOT / "app"), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MatchVehicleAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Keep the console: run.bat deliberately holds it open on a crash so a
    # user who double-clicked the app can still read the traceback, and the
    # frozen exe should behave the same way.
    console=True,
    icon=str(ROOT / "assets" / "icon.ico"),
)
