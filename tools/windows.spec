# PyInstaller build spec for the Windows build.
#
# Build with (from the project root):
#   pyinstaller tools/windows.spec --noconfirm
#
# Produces a *folder* (dist/MatchVehicleAI/) containing MatchVehicleAI.exe
# next to its libraries -- deliberately not a single self-contained .exe.
#
# Why onedir and not onefile
# --------------------------
# A onefile .exe is really a zip with a bootloader glued on front: every
# launch unpacks the *entire* archive into a fresh %TEMP%\_MEIxxxxxx folder
# before the first line of Python runs, then deletes it on exit. This app
# bundles torch, ultralytics, easyocr and opencv -- on the order of a
# gigabyte of DLLs -- so that unpack means writing a gigabyte to disk, and
# having the antivirus scan every freshly written DLL, on *each* start. The
# result is a startup measured in minutes, and launches that appear to fail
# outright when the extraction trips security software or a locked-down
# %TEMP%.
#
# In onedir the libraries are already on disk in their final location. The
# OS loader maps each DLL on demand, nothing is extracted, and the antivirus
# scans the files once (on install) rather than on every run. Startup drops
# to seconds. The cost is that the app ships as a folder rather than one
# file, which the workflow packages as a .zip.
#
# What makes the window appear fast
# ---------------------------------
# Onedir only removes the *extraction* cost. What keeps startup fast after
# that is that no module in ``mash_reid`` imports torch, ultralytics, cv2 or
# easyocr at module level -- they are all function-local imports, pulled in
# the first time detection actually runs. So the GUI opens on little more
# than tkinter + numpy, and the multi-second cost of importing torch is paid
# once, in the background worker thread, when the user clicks Process.
# Bundling those packages here does not change that: it puts the code on
# disk, it does not import it. Keep that property -- hoisting any of those
# imports to module level in mash_reid would move seconds back onto startup.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent

# torch, cv2, easyocr and scipy are covered by PyInstaller's own bundled
# hooks (via pyinstaller-hooks-contrib), which already know which binaries
# and data files each needs -- calling collect_all() on them here would
# override that with a blunter "copy everything" that also drags in test
# suites, C headers and static .lib files, inflating the build without
# fixing anything.
#
# ultralytics is the exception: it loads model/tracker definitions from
# .yaml files at runtime that no import-graph analysis can see, so its data
# files are collected explicitly.
datas = [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "app"), "app"),
    (str(ROOT / "config.py"), "."),
]
datas += collect_data_files("ultralytics")

a = Analysis(
    [str(ROOT / "tools" / "pyinstaller_entry.py")],
    pathex=[str(ROOT), str(ROOT / "app"), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    # Imported lazily inside functions, so the import-graph walk never sees
    # them; without this they would be missing from the build and the app
    # would only fail at the moment the user clicks Process.
    hiddenimports=[
        "torch",
        "torchvision",
        "ultralytics",
        "easyocr",
        "cv2",
        "scipy.optimize",
        "PIL.ImageTk",
        "psutil",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Pulled in transitively by torch/ultralytics but unused here: matplotlib
    # and pandas alone add well over a hundred megabytes to the build.
    #
    # Everything listed here must be provably unreachable at *runtime*, not
    # merely unused at import time: the build's `--selftest` only forces the
    # top-level heavy imports, so a package a bundled library reaches for
    # lazily (mid-call) would slip past CI and fail in the user's hands. That
    # is why the entries below are limited to two closed groups:
    #
    #   * test-only packages, which requirements-lock.txt installs because the
    #     build job runs pytest before packaging, and which no app code path
    #     can reach; and
    #   * matplotlib's exclusive dependency satellites -- matplotlib itself is
    #     already excluded above, so nothing else in the graph imports them.
    #     fonttools alone is tens of megabytes.
    #
    # Deliberately NOT excluded, despite looking like dead weight: polars and
    # ultralytics-platform (ultralytics reaches for them while producing
    # results, i.e. after selftest has passed), sympy/networkx (torch.fx), and
    # setuptools/pkg_resources (several bundled libraries still probe it at
    # call time). Each would trade a modest size win for a failure that only
    # appears once a user clicks Process.
    excludes=[
        "matplotlib",
        "pandas",
        "PyQt5",
        "PySide2",
        "IPython",
        "notebook",
        # test-only -- installed for the build job's pytest run, never imported
        # by the application itself
        "pytest",
        "_pytest",
        "pluggy",
        "iniconfig",
        "pygments",
        # matplotlib-only satellites (matplotlib is excluded above)
        "fontTools",
        "contourpy",
        "kiwisolver",
        "cycler",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MatchVehicleAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep the console window. run.bat deliberately holds its console open on
    # a non-zero exit so that a user who double-clicked the app can still
    # read the traceback instead of watching a window flash and vanish; the
    # frozen build should not be harder to get a bug report out of.
    console=True,
    icon=str(ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MatchVehicleAI",
)
