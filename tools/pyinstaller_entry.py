"""Frozen-build entry point for PyInstaller.

Imports ``app.gui`` as a normal package submodule instead of passing
``app/gui.py`` straight to PyInstaller as the entry script. That matters for
two reasons:

* PyInstaller's static analysis walks the *import graph* starting from the
  entry script, so importing ``app.gui`` here is what makes it discover (and
  bundle) ``mash_reid``, ``config``, ``theme``, ``i18n``, and everything
  those pull in (torch, ultralytics, opencv, easyocr, ...).
* PyInstaller flattens only the entry script itself to the bundle root;
  regular imported modules keep their package-relative path. So
  ``app/gui.py``'s own ``__file__``-based path logic (used to find
  ``assets/`` and to build ``sys.path``) still resolves correctly once it is
  reached via a real ``app.gui`` import rather than being the entry script.

Run with:  python tools/pyinstaller_entry.py
"""

from __future__ import annotations

import logging
import sys

# Heavy dependencies the app imports lazily, the first time detection runs.
# That laziness is what keeps startup fast, but it also means a packaging
# mistake in any of them stays invisible until a user clicks Process --
# long after the build looked healthy. ``--selftest`` forces each one to
# import so CI can prove the bundle is complete right after building it.
_SELFTEST_MODULES = ("torch", "torchvision", "ultralytics", "easyocr", "cv2", "scipy.optimize")


def _selftest() -> int:
    import importlib

    failed = []
    for name in _SELFTEST_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
            print(f"FAIL  {name}: {exc}")
        else:
            print(f"ok    {name}")
    if failed:
        print(f"\n{len(failed)} bundled dependency/ies failed to import.")
        return 1
    print("\nAll bundled dependencies imported successfully.")
    return 0


from app.gui import main

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())

    try:
        main()
    except Exception:
        logging.getLogger("mash_reid.gui").exception("Startup failed")
        raise
