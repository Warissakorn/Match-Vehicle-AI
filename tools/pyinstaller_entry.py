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

from app.gui import main

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("mash_reid.gui").exception("Startup failed")
        raise
