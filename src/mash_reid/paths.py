"""Where the app reads and writes files, aware of running as a frozen build.

Compiling with PyInstaller changes what ``__file__``-based path arithmetic
means, in a way that silently breaks two things if left alone:

- ``settings.json`` would end up inside the executable's bundle rather than
  next to it -- for ``--onefile`` that bundle is a temp folder extracted
  fresh and discarded on every run, so every launch would look like first
  run and forget the user's folders and thresholds.
- Downloaded model weights (``models/``) would land in the same
  temp-and-discarded place, so every launch would re-download instead of
  reusing what was already fetched.

Bundled read-only assets (``assets/icon.ico``) have the opposite need: they
must be found *inside* the bundle, since nothing copies them next to the exe.

Both functions fall back to the project root when running from source
(``python app/gui.py``, the normal dev/test path), so nothing changes there.
"""

from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    """True when running as a PyInstaller-built executable."""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> str:
    """Folder to read bundled assets from (icon, ...)."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return _project_root()


def writable_dir() -> str:
    """Folder to write persistent state to (settings.json, models/)."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return _project_root()


def _project_root() -> str:
    # src/mash_reid/paths.py -> mash_reid -> src -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
