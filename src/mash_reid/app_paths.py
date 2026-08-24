"""Resolve user-writable data locations for the app.

A frozen build (PyInstaller) used to write *beside the executable*: model
weights landed in ``models/``, ``settings.json`` next to the bundled
``config.py``, logs in a CWD-relative ``logs/``. Dropped into ``Program
Files`` -- or any install directory the user can't write to -- every one of
those writes failed silently, and two users sharing one machine shared one
settings file. Frozen builds therefore root everything user-writable under
one data directory instead:

    Windows:  %%LOCALAPPDATA%%\\MatchVehicleAI
    macOS:    ~/Library/Application Support/MatchVehicleAI
    Linux:    ${XDG_DATA_HOME:-~/.local/share}/MatchVehicleAI

Source checkouts keep the historical behaviour *exactly* -- models in
``<project>/models``, settings next to ``config.py``, logs CWD-relative --
so development, tests, and existing installs see no difference. Every
location honours an environment override (``MASH_DATA_DIR`` for the root,
plus the existing ``MASH_MODELS_DIR``) so portable setups and tests can
redirect the lot.

Pure stdlib, no imports from the rest of the package beyond ``config`` --
this module is consulted before torch/easyocr exist and must stay cheap to
import.
"""

from __future__ import annotations

import os
import sys

# ``config`` is a project-root module with no imports of its own (constants +
# dataclasses), so importing it here is cheap and cannot cycle. It is only
# *needed* for the unfrozen legacy locations.
import config  # noqa: E402

_APP_DIR_NAME = "MatchVehicleAI"
_ENV_DATA_DIR = "MASH_DATA_DIR"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle (or any frozen build)."""
    return bool(getattr(sys, "frozen", False))


def _platform_data_root() -> str:
    if sys.platform == "win32":
        # The launcher-created variable; fall back to the documented default
        # for exotic shells where it's somehow unset.
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, _APP_DIR_NAME)
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", _APP_DIR_NAME)
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return os.path.join(xdg, _APP_DIR_NAME)


def _user_data_root_or_none() -> str | None:
    """The explicit/frozen data root, or ``None`` for a plain source run.

    ``None`` is meaningful: it tells each accessor below to fall back to its
    own *historical* location, which is what keeps source runs
    byte-compatible with where things were always written.
    """
    env = os.environ.get(_ENV_DATA_DIR)
    if env:
        return env
    if is_frozen():
        return _platform_data_root()
    return None


def data_root() -> str:
    """The directory all user-writable app state lives under.

    Source runs return the project root, which is what keeps every other
    default below byte-compatible with where things were written before this
    module existed. ``MASH_DATA_DIR`` overrides in either mode.
    """
    root = _user_data_root_or_none()
    if root is not None:
        return root
    # src/mash_reid/app_paths.py -> mash_reid -> src -> project root --
    # the same arithmetic model_manager used for its weights dir.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def models_dir() -> str:
    """Folder for downloaded YOLO weights (``$MASH_MODELS_DIR`` overrides)."""
    env = os.environ.get("MASH_MODELS_DIR")
    if env:
        return env
    return os.path.join(data_root(), "models")


def _legacy_config_dir() -> str:
    """Directory holding ``config.py`` -- where small state files always lived."""
    return os.path.dirname(os.path.abspath(config.__file__))


def settings_file() -> str:
    """``settings.json``: beside ``config.py`` from source, the data root frozen."""
    root = _user_data_root_or_none()
    if root is not None:
        return os.path.join(root, "settings.json")
    return os.path.join(_legacy_config_dir(), config.DEFAULT_SETTINGS_FILE)


def ocr_patterns_file() -> str:
    """User-saved OCR time-pattern collection (see ``timestamp_ocr``)."""
    root = _user_data_root_or_none()
    if root is not None:
        return os.path.join(root, "ocr_patterns.json")
    return os.path.join(_legacy_config_dir(), config.DEFAULT_OCR_PATTERNS_FILE)


def logs_dir() -> str:
    """Rotating-log directory.

    Relative ``logs`` from source (unchanged historical behaviour: relative
    to wherever the user launched from), absolute under the data root when
    frozen -- a CWD-relative log dir is meaningless for a double-clicked
    exe, whose CWD varies by shortcut.
    """
    root = _user_data_root_or_none()
    if root is not None:
        return os.path.join(root, "logs")
    from mash_reid import logging_setup
    return logging_setup.DEFAULT_LOG_DIR


def easyocr_model_dir() -> str | None:
    """Where easyocr stores its detection/recognition models, or ``None``.

    ``None`` keeps easyocr's own default (``~/.EasyOCR``), which is what
    plain source runs have always used. Frozen builds (or an explicit
    ``MASH_DATA_DIR``) redirect into the data root so the one-time download
    survives app updates and doesn't depend on a writable home-relative
    dotfolder.
    """
    root = _user_data_root_or_none()
    if root is not None:
        return os.path.join(root, "easyocr")
    return None


def torch_home() -> str | None:
    """``TORCH_HOME`` value for hub/torchvision weight caches, or ``None``.

    ``None`` leaves torch's default (``~/.cache/torch``), preserving source
    behaviour. Frozen builds point at the data root: the embedder's ResNet50
    checkpoint (~98 MB) is downloaded on first use, and without this it went
    to the user-home cache -- outside our control, wiped independently of
    the app.
    """
    root = _user_data_root_or_none()
    if root is not None:
        return os.path.join(root, "torch")
    return None


def apply_runtime_env() -> None:
    """Export env vars the deferred third-party imports read at call time.

    Must run before torch/ultralytics are first imported (they snapshot
    these paths), so every entry point calls it right at startup. Idempotent;
    a value already set in the environment always wins.
    """
    resolved_torch_home = torch_home()
    if resolved_torch_home and not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = resolved_torch_home
    # ultralytics writes its settings.yaml next to the executable's user dir
    # by default; give it a writable, app-owned location when frozen.
    if is_frozen() and not os.environ.get("YOLO_CONFIG_DIR"):
        os.environ["YOLO_CONFIG_DIR"] = os.path.join(data_root(), "ultralytics")
