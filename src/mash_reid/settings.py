"""Persist GUI settings (folders, model choice, sliders, device, ...) between
runs, so reopening the app doesn't mean reconfiguring everything from
scratch.

Only a fixed whitelist of keys is ever read back from disk -- an old,
foreign, or hand-edited settings file can't inject arbitrary state into the
app this way, and any value of the wrong type (or a corrupt/missing file)
just falls back to the caller's own defaults rather than crashing startup.
Pure stdlib (``json``) -- no new dependency for something this small.
"""

from __future__ import annotations

import json
import logging
import os

import config

log = logging.getLogger(__name__)

# Whitelisted keys and their expected Python type. load() drops anything
# outside this set (unknown key) or with a mismatched type (corrupt file,
# or a foreign file that happens to share this filename).
_SCHEMA: dict[str, type] = {
    "dir_a": str,
    "dir_b": str,
    "model_key": str,
    "device": str,
    "threshold": float,
    "det_conf": float,
    "min_travel": float,
    "max_travel": float,
    "use_gate": bool,
    "one_to_one": bool,
    "extract_interval": float,
    "last_video_dir": str,
    "cluster_same_point": bool,
    "timeline_samples": int,
    "ocr_time_pattern": str,
    "ui_theme": str,
    "step1_open": bool,
    "step2_open": bool,
    "step3_open": bool,
    "split_sash": int,
}


def default_path() -> str:
    """``<project root>/settings.json`` -- next to ``config.py``."""
    root = os.path.dirname(os.path.abspath(config.__file__))
    return os.path.join(root, config.DEFAULT_SETTINGS_FILE)


def load(path: str | None = None) -> dict:
    """Load settings, filtered to the whitelist.

    Returns ``{}`` for a missing file, corrupt JSON, or a JSON value that
    isn't an object -- callers should treat that identically to "no saved
    settings yet" and fall back to their own defaults. Never raises.
    """
    path = path or default_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        log.warning("Settings file %s unreadable, using defaults", path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        log.warning("Settings file %s did not contain a JSON object, ignoring", path)
        return {}

    result: dict = {}
    for key, expected_type in _SCHEMA.items():
        if key not in raw:
            continue
        value = raw[key]
        # bool is a subclass of int in Python, so an explicit check is
        # needed in both directions to avoid a stray 0/1 silently passing
        # as a bool-typed setting, or a real bool passing as a float one.
        if expected_type is bool:
            if not isinstance(value, bool):
                continue
        elif expected_type is int:
            # Same bool-is-an-int trap as below, in the other direction: a
            # stray `true` must not be accepted as an int-typed setting.
            if isinstance(value, bool) or not isinstance(value, int):
                continue
        elif expected_type is float:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                value = float(value)
            if not isinstance(value, float):
                continue
        elif not isinstance(value, expected_type):
            continue
        result[key] = value
    return result


def save(values: dict, path: str | None = None) -> None:
    """Write ``values`` (filtered to the whitelist) to ``path`` as JSON.

    Never raises -- a failed save (read-only filesystem, permissions, ...)
    is logged and otherwise silently skipped. Losing remembered settings on
    the next launch is a far smaller problem than crashing on window close.
    """
    path = path or default_path()
    filtered = {k: v for k, v in values.items() if k in _SCHEMA}
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(filtered, fh, indent=2, sort_keys=True)
    except Exception:
        log.warning("Could not write settings to %s", path, exc_info=True)
