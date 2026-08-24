"""Which build of the app is this?

Releases are stamped by the build workflow: it writes the pushed git tag into
``assets/_build_version.txt`` before running PyInstaller, and since ``assets/``
is already collected into the bundle the frozen app can simply read its own
copy back. That makes the version visible three places -- the window title,
the startup log line, and ``MatchVehicleAI.exe --version`` -- which is how a
downloaded zip can be traced back to the exact release that produced it
instead of "one of the builds".

Unstamped runs (a source checkout, an unbuilt working tree) report ``"dev"``;
``MASH_VERSION`` overrides everything, which is what tests and local builds
use. Pure stdlib, safe to import anywhere.
"""

from __future__ import annotations

import os
import re

_ENV_VERSION = "MASH_VERSION"
_STAMP_FILENAME = "_build_version.txt"

_FALLBACK = "dev"

# A stamp is a short identifier -- a tag like "v1.2.3" or a CI dev-build
# number. Anything longer/stranger than that means we found a corrupt or
# foreign file rather than a real stamp, and "dev" is the honest answer.
_SANE_STAMP = re.compile(r"[A-Za-z0-9._+~-]{1,40}")


def _stamp_candidates() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    # Frozen layout: <bundle>/mash_reid/version.py with assets collected at
    # <bundle>/assets. Source layout: <root>/src/mash_reid/version.py with
    # assets at <root>/assets. Both are "up two levels", so one relative walk
    # covers them; the extra candidate keeps a relocated install working.
    return [
        os.path.join(os.path.dirname(here), "assets", _STAMP_FILENAME),
        os.path.join(here, os.pardir, os.pardir, "assets", _STAMP_FILENAME),
    ]


def get_version() -> str:
    """The build version string: a git tag, a CI dev-build id, or ``"dev"``."""
    env = os.environ.get(_ENV_VERSION)
    if env and _SANE_STAMP.fullmatch(env.strip()):
        return env.strip()

    for candidate in _stamp_candidates():
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                text = fh.read(64).strip()
        except OSError:
            continue
        if text and _SANE_STAMP.fullmatch(text):
            return text
    return _FALLBACK


def is_release() -> bool:
    """True when this build carries a real release stamp (not ``"dev"``)."""
    return get_version() != _FALLBACK
