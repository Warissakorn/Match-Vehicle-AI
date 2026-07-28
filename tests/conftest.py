"""Make ``config``, the ``mash_reid`` package, and ``tools/`` importable during tests."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
# tools/ holds standalone launcher helpers that run before the package is
# installed, so they live outside src/ and need their own path entry.
sys.path.insert(0, os.path.join(_ROOT, "tools"))
