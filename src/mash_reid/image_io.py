"""Unicode-safe image read/write, shared by every module that touches images.

``cv2.imread``/``cv2.imwrite`` silently fail (return ``None``/``False``, no
exception) when the path contains non-ASCII characters (e.g. Thai) on Windows —
this made frame detection find nothing and video extraction write no files.
Reading/writing bytes through Python's ``open`` and decoding/encoding with
``cv2.imdecode``/``cv2.imencode`` handles Unicode paths on every platform.

``cv2`` is passed in rather than imported here so this module stays import-free
of OpenCV, matching the deferred-import pattern used elsewhere.
"""

from __future__ import annotations

import numpy as np


def imread_unicode(cv2, path: str):
    """Read an image, supporting non-ASCII paths; returns a BGR array or None."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(cv2, path: str, image, ext: str = "jpg") -> None:
    """Write ``image`` to ``path``, supporting non-ASCII paths; raises on failure."""
    ok, buf = cv2.imencode(f".{ext.lstrip('.')}", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as .{ext} (unsupported format?)")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
