# Sample frames

Put your still frames here for a quick try-out, e.g.:

```
samples/
  pointA/
    A_20260723_101500.jpg
    A_20260723_101512.jpg
  pointB/
    B_20260723_101745.jpg
    B_20260723_101801.jpg
```

Each image may contain several vehicles — they are detected automatically.

The capture time is read from the filename (see `TIMESTAMP_PATTERNS` in
`config.py`). Supported examples:

- `A_20260723_101530.jpg`  → 2026-07-23 10:15:30
- `2026-07-23_10-15-30.jpg`
- `20260723101530.jpg`

If no timestamp is found in the name, EXIF `DateTimeOriginal` is used, then the
file's modification time as a last resort.

Image files in this folder are git-ignored (only this README is tracked).
