# Mash-Object-AI — Cross-Point Vehicle Re-Identification

Match the **same physical vehicle** as it passes two separate camera points
(**A** and **B**) using **visual appearance only** — no license-plate reading.
Cameras may sit at different distances and angles.

You give it two folders of timestamped still frames (extracted from your video
cameras); it detects every vehicle in every frame, turns each into an
appearance embedding, and tells you which vehicle at A is the same as which
vehicle at B.

```
frames @ A ┐
           ├─► detect vehicles (YOLO) ─► appearance embedding (Re-ID) ─┐
frames @ B ┘                                                           │
                                                                       ▼
                     cosine similarity + travel-time gate ─► ranked A→B matches
```

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Load  | `src/mash_reid/frame_loader.py` | Reads images, gets each frame's timestamp from filename / EXIF / mtime |
| Detect | `src/mash_reid/detector.py` | YOLOv8 finds cars, motorcycles, buses, trucks and crops them |
| Embed | `src/mash_reid/embedder.py` | Each crop → an L2-normalized appearance vector (ResNet50 by default) |
| Match | `src/mash_reid/matcher.py` | Cosine similarity + a travel-time gate (A must precede B), then ranks |
| Run   | `src/mash_reid/pipeline.py` | Ties it together, caches detections per folder |

The **travel-time gate** encodes physical reality: a vehicle passes A *before*
B, within a configurable window (e.g. 0–600 s). This filters out visually
similar but temporally impossible pairs.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights (YOLOv8n ≈ 6 MB, ResNet50 ≈ 100 MB) download automatically on
first run and are cached afterwards.

## Usage

### Desktop GUI (Tkinter)

```bash
python app/gui.py
```

1. Browse to the **Point A** and **Point B** frame folders.
2. Tune sliders if needed (similarity threshold, detection confidence, travel
   window).
3. Click **Process**. First run downloads the models.
4. Click a vehicle in the A gallery → its best B-matches appear on the right
   with similarity scores. Double-click any thumbnail to view the full frame
   with the bounding box.

Slider and toggle changes re-match instantly (no re-detection needed).

### Command line

```bash
python cli.py --dir-a samples/pointA --dir-b samples/pointB \
    --threshold 0.6 --max-travel 600
```

Useful flags: `--conf` (detection confidence), `--min-travel`/`--max-travel`
(seconds), `--no-time-gate`, `--one-to-one` (force a unique A↔B assignment),
`--no-cache`.

## Filename timestamp convention

The capture time is read from the filename. Supported formats (configurable in
`config.py` → `TIMESTAMP_PATTERNS`):

- `A_20260723_101530.jpg` → 2026-07-23 10:15:30
- `2026-07-23_10-15-30.jpg`
- `20260723101530.jpg`

Falls back to EXIF `DateTimeOriginal`, then file modification time.

## Tests

```bash
python -m pytest tests/
```

The tests cover similarity, temporal gating, ranking, one-to-one assignment,
and timestamp parsing — all with synthetic data, so no model or network needed.

## Configuration

All tunables live in `config.py`: vehicle classes, YOLO weights, detection
confidence, timestamp patterns, similarity threshold, and travel-time window.

## Swapping the appearance model

The default embedder is ImageNet ResNet50 — a reliable baseline. For higher
cross-angle accuracy, implement the `Embedder` interface in
`src/mash_reid/embedder.py` (e.g. a dedicated vehicle Re-ID model such as OSNet
via `torchreid`, or CLIP) and return it from `get_default_embedder`. Nothing
else in the pipeline needs to change.
