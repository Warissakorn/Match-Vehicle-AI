# Match-Vehicle-AI — Cross-Point Vehicle Re-Identification

[![CI](https://github.com/Warissakorn/Match-Vehicle-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Warissakorn/Match-Vehicle-AI/actions/workflows/ci.yml)

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
| Extract | `src/mash_reid/video_extractor.py` | (optional) Cuts timestamped still frames from A/B videos, one or many clips at a time |
| Fix time | `src/mash_reid/timestamp_ocr.py` | (optional) OCRs the true on-screen clock burned into each frame |
| Load  | `src/mash_reid/frame_loader.py` | Reads images, gets each frame's timestamp from an OCR sidecar / filename / EXIF / mtime |
| Detect | `src/mash_reid/detector.py` | YOLOv8 finds cars, motorcycles, buses, trucks and crops them |
| Embed | `src/mash_reid/embedder.py` | Each crop → an L2-normalized appearance vector (ResNet50 by default) |
| Match | `src/mash_reid/matcher.py` | Cosine similarity + a travel-time gate (A must precede B), then ranks |
| Run   | `src/mash_reid/pipeline.py` | Ties it together, caches detections per folder |

The **travel-time gate** encodes physical reality: a vehicle passes A *before*
B, within a configurable window (e.g. 0–600 s). This filters out visually
similar but temporally impossible pairs — which makes an **accurate
timestamp** the thing the gate lives or dies on; see "Fixing inaccurate
timestamps (OCR)" below if a video's declared fps/start-time drifts from the
footage's real clock.

`matcher.py` also groups **repeat sightings of the same vehicle within one
point** (e.g. a car circling back past the same camera) via
`cluster_same_point` — appearance-only, no time gate, since all detections are
already known to be at that point. Every detection still appears in the
GUI/CLI output; repeats are just tagged so they read as one vehicle instead of
several.

## Quick start (one-click launcher)

The easiest way to run the app. The launcher creates a local virtual
environment, installs dependencies on first run, and opens the GUI. Later runs
reuse the environment (reinstalling only if `requirements.txt` changed).

**Linux / macOS**

```bash
./run.sh
```

**Windows**

```bat
run.bat
```

(double-clicking the file works on most desktops)

## Manual install

If you prefer to set things up yourself:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights (YOLOv8n ≈ 6 MB, ResNet50 ≈ 100 MB) download automatically on
first run and are cached afterwards.

## Video input

If you have **videos** of points A and B rather than pre-cut frames, extract
timestamped stills first. Each video's start time is read from its filename
(e.g. `A_20260723_101500.mp4`); each frame's real-world time is then
`start_time + frame_index / fps`, so the travel-time gate keeps working.

**GUI:** click **From video...** next to a point's folder, pick one or more
videos (multi-select in the file dialog), set the interval, and extract — the
output folder is filled in for you. Picking a single video keeps the editable
**Start time** field (auto-filled from the filename); picking several treats
them as clips of the same point (e.g. a camera that splits recordings hourly)
and combines them into one folder — the Start time field is disabled since
each clip resolves its own start time from its own filename instead.

**CLI:**

```bash
python extract_video.py --video A_20260723_101500.mp4 --point A --interval 1.0
python extract_video.py --video B_20260723_101745.mp4 --point B --interval 1.0

# Multiple clips of the same point, combined into one output folder:
python extract_video.py --video A_part1.mp4 A_part2.mp4 A_part3.mp4 --point A
```

Then point the app at the two output folders. Useful flags: `--out` (output
folder), `--start-time "YYYY-MM-DD HH:MM:SS"` (override the filename time,
single-video only), `--ext`, `--ocr-time` (see below).

## Fixing inaccurate timestamps (OCR)

`start_time + frame_index / fps` is only as accurate as the video's declared
fps and its filename's start time — in the field these can drift from the
camera's actual on-screen clock, and since the travel-time gate depends
directly on timestamps, a wrong clock means wrong or missing matches. If your
footage burns the true capture time into the frame as on-screen text (common
on CCTV), OCR-ing it fixes this without re-extracting anything.

**GUI:** click **Fix times (OCR)...** next to a point's folder (after
extracting or on an existing folder of stills). It locates the on-screen
timestamp once from a sample frame, reads every frame's clock, and writes the
result as a `.timestamps.json` sidecar in that folder — a one-time cost per
folder, not a per-run one. Click **Process** again afterwards to pick up the
corrected times; each thumbnail's second caption line shows `[ocr]` when a
frame's time came from this sidecar (vs. `[filename]`/`[exif]`/`[mtime]`).

**CLI:** `--ocr-time` on `cli.py` or `extract_video.py` runs the same OCR pass
before processing / right after extraction.

Timestamp priority is: **OCR sidecar → filename → EXIF → file mtime** — the
sidecar wins whenever present, since it reflects the footage's real clock
rather than an assumption about it. Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR); install it via `requirements.txt` (a one-time model
download on first use).

## Usage

### Desktop GUI (Tkinter)

```bash
python app/gui.py          # or just ./run.sh
```

1. Browse to the **Point A** and **Point B** frame folders — or click
   **From video...** to extract frames from a video first, and optionally
   **Fix times (OCR)...** if the footage has an on-screen clock (see above).
2. Pick a **Detection model** from the dropdown (or click **Manage models...**
   to download / update weights). Pick a **Device** (Auto / CPU / CUDA —
   Auto uses a GPU when one's available). Tune sliders if needed (similarity
   threshold, detection confidence, travel window).
3. Click **Process**. First run downloads the models. The right panel starts by
   showing every vehicle detected at point B — useful for browsing before
   you've picked anything from A ("Show all B" returns to this view any time).
   Galleries cap at `config.DEFAULT_MAX_GALLERY_THUMBNAILS` (300 by default,
   earliest by time) since real footage can produce thousands of vehicles per
   point and rendering a thumbnail for every one would freeze the window; the
   status bar says so when a gallery is truncated. Matching against an A
   vehicle is unaffected — it isn't limited to what's currently rendered.
4. Click a vehicle in the A gallery → its best B-matches appear on the right
   instead, with similarity scores. Click **✓ Same** / **✗ Diff** on a candidate
   to label it as training data (saved under `training_data/`, see below).
   Double-click any thumbnail to view the full frame with the bounding box.

Every thumbnail shows a second, smaller caption line with its source frame's
filename and where its timestamp came from (`[ocr]`/`[filename]`/`[exif]`/
`[mtime]`) — useful for tracing a result back to the exact frame, and for
confirming an OCR fix actually took effect.

Vehicles seen more than once at the *same* point (e.g. a car circling back past
the same camera) are tagged `•GrpN(xK)` in their caption — every detection is
still shown, the tag just flags that they're believed to be one vehicle.

Slider and toggle changes re-match instantly (no re-detection needed).

The bottom bar shows CPU/RAM/GPU usage alongside the status text (each field
reads "n/a" if its dependency or a GPU isn't present) — useful for telling
whether a long run is CPU-, memory-, or GPU-bound. Folder paths, model/device
choice, sliders, and the extractor's last-used interval and video folder are
all remembered in `settings.json` (next to `config.py`, git-ignored) and
restored the next time you open the app.

### Collecting training data

Every time you click **✓ Same** or **✗ Diff** on a proposed A/B match in the
GUI, the pair is saved under `training_data/` for later use fine-tuning or
evaluating a Re-ID model:

```
training_data/
  manifest.csv     # one row per labeled pair: paths, bboxes, timestamps, label, similarity
  positive/        # label = same vehicle
    <pair_id>_A.jpg
    <pair_id>_B.jpg
  negative/        # label = different vehicle
    <pair_id>_A.jpg
    <pair_id>_B.jpg
```

The status bar shows a running count of labeled pairs collected so far this
session. `training_data/` is git-ignored.

### Command line

```bash
python cli.py --dir-a samples/pointA --dir-b samples/pointB \
    --threshold 0.6 --max-travel 600
```

Useful flags: `--model` (detection model, see below), `--device` (`auto` /
`cpu` / `cuda` / `cuda:N`), `--conf` (detection confidence), `--min-travel`/
`--max-travel` (seconds), `--no-time-gate`, `--one-to-one` (force a unique
A↔B assignment), `--no-cache`, `--ocr-time` (fix timestamps via OCR before
processing, see above).

## GPU

Detection and embedding both run on CPU by default unless a GPU is available,
in which case `auto` (the default) uses it automatically — no flag needed.
Override with `--device cpu` / `--device cuda` on the CLI, or the **Device**
dropdown in the GUI. The embedder batches crops in chunks of
`config.DEFAULT_EMBED_BATCH_SIZE` (64 by default) so a single busy frame's
detections can't blow past available GPU memory in one forward pass, and
optionally runs in fp16 on CUDA (`config.USE_HALF_PRECISION_ON_CUDA`).

### GPU not detected?

`requirements.txt` installs plain `torch`/`torchvision`, with no CUDA-specific
index — on some platforms (notably Windows) a plain `pip install torch` gives
you a **CPU-only build**, even on a machine with a real NVIDIA GPU and driver.
That build's `torch.cuda.is_available()` is always `False`, so `cuda` won't
appear in the Device dropdown / `--device` choices — this looks identical to
"no GPU" from the app's side, but is actually just the wrong wheel installed.

The app now tells you which case you're in: the GUI's Device row and the
CLI's `Device:` line both print a reason (e.g. *"This PyTorch build has no
CUDA support (CPU-only wheel)"* vs. *"No CUDA-capable GPU or driver
detected"*) whenever CUDA isn't available.

Check what you actually have installed:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If `torch.version.cuda` prints `None`, reinstall a CUDA build — pick the
command for your CUDA driver version at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
e.g.:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Detection models

You can choose which YOLO model does the detection and keep the weights current.
Two families are offered — **YOLOv8** (the original baseline) and **YOLO11**
(newer, a bit faster/more accurate; recommended). Within a family the size goes
`n` (nano, fastest) → `s` → `m` → `l` → `x` (largest, most accurate). All are
COCO-pretrained, so the same vehicle classes apply.

**Manage from the command line:**

```bash
python models_cli.py list                 # available + what's downloaded
python models_cli.py download yolo11n.pt  # fetch a model
python models_cli.py update yolo11n.pt    # re-fetch the latest weights
python models_cli.py remove yolov8x.pt    # delete local weights
```

Then run with a chosen model:

```bash
python cli.py --dir-a samples/pointA --dir-b samples/pointB --model yolo11n.pt
```

`--model` also accepts a **custom `.pt` path** (e.g. your own fine-tuned vehicle
model), which is passed straight to YOLO.

**Manage from the GUI:** use the **Detection model** dropdown and the
**Manage models...** button (download / update / remove, with install status).

**Where weights live / staying up to date:** downloaded weights are cached in
`<project>/models` (override with the `MASH_MODELS_DIR` env var or `--models-dir`).
"Latest" tracks the installed `ultralytics` version, so to pull newer published
weights, upgrade the package (`pip install -U ultralytics`) and then
`python models_cli.py update <model>`.

## Logs

Every run writes a timestamped log file under `logs/` (e.g.
`logs/mash_reid_20260723_101530.log`) for later analysis. The file captures full
DEBUG detail — frames loaded, per-frame detection counts, unreadable images,
model load, extraction stats, and errors — while the console shows INFO.

- CLIs: `--log-dir <dir>` changes the folder, `--verbose` also prints DEBUG to
  the console.
- The GUI logs automatically and shows the log path in its status bar.

If detection finds nothing, the log tells you why (e.g. "could not read image
…" for every frame points to an unreadable folder). Log files are git-ignored.

## Filename timestamp convention

Capture time is resolved in priority order: **OCR sidecar** (see "Fixing
inaccurate timestamps" above) → **filename** → **EXIF** `DateTimeOriginal` →
**file modification time**. Supported filename formats (configurable in
`config.py` → `TIMESTAMP_PATTERNS`):

- `A_20260723_101530.jpg` → 2026-07-23 10:15:30
- `2026-07-23_10-15-30.jpg`
- `20260723101530.jpg`

## Tests

```bash
python -m pytest tests/
```

The tests cover similarity, temporal gating, ranking, one-to-one assignment,
timestamp parsing (including OCR overlay text), device resolution, embedder
batching, resource sampling, and settings persistence — all with synthetic
data or fakes, so no model, GPU, or network is needed. Tests touching an
optional heavy dependency (`cv2`, `torch`, `psutil`, `easyocr`) skip cleanly
via `pytest.importorskip` when it isn't installed, matching CI's minimal
`numpy`/`scipy`/`pytest`-only environment.

## Configuration

All tunables live in `config.py`: vehicle classes, default detection model,
detection confidence, timestamp patterns, similarity threshold, travel-time
window, embedder batch size / half precision, resource-bar poll interval, and
the settings file location. The selectable detection models are catalogued in
`src/mash_reid/model_registry.py` and downloaded/updated via
`src/mash_reid/model_manager.py` — add a `ModelInfo` entry to offer another one.

## Swapping the appearance model

The default embedder is ImageNet ResNet50 — a reliable baseline. For higher
cross-angle accuracy, implement the `Embedder` interface in
`src/mash_reid/embedder.py` (e.g. a dedicated vehicle Re-ID model such as OSNet
via `torchreid`, or CLIP) and return it from `get_default_embedder`. Nothing
else in the pipeline needs to change.
