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

`matcher.py` can also group **repeat sightings of the same vehicle within one
point** (e.g. a car circling back past the same camera) via
`cluster_same_point` — appearance-only, no time gate, since all detections are
already known to be at that point. Every detection still appears in the
GUI/CLI output; repeats are just tagged so they read as one vehicle instead of
several.

**This is off by default.** It compares every vehicle at a point against every
other, which at a real gallery size (12,000 vehicles) is a ~590 GFLOP pass
peaking over 1 GB — per point, and its result isn't cached, so on a warm cache
it becomes the single dominant cost of a run while feeding nothing but a
caption tag. Nothing else depends on it: matching, the travel-time gate and
the training export are all independent. Turn it on with the **Group repeat
sightings** checkbox in the GUI or `--cluster-same-point` on the CLI.

## Quick start (one-click launcher)

The easiest way to run the app. The launcher creates a local virtual
environment, installs dependencies on first run, and opens the GUI. Later runs
reuse the environment (reinstalling only if `requirements.txt` changed). If it
detects an NVIDIA GPU (`nvidia-smi`), it installs a CUDA-enabled PyTorch build
automatically — see "GPU not detected?" below if that doesn't end up working.

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

### How it works: sample, then fit

Reading the clock on every frame is both slow and fragile — a 12,000-frame
folder takes tens of minutes, and each reading is accepted on its own, so a
single misread digit is committed silently. Worse, frames the OCR *fails* on
keep their filename time, so one folder ends up carrying two different clocks.

Extracted frames carry their source frame index in the filename and are
sampled at a fixed interval, so the clock is a straight line in that index:

```
true_time = clip_start + frame_index / fps
```

That's two unknowns per clip. So instead of reading 12,000 frames, the app
reads about **40**, then fits the line through them with a robust estimator
(Theil–Sen plus one outlier-rejection pass). Roughly **300× less OCR**, and
*more* accurate — a misread lands far off the line and is discarded, and
frames whose clock couldn't be read at all still get a correct time from the
fit, which is what removes the two-clocks problem.

### Reviewing the result

**GUI:** click **Fix times...** next to a point's folder. When the scan
finishes, a review window shows what was found:

- the fitted **rate** (as fps) and each clip's **start time**
- how many sampled readings **agree** with the line, the typical error and the
  worst one
- warnings when something looks wrong (implausible frame rate, readings that
  don't line up, a worst-case error suggesting gaps or a variable frame rate)
- the sampled frames, each with the **clock crop**, the raw OCR text, and —
  importantly — the **filename clock beside the read clock**

That last column is the point of the review. A robust fit defends against
*independent* misreads, but it cannot detect a *systematic* one: if the
overlay font makes every `8` read as `6`, the readings stay mutually
consistent and the residuals look perfect. The gap between the two clocks is
what a human can actually judge ("everything is exactly 3 h 12 m off" is a
timezone bug, not a fit problem).

Pick any sampled frame to see the alternative readings the OCR considered
(with confidence), choose a different one, or type the correct time. The fit
re-runs instantly on every edit, so you can watch the errors settle. There's
also a **± seconds** nudge per clip or across all clips, and **Read every
frame instead** as a fallback to the slow exhaustive pass for footage a
straight line genuinely can't describe.

**Zoom in...** opens the selected frame's full image (not just the small
crop shown inline) in its own window, with the detected/marked clock region
outlined — scroll the mouse wheel to zoom in and out around the cursor, and
use the scrollbars to pan; useful when the clock sits near an edge the
default crop preview doesn't show enough of.

If the overlay's clock uses a layout the built-in parser can't guess (a
different field order, an unusual separator), the **time pattern** picker —
the only place this is edited, so there's one place to look — offers a
dropdown of ready-made layouts (ISO with colon or dot time, day/month/year
with a 12-hour clock, compact with no separators, ...) plus any you've saved
yourself. Typing your own uses the same small token vocabulary (`YYYY`, `MO`,
`DD`, `HH`/`hh`, `MI`, `SS`, `AP`; anything else is matched literally) — e.g.
`YYYY-MO-DD HH.MI.SS` for `2026-07-24 17.40.50`; click **?** for the full
list. **Re-parse** applies the current pattern to the OCR text already
collected — no new OCR call. **Save as...** names and keeps a pattern for
reuse (stored in `ocr_patterns.json`, git-ignored); **Manage...** reviews or
deletes saved ones. The last pattern used is remembered automatically for
the next scan on either point.

If automatic detection locks onto the wrong text entirely (a camera-model
watermark, a plate — visible as the same confident-looking non-time reading
on every sampled frame), **Mark clock position...** lets you draw a box
around the clock by hand on one frame; **Re-sample clock** re-reads the
samples with that fixed region, which is also faster than the automatic
probe. Both the region and every correction are saved, so clicking **Fix
times...** again on the same folder reopens this exact review — corrections,
custom pattern, marked region and all — with **no OCR call at all**, rather
than reading the clock from scratch every time.

Click **Process** again afterwards to pick up the corrected times. Each
thumbnail's second caption line shows `[timeline]` when a frame's time came
from a fit (or `[ocr]` from the older every-frame pass, vs.
`[filename]`/`[exif]`/`[mtime]`).

**CLI:** `--ocr-time` on `cli.py` or `extract_video.py` does the same fit
headlessly. Since nobody is there to eyeball a suspicious result, it *refuses*
to write when the sanity checks fail — `--force` overrides. `--ocr-samples N`
changes how many frames are sampled, and `--ocr-every-frame` selects the old
exhaustive path.

Timestamp priority is: **sidecar → filename → EXIF → file mtime** — the
sidecar wins whenever present, since it reflects the footage's real clock
rather than an assumption about it. Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR); install it via `requirements.txt` (a one-time model
download on first use).

> Accuracy is bounded by the **time span** the samples cover, not by how many
> you take: the clock only displays whole seconds, and that 1 s quantum
> divides by the span. Samples spread across a three-hour clip pin the rate far
> better than the same number bunched into a few minutes — which is why they're
> spread evenly, endpoints included.

## Usage

### Desktop GUI (Tkinter)

```bash
python app/gui.py          # or just ./run.sh
```

The window is laid out as the three steps the workflow actually has, in order.
Each step folds away with the triangle in its header (or by clicking the header
itself), and the fold state is remembered between runs — once the folders and
clocks are settled, folding steps 1 and 2 hands the whole height back to the
galleries, which is where the actual comparing happens. **A draggable divider**
separates the settings from the results: drag it to give either half as much
height as you want, and its position is remembered too. The settings half
scrolls, so wherever the divider sits nothing is out of reach.

**Step 1 — Frames.** Browse to the **Point A** and **Point B** frame folders, or
click **From video...** to extract frames from a video first.

**Step 2 — Timestamps and model.** Each point shows a one-line summary of where
its times currently come from — `no timestamp review yet - times come from
filenames`, or e.g. `12557 frame time(s) confirmed from a fitted clock, 29.97
fps`. Green means reviewed and confirmed, amber means fitted but not confirmed
(or the fit raised warnings), grey means nothing has been done yet. Click
**Fix times...** to open the review (see above) if the footage has an on-screen
clock. This line matters because matching is gated on travel time: with wrong
clocks, Process runs to completion and quietly finds nothing. Also here: the
**Detection model** dropdown (**Manage models...** downloads / updates weights)
and the **Device** picker (Auto / CPU / CUDA — Auto uses a GPU when one is
available).

**Step 3 — Match.** Tune the similarity threshold and travel window, then click
**Process**. First run downloads the models. Everything in this step except
**Detect conf** re-filters an existing run instantly; detection confidence
changes what gets detected at all, so it needs another Process.

After a run, the right panel starts by showing every vehicle detected at point B
— useful for browsing before you've picked anything from A ("Show all B" returns
to this view any time). Galleries cap at `config.DEFAULT_MAX_GALLERY_THUMBNAILS`
(300 by default, earliest by time) since real footage can produce thousands of
vehicles per point and rendering a thumbnail for every one would freeze the
window; each gallery's own header says so when it is truncated. Matching against
an A vehicle is unaffected — it isn't limited to what's currently rendered.

Click a vehicle in the A gallery → its best B-matches appear on the right, with
similarity scores. Click **✓ Same** / **✗ Diff** on a candidate to label it as
training data (saved under `training_data/`, see below). Double-click any
thumbnail to open the full frame in the same scrollable, mouse-wheel-zoomable
viewer the timestamp review uses, with the vehicle's box outlined — enough to
read a plate rather than just confirm a shape. Both galleries scroll with the
mouse wheel.

The **Fix times...** review window has the same treatment: its Apply / Cancel /
"Read every frame instead" footer is pinned to the bottom edge and everything
above it scrolls, so on a screen too short for the full dialog the two decisions
it exists to collect are still reachable instead of being clipped off-screen.

Every thumbnail shows a second, smaller caption line with its source frame's
filename and where its timestamp came from (`[ocr]`/`[filename]`/`[exif]`/
`[mtime]`) — useful for tracing a result back to the exact frame, and for
confirming an OCR fix actually took effect.

Vehicles seen more than once at the *same* point (e.g. a car circling back past
the same camera) are tagged `•GrpN(xK)` in their caption — every detection is
still shown, the tag just flags that they're believed to be one vehicle.

Slider and toggle changes re-match instantly (no re-detection needed). Dragging
a slider coalesces into a single refresh once the pointer settles, so a drag
across the track costs one gallery rebuild rather than one per pixel of travel.

### Language

The **Language** picker in the status bar switches the interface between
English and Thai; the choice is remembered in `settings.json`. Thai uses
**TH SarabunPSK** (falling back to TH Sarabun New, Sarabun, Leelawadee UI,
Tahoma or Noto Sans Thai, whichever the machine has), because none of the
design system's Latin faces carry Thai glyphs at all.

Sarabun is drawn on a much smaller body than a Latin UI face, so it is scaled
up by 1.45 — the English UI's 11pt body becomes 16pt Sarabun, which is the
conventional equivalence and comes out the same visual size rather than
noticeably smaller. The fallback faces are normally proportioned and are left
at 1.0; applying Sarabun's correction to Tahoma would overshoot. See
`_FAMILY_SCALE` in `app/theme.py`.

Switching language rebuilds the window rather than retranslating in place:
every label is constructed with its final text, so there is no registry of
"this widget shows string X" to walk — and a rebuild also picks up the font
change, which a text substitution could not. Settings live in Tk variables and
results in plain data, both of which outlive the widgets, so nothing is lost.

Strings live in `app/i18n.py`, keyed by their **English source text**. An
untranslated string therefore renders as readable English rather than as an
error or a placeholder, so adding a label without touching the catalog never
breaks the Thai UI. `tests/test_i18n.py` guards the cost of that choice — it
fails if a catalog key no longer has a call site.

### Icon

`assets/icon.ico` / `icon.png` are generated by `python tools/make_icon.py`,
which reads the brand indigo from `app/palette.py` so the mark cannot drift
from the UI. The mark is two camera points joined by one path — a hollow ring
(seen at A) and a filled dot (matched at B). Deliberately not a car
silhouette: below ~32px a vehicle is an unreadable smudge, while two nodes and
a bar stay legible at the 16px the taskbar actually uses.

### Appearance

The window's look follows `app/theme.py`, which ports the design system in
`genesisDESIGN_1.md` onto ttk: its palette, type scale, 4px spacing grid, 1px
borders, and the rule that only one filled indigo button (**Process**) appears
per view. What could not be ported is documented at the top of that file —
rounded corners, shadows and backdrop blur have no Tk equivalent, and the
document's web fonts (General Sans / DM Sans / JetBrains Mono) are used when
installed and substituted with the closest system faces otherwise.

**Light and dark.** The button at the bottom right of the status bar switches
schemes; it is labelled with the mode it will switch *to*. The choice is
remembered in `settings.json`. Dark is a designed palette rather than an
inversion — text is `#F5F5F7` rather than pure white, the darkest surface is
`#0A0A0B` rather than pure black (the design document forbids pure values for
text), and the indigo moves up its ramp to `#818CF8` because the light-mode
value reads as near-black against a dark surface. Both schemes are checked
against WCAG AA contrast ratios; the semantic colours have separate darker
variants (`SUCCESS_TEXT`, `WARNING_TEXT`, `ERROR_TEXT`) for use as coloured
text, because the document's own values are specified for status chips and
measure only 2.2–2.5:1 as text on white.

`ttk.Labelframe` is used nowhere in the app: it draws its caption straddling
the top border and `clam` leaves no gap behind it, so the 1px rule ran straight
through the text. `theme.panel()` puts the caption above the border instead.
For the same reason `Card.TFrame` (bordered) and `Surface.TFrame` (the same
fill, no border) are separate styles: a bordered style reused for the rows and
fillers *inside* a card makes each of them draw its own rectangle, which in one
case rendered as a rule running through a section title.

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

**`run.sh`/`run.bat` handle this automatically** by running
`tools/install_torch.py` during dependency installation. That helper:

1. reads the installed NVIDIA driver version from `nvidia-smi`;
2. picks the **CUDA build that driver actually supports** (`cu124`, `cu121` or
   `cu118`) and installs it *before* the regular `requirements.txt` install, so
   the plain install sees a compatible `torch` already there and leaves it alone;
3. verifies the result and prints one line telling you whether the GPU will
   really be used — e.g. `GPU ready: NVIDIA GeForce RTX 3060 (CUDA 12.1)`.

Matching the wheel to the driver matters: a build newer than the driver
installs *successfully* but still reports `torch.cuda.is_available() == False`,
which looks exactly like having no GPU. If your driver is older than every CUDA
build above, the helper says so and stays on CPU rather than installing
something that can't initialize — update the driver to enable GPU support.

The whole step is best-effort (any failure falls back to the CPU build) and
only runs when dependencies are installed, not on every launch. **Delete
`.venv` (or just `.venv/.requirements.sha256`) and re-run the launcher to force
a re-check** — e.g. after installing or updating a GPU driver, or after
upgrading from a version of this project that didn't have the check.

If you installed manually (`pip install -r requirements.txt` yourself) or the
automatic install still isn't picking up your GPU, the app tells you which
case you're in: the GUI's Device row and the CLI's `Device:` line both print a
reason (e.g. *"This PyTorch build has no CUDA support (CPU-only wheel)"* vs.
*"No CUDA-capable GPU or driver detected"*) whenever CUDA isn't available.
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

## When the app won't start

On Windows, double-clicking `run.bat` opens a console that closes with the
process — so a startup crash used to look like "the window flashes and
disappears", with the traceback gone. Two things now prevent that:

- `run.bat` holds the console open when the app exits non-zero, so the error
  is readable.
- The traceback is written to `logs/` before the process exits, regardless of
  how the app was launched.

`tests/test_gui_startup.py` is the guard against the class of bug that caused
it: the whole window (and the dialogs one click away) is constructed against a
fake Tkinter, so a missing name or a bad call fails a test rather than the
user's launcher. It exists because `python -m py_compile` cannot catch an
undefined name that is only resolved when the function runs — which is exactly
what shipped.

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
the settings file location.

Timeline-fit tunables sit there too — how many frames to sample
(`DEFAULT_TIMELINE_SAMPLE_TARGET`, and the per-clip floor), the outlier
tolerance, and the thresholds behind each review warning. One of them is
load-bearing rather than cosmetic: `TIMELINE_INLIER_FLOOR_SECONDS`. The
overlay clock shows whole seconds, so on clean footage the readings agree
*exactly* and the measured spread is zero; without a floor under the outlier
tolerance, the band collapses to zero width and every sample gets rejected.
Same-point clustering is switched by `DEFAULT_ENABLE_SAME_POINT_CLUSTERING`.

The selectable detection models are catalogued in
`src/mash_reid/model_registry.py` and downloaded/updated via
`src/mash_reid/model_manager.py` — add a `ModelInfo` entry to offer another one.

## Swapping the appearance model

The default embedder is ImageNet ResNet50 — a reliable baseline. For higher
cross-angle accuracy, implement the `Embedder` interface in
`src/mash_reid/embedder.py` (e.g. a dedicated vehicle Re-ID model such as OSNet
via `torchreid`, or CLIP) and return it from `get_default_embedder`. Nothing
else in the pipeline needs to change.
