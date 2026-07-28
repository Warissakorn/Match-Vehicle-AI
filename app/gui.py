"""Simple Tkinter desktop app for cross-point vehicle Re-ID.

Workflow:
    1. Pick the folder of frames for point A and for point B.
    2. Adjust thresholds / travel-time window if needed.
    3. Click "Process" -- detection + embedding run in a background thread.
    4. Before selecting anything from A, the right panel shows *every* vehicle
       detected at point B ("Show all B" returns to this view any time).
    5. Click a vehicle in the A gallery -> its best B-matches appear on the
       right instead, with similarity scores and "✓ Same" / "✗ Diff" buttons to
       label the pair -- saved under ``training_data/`` for later model
       training. Double-click any thumbnail to see the full frame with the
       bounding box drawn.

Repeat sightings of the same vehicle within one point (e.g. circling back past
the same camera) are tagged "•GrpN(xK)" in their caption so duplicates read as
one vehicle without hiding any individual detection.

Run with:  python app/gui.py
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

# Make ``config`` (project root) and the ``mash_reid`` package importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import config  # noqa: E402
from mash_reid import (  # noqa: E402
    logging_setup,
    matcher,
    model_manager,
    model_registry,
    pipeline,
    settings,
    sysmon,
    timestamp_ocr,
    training_export,
    video_extractor,
)
from mash_reid.matcher import VehicleRecord  # noqa: E402

# Named under the "mash_reid" tree (rather than __name__, which would be
# "app.gui" or "__main__") so this module's log lines land in the same
# rotating file the rest of the package writes to -- logging_setup only
# attaches handlers to "mash_reid" and its descendants.
log = logging.getLogger("mash_reid.gui")

THUMB_SIZE = (140, 110)


def _load_thumb(record: VehicleRecord, size=THUMB_SIZE):
    """Load and crop a record's vehicle box into a Tk-ready thumbnail image."""
    from PIL import Image, ImageTk

    img = Image.open(record.frame_path).convert("RGB")
    x1, y1, x2, y2 = record.bbox
    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail(size)
    return ImageTk.PhotoImage(crop)


def _show_full_frame(parent, record: VehicleRecord):
    """Popup showing the full source frame with the bounding box drawn."""
    from PIL import Image, ImageDraw, ImageTk

    img = Image.open(record.frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(record.bbox, outline=(255, 0, 0), width=4)
    # Scale down large frames to fit typical screens.
    max_side = 900
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    top = tk.Toplevel(parent)
    top.title(f"{record.point}  {os.path.basename(record.frame_path)}  "
              f"{record.timestamp:%Y-%m-%d %H:%M:%S}")
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(top, image=photo)
    label.image = photo  # keep a reference
    label.pack()


class FrameZoomDialog(tk.Toplevel):
    """Full frame, scrollable and mouse-wheel zoomable.

    Replaces an earlier version that upscaled just the tiny detected clock
    crop: that clipped the view to whatever region auto-detection (or a
    manually drawn box) happened to land on, so a clock sitting at the edge
    of frame -- or a region drawn a little too tight -- was invisible no
    matter how far it was enlarged. Showing the whole frame instead, with
    the region outlined for reference, means the on-screen clock is findable
    wherever it actually is.

    Opens at a modest fit-to-window size rather than a large fixed popup;
    the scroll wheel zooms in/out around the cursor, and scrollbars pan
    once the image is larger than the window.
    """

    INITIAL_SIZE = (720, 520)
    MIN_SCALE, MAX_SCALE = 0.2, 8.0
    WHEEL_STEP = 1.15

    def __init__(self, parent, image_path: str, title: str,
                region: tuple[int, int, int, int] | None = None):
        super().__init__(parent)
        self.title(title)
        self._photo = None  # keep-alive ref

        from PIL import Image, ImageDraw

        img = Image.open(image_path).convert("RGB")
        if region is not None:
            draw = ImageDraw.Draw(img)
            draw.rectangle(region, outline=(255, 48, 48), width=max(2, img.width // 300))
        self._base_image = img

        win_w, win_h = self.INITIAL_SIZE
        # Fit-to-window initial scale, same idea as the other preview dialogs,
        # but never enlarged past 1:1 just to fill the window -- a small
        # source image should open at its real size, not pre-blurred.
        self._scale = min(win_w / img.width, win_h / img.height, 1.0)

        self.canvas = tk.Canvas(self, width=win_w, height=win_h, background="#202020")
        hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._image_id = self.canvas.create_image(0, 0, anchor="nw")
        self._redraw()

        ttk.Label(self, text="Scroll to zoom - drag the scrollbars to pan.",
                  font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2,
                                                  sticky="w", padx=4, pady=(2, 4))

        # Windows/macOS deliver <MouseWheel> with event.delta; X11 (Linux)
        # sends <Button-4>/<Button-5> instead -- bind all three so the wheel
        # zooms regardless of platform.
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e, self.WHEEL_STEP))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e, 1 / self.WHEEL_STEP))

    def _redraw(self):
        from PIL import Image, ImageTk

        w = max(1, int(self._base_image.width * self._scale))
        h = max(1, int(self._base_image.height * self._scale))
        resized = self._base_image.resize((w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self._image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))

    def _on_wheel(self, event):
        factor = self.WHEEL_STEP if event.delta > 0 else 1 / self.WHEEL_STEP
        self._zoom_at(event, factor)

    def _zoom_at(self, event, factor):
        """Rescale the image, keeping the point under the cursor stationary."""
        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
        if new_scale == self._scale:
            return

        old_total_w = self._base_image.width * self._scale
        old_total_h = self._base_image.height * self._scale
        cursor_x = self.canvas.canvasx(event.x)
        cursor_y = self.canvas.canvasy(event.y)
        # Where the cursor points, as a fraction of the full image -- this is
        # what must land back under the cursor after the resize.
        rel_x = cursor_x / old_total_w if old_total_w else 0.0
        rel_y = cursor_y / old_total_h if old_total_h else 0.0

        self._scale = new_scale
        self._redraw()

        new_total_w = self._base_image.width * self._scale
        new_total_h = self._base_image.height * self._scale
        target_x = rel_x * new_total_w - event.x
        target_y = rel_y * new_total_h - event.y
        if new_total_w:
            self.canvas.xview_moveto(max(0.0, target_x / new_total_w))
        if new_total_h:
            self.canvas.yview_moveto(max(0.0, target_y / new_total_h))


class ScrollableThumbs(ttk.Frame):
    """A vertically scrolling frame that lays out thumbnail widgets in a grid."""

    def __init__(self, parent, columns=2):
        super().__init__(parent)
        self.columns = columns
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self._scroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._inner = ttk.Frame(self._canvas)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scroll.pack(side="right", fill="y")
        self._refs: list = []  # keep PhotoImage refs alive

    def clear(self):
        for child in self._inner.winfo_children():
            child.destroy()
        self._refs.clear()

    def add_card(self, record: VehicleRecord, caption: str, on_click, on_double,
                on_confirm=None, on_reject=None, subcaption: str | None = None):
        """Add a thumbnail card. If ``on_confirm``/``on_reject`` (no-arg
        callables) are given, a "Same"/"Different" button pair is shown below
        the caption for labeling this candidate as training data; both
        disable themselves after either is clicked. ``subcaption`` -- when
        given -- is shown as a second, smaller line (source frame filename
        and where its timestamp came from) so a result can be traced back to
        the exact image it was cropped from.
        """
        idx = len(self._refs)
        photo = _load_thumb(record)
        self._refs.append(photo)

        card = ttk.Frame(self._inner, relief="ridge", borderwidth=1, padding=3)
        card.grid(row=idx // self.columns, column=idx % self.columns, padx=4, pady=4)

        btn = tk.Label(card, image=photo, cursor="hand2")
        btn.pack()
        btn.bind("<Button-1>", lambda e: on_click(record))
        btn.bind("<Double-Button-1>", lambda e: on_double(record))

        ttk.Label(card, text=caption, font=("TkDefaultFont", 8)).pack()
        if subcaption:
            ttk.Label(card, text=subcaption, font=("TkDefaultFont", 7),
                      foreground="#666666").pack()

        if on_confirm or on_reject:
            row = ttk.Frame(card)
            row.pack(fill="x")
            status_lbl = ttk.Label(card, text="", font=("TkDefaultFont", 7))
            confirm_btn = ttk.Button(row, text="✓ Same", width=6)
            reject_btn = ttk.Button(row, text="✗ Diff", width=6)

            def _mark_done(text):
                confirm_btn.config(state="disabled")
                reject_btn.config(state="disabled")
                status_lbl.config(text=text)

            if on_confirm:
                def _do_confirm():
                    on_confirm()
                    _mark_done("saved: same vehicle")
                confirm_btn.config(command=_do_confirm)
                confirm_btn.pack(side="left", expand=True, fill="x")
            if on_reject:
                def _do_reject():
                    on_reject()
                    _mark_done("saved: different vehicle")
                reject_btn.config(command=_do_reject)
                reject_btn.pack(side="left", expand=True, fill="x")
            status_lbl.pack(fill="x")


class ReIDApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)

        # Detection model: keep a display<->key mapping so the combobox can show
        # friendly names while we pass the weights key to the pipeline.
        self._model_display_to_key = {m.display_name: m.key for m in model_registry.all_models()}
        self._model_key_to_display = {m.key: m.display_name for m in model_registry.all_models()}

        saved = settings.load()  # {} if this is the first run, or the file's gone/corrupt

        self.dir_a = tk.StringVar(value=saved.get("dir_a", ""))
        self.dir_b = tk.StringVar(value=saved.get("dir_b", ""))
        model_key = saved.get("model_key", config.YOLO_WEIGHTS)
        self.model_key = tk.StringVar(value=model_key)
        self.model_display = tk.StringVar(
            value=self._model_key_to_display.get(model_key, model_registry.default().display_name))
        self.device = tk.StringVar(value=saved.get("device", "auto"))
        self.threshold = tk.DoubleVar(
            value=saved.get("threshold", config.DEFAULT_SIMILARITY_THRESHOLD))
        self.det_conf = tk.DoubleVar(
            value=saved.get("det_conf", config.DEFAULT_DETECTION_CONF))
        self.min_travel = tk.DoubleVar(value=saved.get("min_travel", 0.0))
        self.max_travel = tk.DoubleVar(value=saved.get("max_travel", 600.0))
        self.use_gate = tk.BooleanVar(value=saved.get("use_gate", True))
        self.one_to_one = tk.BooleanVar(value=saved.get("one_to_one", False))
        self.cluster_same_point = tk.BooleanVar(
            value=saved.get("cluster_same_point",
                            config.DEFAULT_ENABLE_SAME_POINT_CLUSTERING))
        # Shared with VideoExtractDialog so the last-used interval and video
        # folder are remembered across dialog opens, not just within one.
        self.extract_interval = tk.DoubleVar(value=saved.get("extract_interval", 1.0))
        self.timeline_samples = tk.IntVar(
            value=saved.get("timeline_samples", config.DEFAULT_TIMELINE_SAMPLE_TARGET))
        # Last-used overlay format ("YYYY-MO-DD HH.MI.SS"), for cameras whose
        # clock the built-in tolerant parser can't guess -- see
        # timestamp_ocr.compile_custom_pattern. No editing UI lives on this
        # window; it's set and edited from TimelineReviewDialog's pattern
        # picker (the only place that can test a pattern against real OCR
        # text without a new scan) and just remembered here so a fresh scan
        # on either point starts from whatever was last used.
        self.ocr_time_pattern = tk.StringVar(value=saved.get("ocr_time_pattern", ""))
        self.last_video_dir = tk.StringVar(value=saved.get("last_video_dir", ""))
        self.status = tk.StringVar(value="Select folders for point A and B, then Process.")

        self._res_a = None
        self._res_b = None
        self._a_by_id: dict[int, VehicleRecord] = {}
        self._b_by_id: dict[int, VehicleRecord] = {}
        self._a_clusters: dict[int, int] = {}  # record_id -> cluster id (same-point repeats)
        self._b_clusters: dict[int, int] = {}
        # cluster id -> member count, precomputed once per Process so rendering
        # a card is O(1) instead of scanning every record's cluster id.
        self._a_cluster_sizes: Counter = Counter()
        self._b_cluster_sizes: Counter = Counter()
        self._last_selected_a: VehicleRecord | None = None
        self._queue: queue.Queue = queue.Queue()

        self._build_controls()
        self._build_status_bar()
        self._build_panels()

    # --- UI construction ---------------------------------------------------

    def _build_controls(self):
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 6))

        # Folder pickers (with an "extract frames from a video" shortcut)
        for label, var, pt in (("Point A folder", self.dir_a, "A"),
                               ("Point B folder", self.dir_b, "B")):
            row = ttk.Frame(top)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=14).pack(side="left")
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=4)
            ttk.Button(row, text="Browse...",
                       command=lambda v=var: self._browse(v)).pack(side="left")
            ttk.Button(row, text="From video...",
                       command=lambda v=var, p=pt: self._extract_from_video(v, p)).pack(side="left", padx=(4, 0))
            btn = ttk.Button(row, text="Fix times...")
            btn.config(command=lambda v=var, p=pt, b=btn: self._fix_times(v, p, b))
            btn.pack(side="left", padx=(4, 0))

        # Detection model picker + manager
        mrow = ttk.Frame(top)
        mrow.pack(fill="x", pady=2)
        ttk.Label(mrow, text="Detection model", width=14).pack(side="left")
        self._model_combo = ttk.Combobox(
            mrow, textvariable=self.model_display, state="readonly",
            values=[m.display_name for m in model_registry.all_models()])
        self._model_combo.pack(side="left", fill="x", expand=True, padx=4)
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(mrow, text="Manage models...",
                   command=self._open_model_manager).pack(side="left")
        self.model_status = tk.StringVar()
        ttk.Label(mrow, textvariable=self.model_status, width=14,
                  font=("TkDefaultFont", 8)).pack(side="left", padx=(6, 0))
        self._refresh_model_status()

        # Compute device: "auto" resolves to CUDA if a GPU is present, else
        # CPU. Probing CUDA touches torch, so it's deferred to a background
        # thread rather than blocking window construction.
        drow = ttk.Frame(top)
        drow.pack(fill="x", pady=2)
        ttk.Label(drow, text="Device", width=14).pack(side="left")
        self._device_combo = ttk.Combobox(
            drow, textvariable=self.device, state="readonly", values=["auto"], width=10)
        self._device_combo.pack(side="left", padx=4)
        self.device_status = tk.StringVar(value="probing...")
        # wraplength so a long GPU name ("CUDA (NVIDIA GeForce RTX 4060
        # Laptop GPU)") wraps onto a second line instead of running past the
        # window's edge with no way to see the rest of it.
        ttk.Label(drow, textvariable=self.device_status, font=("TkDefaultFont", 8),
                  wraplength=420, justify="left").pack(side="left", padx=(6, 0), fill="x", expand=True)
        self._probe_devices()

        # Sliders / options
        opts = ttk.Frame(top)
        opts.pack(fill="x", pady=4)
        self._add_slider(opts, "Similarity", self.threshold, 0.0, 1.0)
        self._add_slider(opts, "Detect conf", self.det_conf, 0.05, 0.9)
        self._add_slider(opts, "Min travel (s)", self.min_travel, 0.0, 1800.0)
        self._add_slider(opts, "Max travel (s)", self.max_travel, 10.0, 3600.0)

        toggles = ttk.Frame(top)
        toggles.pack(fill="x", pady=2)
        ttk.Checkbutton(toggles, text="Use time gate", variable=self.use_gate,
                        command=self._rematch).pack(side="left", padx=4)
        ttk.Checkbutton(toggles, text="One-to-one", variable=self.one_to_one,
                        command=self._rematch).pack(side="left", padx=4)
        ttk.Checkbutton(toggles, text="Group repeat sightings (slow)",
                        variable=self.cluster_same_point).pack(side="left", padx=4)
        self._process_btn = ttk.Button(toggles, text="Process", command=self._on_process)
        self._process_btn.pack(side="right", padx=4)

    def _build_status_bar(self):
        """Bottom-anchored status + resource bar.

        Packed with side="bottom" *before* the gallery panes (which pack
        with fill="both", expand=True) so it stays pinned to the window's
        bottom edge regardless of resizing -- Tkinter's pack manager
        allocates side="bottom" slots from whatever was packed so far, so a
        bottom bar has to claim its space before an expanding widget grabs
        the rest of the cavity. The status label previously lived here
        without ever actually anchoring to the bottom (it just sat between
        the controls and the galleries).
        """
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Label(bar, textvariable=self.status, relief="sunken",
                  anchor="w").pack(side="left", fill="x", expand=True)
        self.resource_status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.resource_status, relief="sunken",
                  anchor="e", font=("TkDefaultFont", 8)).pack(side="right")
        self._poll_resources()

    def _poll_resources(self):
        """Refresh the CPU/RAM/GPU readout. Best-effort: a probe failure
        (missing psutil, no GPU, ...) shows "n/a" for that field rather than
        breaking the loop -- see ``mash_reid.sysmon``.
        """
        try:
            self.resource_status.set(sysmon.format_sample(sysmon.sample()))
        except Exception:
            log.debug("Resource sample failed", exc_info=True)
        self.after(config.RESOURCE_POLL_MS, self._poll_resources)

    def _add_slider(self, parent, label, var, lo, hi):
        frame = ttk.Frame(parent)
        frame.pack(side="left", fill="x", expand=True, padx=4)
        head = ttk.Frame(frame)
        head.pack(fill="x")
        ttk.Label(head, text=label, font=("TkDefaultFont", 8)).pack(side="left")
        val = ttk.Label(head, font=("TkDefaultFont", 8))
        val.pack(side="right")

        def _on_move(_=None):
            val.config(text=f"{var.get():.2f}" if hi <= 1 else f"{var.get():.0f}")
            self._rematch()

        scale = ttk.Scale(frame, from_=lo, to=hi, variable=var,
                          orient="horizontal", command=_on_move)
        scale.pack(fill="x")
        _on_move()

    def _build_panels(self):
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Labelframe(panes, text="Point A vehicles (click one)")
        right = ttk.Labelframe(panes, text="Point B vehicles")
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        self.gallery_a = ScrollableThumbs(left, columns=2)
        self.gallery_a.pack(fill="both", expand=True)

        b_toolbar = ttk.Frame(right)
        b_toolbar.pack(fill="x")
        self.b_view_label = tk.StringVar(value="Showing: all point B vehicles")
        ttk.Label(b_toolbar, textvariable=self.b_view_label,
                 font=("TkDefaultFont", 8)).pack(side="left", padx=4)
        ttk.Button(b_toolbar, text="Show all B",
                  command=self._show_all_b).pack(side="right", padx=4)

        self.gallery_b = ScrollableThumbs(right, columns=2)
        self.gallery_b.pack(fill="both", expand=True)

    # --- Actions -----------------------------------------------------------

    def _browse(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _extract_from_video(self, folder_var, point):
        """Open a dialog to extract frames from a video into a point folder."""
        VideoExtractDialog(self, point, folder_var)

    def _fix_times(self, folder_var, point, button=None):
        """Read the true on-screen clock and fit a timeline for a point's folder.

        Only a sample of frames is read (see ``mash_reid.timeline``); the
        clock's rate and each clip's start time are then solved for and
        applied to every frame. The result opens in a review window for
        confirmation rather than being written straight out -- a robust fit
        catches *independent* misreads, but a systematically misread digit
        produces a self-consistent line that only a human can spot.

        Uses its own queue/thread pair (independent of ``_process_worker``'s)
        since it can be triggered before or after a Process run.
        """
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("No folder", f"Pick a valid folder for point {point} first.",
                                 parent=self)
            return

        # A previous review's samples, fit, region and pattern may already be
        # saved as a v2 sidecar -- reopen it directly instead of reading the
        # clock again. This is the same handful of small image reads the
        # gallery already does synchronously per card, so it's done here
        # rather than threaded, to keep the reopen path simple.
        try:
            saved = timestamp_ocr.load_scan_for_review(folder)
        except Exception:
            log.warning("Could not reopen saved review for %s", folder, exc_info=True)
            saved = None
        if saved is not None:
            self.status.set(f"Point {point}: reopened the saved timestamp review.")
            TimelineReviewDialog(self, point, saved)
            return

        # Compile the custom pattern (if any) up front and fail fast -- a typo
        # here shouldn't be discovered only after a full background scan.
        custom_pattern = None
        pattern_text = timestamp_ocr.strip_pattern_label(self.ocr_time_pattern.get())
        if pattern_text:
            try:
                custom_pattern = timestamp_ocr.compile_custom_pattern(pattern_text)
            except ValueError as exc:
                messagebox.showerror("Bad time pattern", str(exc), parent=self)
                return

        q: queue.Queue = queue.Queue()
        # Read Tk vars here, on the main thread, before handing off.
        device = self.device.get()
        samples = self.timeline_samples.get()
        if button is not None:
            button.config(state="disabled")  # one scan per folder at a time
        self.status.set(f"Reading clock samples for point {point}...")

        def worker():
            try:
                names = sorted(
                    f for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS
                )
                if not names:
                    q.put(("error", f"No images found in {folder}"))
                    return

                def progress(done, total, msg):
                    q.put(("status", f"[{done}/{total}] Reading clock ({point}): {msg}"))

                result = timestamp_ocr.scan_folder(
                    folder, names, device=device, sample_count=samples, progress=progress,
                    custom_pattern=custom_pattern)
                q.put(("done", result))
            except Exception as exc:  # surface errors to the UI thread
                log.warning("Timestamp scan failed for %s", folder, exc_info=True)
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        def finish():
            if button is not None:
                button.config(state="normal")

        def poll():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "status":
                        self.status.set(payload)
                    elif kind == "error":
                        finish()
                        self.status.set("Reading the clock failed.")
                        messagebox.showerror("Could not read timestamps", payload, parent=self)
                        return
                    elif kind == "done":
                        finish()
                        read = sum(1 for s in payload.samples if s.chosen)
                        self.status.set(
                            f"Point {point}: read {read}/{len(payload.samples)} sampled "
                            f"clock(s). Review and confirm to apply.")
                        TimelineReviewDialog(self, point, payload)
                        return
            except queue.Empty:
                pass
            self.after(150, poll)

        self.after(150, poll)

    def _on_model_selected(self, _event=None):
        key = self._model_display_to_key.get(self.model_display.get())
        if key:
            self.model_key.set(key)
        self._refresh_model_status()

    def _refresh_model_status(self):
        """Show whether the chosen model's weights are already downloaded."""
        installed = model_manager.is_installed(self.model_key.get())
        self.model_status.set("✓ downloaded" if installed else "not downloaded")

    def _open_model_manager(self):
        ModelManagerDialog(self, on_close=self._refresh_model_status)

    def _probe_devices(self):
        """Populate the Device dropdown once CUDA availability is known.

        Probing touches torch (a heavy, deferred import), so it runs in a
        background thread; only the queue-drained callback on the main
        thread ever touches the Tk widgets, same convention as every other
        background task in this file.
        """
        q: queue.Queue = queue.Queue()

        def worker():
            from mash_reid import device as device_module

            try:
                devices = device_module.list_available_devices()
                desc = device_module.describe_device(device_module.resolve_device("auto"))
                # "cuda" missing from the list usually isn't "no GPU" -- it's
                # commonly a CPU-only torch wheel (see README's GPU section).
                # Surface the actual reason instead of leaving the user to
                # guess why the dropdown doesn't offer it.
                reason = "" if "cuda" in devices else device_module.diagnose_cuda_unavailable()
            except Exception:
                log.warning("Device probe failed", exc_info=True)
                devices, desc, reason = ["auto", "cpu"], "CPU", ""
            q.put((devices, desc, reason))

        def poll():
            try:
                devices, desc, reason = q.get_nowait()
            except queue.Empty:
                self.after(150, poll)
                return
            self._device_combo.config(values=devices)
            status = f"auto -> {desc}"
            if reason:
                status += f"  ({reason})"
            self.device_status.set(status)

        threading.Thread(target=worker, daemon=True).start()
        self.after(150, poll)

    def _current_match_config(self) -> config.MatchConfig:
        return config.MatchConfig(
            similarity_threshold=self.threshold.get(),
            top_k=config.DEFAULT_TOP_K,
            use_time_gate=self.use_gate.get(),
            min_travel_seconds=self.min_travel.get(),
            max_travel_seconds=self.max_travel.get(),
            one_to_one=self.one_to_one.get(),
        )

    def _on_process(self):
        dir_a, dir_b = self.dir_a.get().strip(), self.dir_b.get().strip()
        if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
            messagebox.showerror("Invalid folders", "Please choose valid A and B folders.")
            return
        self._process_btn.config(state="disabled")
        self.status.set("Loading models and processing frames... (first run downloads weights)")
        # Read the Tk var here, on the main thread, and hand it to the worker
        # as a plain bool -- worker threads must not touch Tk variables.
        cluster = self.cluster_same_point.get()
        thread = threading.Thread(target=self._process_worker,
                                  args=(dir_a, dir_b, cluster), daemon=True)
        thread.start()
        self.after(100, self._poll_queue)

    def _process_worker(self, dir_a, dir_b, cluster_same_point):
        try:
            pcfg = config.PipelineConfig(
                yolo_weights=self.model_key.get(), detection_conf=self.det_conf.get(),
                device=self.device.get())
            detector, embedder = pipeline.build_pipeline(pcfg)

            def progress(done, total, msg):
                self._queue.put(("status", f"[{done}/{total}] {msg}"))

            self._queue.put(("status", "Processing point A..."))
            res_a = pipeline.process_point(dir_a, "A", detector, embedder, pcfg, progress=progress)
            self._queue.put(("status", "Processing point B..."))
            res_b = pipeline.process_point(dir_b, "B", detector, embedder, pcfg, progress=progress)

            # Group repeat sightings of the same vehicle within each point
            # (e.g. a car circling back past the same camera). Computed here,
            # off the GUI thread: with real galleries running into the
            # thousands of vehicles this is seconds of work, and would freeze
            # the window if run from the Tkinter callback instead.
            #
            # Off by default (see config.DEFAULT_ENABLE_SAME_POINT_CLUSTERING):
            # it is an N x N similarity pass per point whose result isn't
            # cached, so on a warm cache it dominates the whole run while only
            # affecting a caption suffix. Empty dicts disable it cleanly --
            # _cluster_tag already renders nothing for an unknown record id.
            a_clusters: dict[int, int] = {}
            b_clusters: dict[int, int] = {}
            if cluster_same_point:
                self._queue.put(("status", "Grouping repeat sightings..."))
                a_clusters = matcher.cluster_same_point(res_a.records)
                b_clusters = matcher.cluster_same_point(res_b.records)

            self._queue.put(("done", (res_a, res_b, a_clusters, b_clusters)))
        except Exception as exc:  # surface errors to the UI thread
            self._queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "error":
                    self._process_btn.config(state="normal")
                    self.status.set("Error.")
                    messagebox.showerror("Processing failed", payload)
                    return
                elif kind == "done":
                    self._on_processed(*payload)
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_processed(self, res_a, res_b, a_clusters, b_clusters):
        self._res_a, self._res_b = res_a, res_b
        self._a_by_id = {r.record_id: r for r in res_a.records}
        self._b_by_id = {r.record_id: r for r in res_b.records}
        self._a_clusters = a_clusters
        self._b_clusters = b_clusters
        # Count members once here rather than rescanning per card in
        # _cluster_tag: at 300 cards over 12,000 records that scan was 3.6M
        # dict iterations for every gallery repopulate, on the GUI thread.
        self._a_cluster_sizes = Counter(a_clusters.values())
        self._b_cluster_sizes = Counter(b_clusters.values())
        self._process_btn.config(state="normal")
        self.status.set(
            f"A: {len(res_a.records)} vehicles / {res_a.frame_count} frames   |   "
            f"B: {len(res_b.records)} vehicles / {res_b.frame_count} frames"
        )
        self._populate_a()
        self._last_selected_a = None
        self._populate_b_all()
        self._save_settings()

    def _collect_settings(self) -> dict:
        """Current values of everything ``mash_reid.settings`` remembers."""
        return {
            "dir_a": self.dir_a.get(),
            "dir_b": self.dir_b.get(),
            "model_key": self.model_key.get(),
            "device": self.device.get(),
            "threshold": self.threshold.get(),
            "det_conf": self.det_conf.get(),
            "min_travel": self.min_travel.get(),
            "max_travel": self.max_travel.get(),
            "use_gate": self.use_gate.get(),
            "one_to_one": self.one_to_one.get(),
            "extract_interval": self.extract_interval.get(),
            "last_video_dir": self.last_video_dir.get(),
            "cluster_same_point": self.cluster_same_point.get(),
            "timeline_samples": self.timeline_samples.get(),
            "ocr_time_pattern": self.ocr_time_pattern.get(),
        }

    def _save_settings(self):
        """Best-effort settings save -- see ``mash_reid.settings.save``."""
        try:
            settings.save(self._collect_settings())
        except Exception:
            log.warning("Could not save settings", exc_info=True)

    def _cluster_tag(self, clusters: dict[int, int], sizes: Counter, record_id: int) -> str:
        """Return a " •GrpN(xK)" suffix when this record shares a same-point
        cluster with other detections; empty string for a singleton.

        ``sizes`` is the precomputed ``Counter`` of cluster ids built in
        ``_on_processed``. Empty ``clusters`` (clustering switched off) makes
        this return "" for every record, which is how the feature is disabled
        without touching any caller.
        """
        cluster_id = clusters.get(record_id)
        if cluster_id is None:
            return ""
        size = sizes[cluster_id]
        return f"  •Grp{cluster_id}(x{size})" if size > 1 else ""

    def _subcaption(self, rec: VehicleRecord) -> str:
        """Second caption line: source frame filename + where its timestamp
        came from (ocr/filename/exif/mtime) -- lets you trace a result back
        to the exact frame and confirm at a glance whether OCR-corrected
        times are actually in effect.
        """
        name = os.path.basename(rec.frame_path)
        if len(name) > 22:
            name = name[:10] + "…" + name[-9:]
        return f"{name}  [{rec.timestamp_source}]"

    def _populate_a(self):
        self.gallery_a.clear()
        if not self._res_a:
            return
        records = self._res_a.records
        shown = records[: config.DEFAULT_MAX_GALLERY_THUMBNAILS]
        for rec in shown:
            cap = f"A#{rec.record_id}  {rec.timestamp:%Y-%m-%d %H:%M:%S}"
            cap += self._cluster_tag(self._a_clusters, self._a_cluster_sizes, rec.record_id)
            self.gallery_a.add_card(rec, cap, self._on_select_a, self._on_double,
                                    subcaption=self._subcaption(rec))
        if len(shown) < len(records):
            self.status.set(
                f"Point A: showing first {len(shown)} of {len(records)} vehicles "
                f"(earliest by time) -- narrowing the gallery isn't needed to match, "
                f"only to browse it."
            )

    def _show_all_b(self):
        self._last_selected_a = None
        self._populate_b_all()

    def _populate_b_all(self):
        """Default B view: every detected vehicle at point B, grouped by
        same-point cluster tag. Shown before any point-A vehicle is selected,
        and reachable again afterwards via the "Show all B" button.

        Rendering is capped at ``config.DEFAULT_MAX_GALLERY_THUMBNAILS`` --
        real galleries can hold thousands of vehicles, and a thumbnail card
        per vehicle (each an image load + several Tkinter widgets) would
        freeze the window well before that many render.
        """
        self.gallery_b.clear()
        if not self._res_b:
            self.b_view_label.set("Showing: all point B vehicles")
            return
        records = self._res_b.records
        shown = records[: config.DEFAULT_MAX_GALLERY_THUMBNAILS]
        if len(shown) < len(records):
            self.b_view_label.set(
                f"Showing: first {len(shown)} of {len(records)} point B vehicles")
        else:
            self.b_view_label.set("Showing: all point B vehicles")
        for rec in shown:
            cap = f"B#{rec.record_id}  {rec.timestamp:%Y-%m-%d %H:%M:%S}"
            cap += self._cluster_tag(self._b_clusters, self._b_cluster_sizes, rec.record_id)
            self.gallery_b.add_card(rec, cap, lambda r: None, self._on_double,
                                    subcaption=self._subcaption(rec))

    def _on_select_a(self, rec_a: VehicleRecord):
        if not self._res_b:
            return
        self._last_selected_a = rec_a
        results = matcher.match([rec_a], self._res_b.records, self._current_match_config())
        self.gallery_b.clear()
        self.b_view_label.set(f"Showing: matches for A#{rec_a.record_id}")
        result = results[0] if results else None
        if not result or not result.candidates:
            self.status.set(f"A#{rec_a.record_id}: no B match above threshold.")
            return
        self.status.set(f"A#{rec_a.record_id}: {len(result.candidates)} candidate(s).")
        for cand in result.candidates:
            rec_b = self._b_by_id[cand.b_record_id]
            dt = (rec_b.timestamp - rec_a.timestamp).total_seconds()
            cap = f"B#{rec_b.record_id}  sim={cand.similarity:.2f}  +{dt:.0f}s"
            cap += self._cluster_tag(self._b_clusters, self._b_cluster_sizes, rec_b.record_id)
            self.gallery_b.add_card(
                rec_b, cap, lambda r: None, self._on_double,
                on_confirm=lambda a=rec_a, b=rec_b, s=cand.similarity: self._label_pair(a, b, s, True),
                on_reject=lambda a=rec_a, b=rec_b, s=cand.similarity: self._label_pair(a, b, s, False),
                subcaption=self._subcaption(rec_b),
            )

    def _label_pair(self, rec_a: VehicleRecord, rec_b: VehicleRecord,
                    similarity: float, same: bool) -> None:
        """Save a user-confirmed/rejected A/B pair as labeled training data."""
        try:
            training_export.export_labeled_pair(
                config.DEFAULT_TRAINING_DATA_DIR, rec_a, rec_b, same, similarity)
        except Exception as exc:
            messagebox.showerror("Could not save training pair", str(exc), parent=self)
            return
        pos, neg = training_export.count_labeled_pairs(config.DEFAULT_TRAINING_DATA_DIR)
        self.status.set(
            f"Saved {'match' if same else 'non-match'} to {config.DEFAULT_TRAINING_DATA_DIR}/. "
            f"Collected so far: {pos} positive, {neg} negative."
        )

    def _on_double(self, rec: VehicleRecord):
        _show_full_frame(self, rec)

    def _rematch(self):
        # Re-run matching for the currently selected A vehicle when a slider or
        # toggle changes. No model work involved -- purely numeric, so instant.
        # If nothing is selected (the "all B" default view), there is nothing
        # threshold-dependent to refresh.
        if self._last_selected_a is not None and self._res_b is not None:
            self._on_select_a(self._last_selected_a)


class VideoExtractDialog(tk.Toplevel):
    """Modal dialog: pick one or more videos, extract timestamped frames for
    one point.

    A single video keeps the original workflow (editable start time, output
    defaults to a name derived from that video). Selecting several videos at
    once treats them as clips of the same point -- e.g. a camera that splits
    recordings every hour -- and combines them into one output folder; each
    clip resolves its own start time from its own filename, so the Start
    time field doesn't apply and is disabled.

    On success the output folder is written back into the point's folder
    entry so the user can immediately Process it.
    """

    def __init__(self, parent, point, folder_var):
        super().__init__(parent)
        self.title(f"Extract frames for point {point}")
        self.point = point
        self.folder_var = folder_var
        # Kept for two settings ReIDApp remembers across runs -- the last
        # interval used and the last folder browsed for a video. Read with
        # getattr() defaults so this dialog still works if ever opened with
        # a parent that isn't a ReIDApp (e.g. in a standalone test).
        self._app = parent
        self._queue: queue.Queue = queue.Queue()
        self.video_paths: list[str] = []

        self.out_dir = tk.StringVar()
        default_interval = getattr(parent, "extract_interval", None)
        self.interval = tk.DoubleVar(value=default_interval.get() if default_interval else 1.0)
        self.start_time = tk.StringVar()
        self.status = tk.StringVar(value="Choose one or more videos to begin.")

        self._build()
        self.transient(parent)
        self.grab_set()

    def _build(self):
        pad = {"padx": 8, "pady": 4}

        vrow = ttk.Frame(self)
        vrow.pack(fill="x", **pad)
        ttk.Label(vrow, text="Video(s)", width=10).pack(side="left", anchor="n")
        self._video_list = tk.Listbox(vrow, height=4, width=48)
        self._video_list.pack(side="left", fill="x", expand=True)
        vbtns = ttk.Frame(vrow)
        vbtns.pack(side="left", padx=4)
        ttk.Button(vbtns, text="Browse...", command=self._pick_video).pack(fill="x")
        ttk.Button(vbtns, text="Clear", command=self._clear_videos).pack(fill="x", pady=(2, 0))

        orow = ttk.Frame(self)
        orow.pack(fill="x", **pad)
        ttk.Label(orow, text="Output", width=10).pack(side="left")
        ttk.Entry(orow, textvariable=self.out_dir, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(orow, text="Browse...", command=self._pick_out).pack(side="left", padx=4)

        srow = ttk.Frame(self)
        srow.pack(fill="x", **pad)
        ttk.Label(srow, text="Start time", width=10).pack(side="left")
        self._start_entry = ttk.Entry(srow, textvariable=self.start_time, width=24)
        self._start_entry.pack(side="left")
        self._start_hint = ttk.Label(
            srow, text="(auto from filename; edit as YYYY-MM-DD HH:MM:SS)")
        self._start_hint.pack(side="left", padx=6)

        irow = ttk.Frame(self)
        irow.pack(fill="x", **pad)
        ttk.Label(irow, text="Interval (s)", width=10).pack(side="left")
        ttk.Spinbox(irow, from_=0.1, to=60.0, increment=0.5, textvariable=self.interval,
                    width=8).pack(side="left")

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", **pad)

        brow = ttk.Frame(self)
        brow.pack(fill="x", **pad)
        self._extract_btn = ttk.Button(brow, text="Extract", command=self._start)
        self._extract_btn.pack(side="right")
        ttk.Button(brow, text="Close", command=self.destroy).pack(side="right", padx=4)

    def _pick_video(self):
        last_dir_var = getattr(self._app, "last_video_dir", None)
        initial_dir = last_dir_var.get() if last_dir_var and last_dir_var.get() else None
        paths = filedialog.askopenfilenames(
            parent=self,
            initialdir=initial_dir,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.video_paths = list(paths)
        self._video_list.delete(0, "end")
        for p in self.video_paths:
            self._video_list.insert("end", os.path.basename(p))
        if last_dir_var is not None:
            last_dir_var.set(os.path.dirname(self.video_paths[0]))

        if len(self.video_paths) == 1:
            video = self.video_paths[0]
            self.out_dir.set(video_extractor.default_output_dir(video, self.point))
            start, source = video_extractor.resolve_start_time(video, None)
            self.start_time.set(start.strftime("%Y-%m-%d %H:%M:%S"))
            self._start_entry.config(state="normal")
            self.status.set(f"Start time from {source}. Adjust if needed, then Extract.")
        else:
            # Multiple clips: each resolves its own start time, so a single
            # shared override doesn't make sense here -- disable the field.
            self.out_dir.set(video_extractor.default_output_dir_multi(self.video_paths, self.point))
            self.start_time.set("")
            self._start_entry.config(state="disabled")
            self.status.set(
                f"{len(self.video_paths)} clips selected. Each resolves its own "
                f"start time from its filename; they'll be combined into one folder.")

    def _clear_videos(self):
        self.video_paths = []
        self._video_list.delete(0, "end")
        self._start_entry.config(state="normal")
        self.status.set("Choose one or more videos to begin.")

    def _pick_out(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.out_dir.set(path)

    def _parse_start(self):
        text = self.start_time.get().strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                from datetime import datetime
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        raise ValueError(f"Bad start time '{text}'. Use YYYY-MM-DD HH:MM:SS.")

    def _start(self):
        out = self.out_dir.get().strip()
        if not self.video_paths:
            messagebox.showerror("No video", "Please choose at least one video file.", parent=self)
            return
        if not all(os.path.isfile(v) for v in self.video_paths):
            messagebox.showerror("No video", "One or more selected videos no longer exist.", parent=self)
            return
        if not out:
            messagebox.showerror("No output", "Please choose an output folder.", parent=self)
            return

        start_dt = None
        if len(self.video_paths) == 1:
            try:
                start_dt = self._parse_start()
            except ValueError as exc:
                messagebox.showerror("Invalid start time", str(exc), parent=self)
                return

        interval = self.interval.get()
        default_interval = getattr(self._app, "extract_interval", None)
        if default_interval is not None:
            default_interval.set(interval)  # remember for the next time this dialog opens

        self._extract_btn.config(state="disabled")
        self.status.set("Extracting frames...")
        thread = threading.Thread(
            target=self._worker, args=(list(self.video_paths), out, start_dt, interval),
            daemon=True,
        )
        thread.start()
        self.after(100, self._poll)

    def _worker(self, videos, out, start_dt, interval):
        try:
            def progress(done, total, msg):
                total_str = str(total) if total else "?"
                self._queue.put(("status", f"[{done}/{total_str}] {msg}"))

            if len(videos) == 1:
                written = video_extractor.extract_frames(
                    videos[0], out, self.point, interval_seconds=interval,
                    start_time=start_dt, progress=progress,
                )
            else:
                written = video_extractor.extract_many(
                    videos, out, self.point, interval_seconds=interval, progress=progress,
                )
            self._queue.put(("done", (out, len(written))))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "error":
                    self._extract_btn.config(state="normal")
                    messagebox.showerror("Extraction failed", payload, parent=self)
                    return
                elif kind == "done":
                    out, count = payload
                    self._extract_btn.config(state="normal")
                    self.folder_var.set(out)  # feed it back to the main window
                    self.status.set(f"Done: {count} frames -> {out}")
                    messagebox.showinfo(
                        "Extraction complete",
                        f"Wrote {count} frames to:\n{out}\n\n"
                        f"This folder is now set as point {self.point}.",
                        parent=self,
                    )
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)


class ModelManagerDialog(tk.Toplevel):
    """Check / download / update / remove detection-model weights.

    Lists the catalog with install status; the selected model can be downloaded,
    refreshed to the latest weights, or deleted. Network work runs in a worker
    thread so the UI stays responsive, with status pushed back via a queue.
    """

    COLUMNS = ("family", "size", "approx", "status")

    def __init__(self, parent, on_close=None):
        super().__init__(parent)
        self.title("Manage detection models")
        self.geometry("760x420")
        self._on_close = on_close
        self._queue: queue.Queue = queue.Queue()
        self.status = tk.StringVar()

        self._build()
        self._refresh_rows()
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        info = ttk.Label(
            self,
            text=("★ = newer generation, recommended. Weights download to "
                  f"{model_manager.default_models_dir()}.\n"
                  f"ultralytics {model_manager.ultralytics_version() or 'not installed'} "
                  "— upgrade the package then Update for newer weights."),
            justify="left", font=("TkDefaultFont", 8), padding=(8, 6, 8, 0))
        info.pack(fill="x")

        tree_frame = ttk.Frame(self, padding=8)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=self.COLUMNS, show="tree headings",
                                 selectmode="browse")
        self.tree.heading("#0", text="Model")
        self.tree.heading("family", text="Family")
        self.tree.heading("size", text="Size")
        self.tree.heading("approx", text="~MB")
        self.tree.heading("status", text="Status")
        self.tree.column("#0", width=200)
        self.tree.column("family", width=90, anchor="center")
        self.tree.column("size", width=90, anchor="center")
        self.tree.column("approx", width=70, anchor="e")
        self.tree.column("status", width=160, anchor="center")
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w",
                  padding=4).pack(fill="x", padx=8)

        btns = ttk.Frame(self, padding=8)
        btns.pack(fill="x")
        self._dl_btn = ttk.Button(btns, text="Download", command=lambda: self._run("download"))
        self._dl_btn.pack(side="left")
        self._up_btn = ttk.Button(btns, text="Update (latest)", command=lambda: self._run("update"))
        self._up_btn.pack(side="left", padx=4)
        self._rm_btn = ttk.Button(btns, text="Remove", command=self._remove)
        self._rm_btn.pack(side="left")
        ttk.Button(btns, text="Close", command=self._close).pack(side="right")

    def _refresh_rows(self):
        selected = self._selected_key()
        self.tree.delete(*self.tree.get_children())
        for row in model_manager.status():
            name = ("★ " if row["recommended"] else "") + row["key"]
            state = f"downloaded ({row['size_mb']:.0f} MB)" if row["installed"] else "not downloaded"
            self.tree.insert("", "end", iid=row["key"], text=name,
                             values=(row["family"], row["size"],
                                     f"{row['approx_mb']:.0f}", state))
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)

    def _selected_key(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        for b in (self._dl_btn, self._up_btn, self._rm_btn):
            b.config(state=state)

    def _remove(self):
        key = self._selected_key()
        if not key:
            self.status.set("Select a model first.")
            return
        if model_manager.remove(key):
            self.status.set(f"Removed {key}.")
        else:
            self.status.set(f"{key} was not downloaded.")
        self._refresh_rows()

    def _run(self, action):
        key = self._selected_key()
        if not key:
            self.status.set("Select a model first.")
            return
        self._set_busy(True)
        self.status.set(f"{action.capitalize()} {key}...")
        thread = threading.Thread(target=self._worker, args=(action, key), daemon=True)
        thread.start()
        self.after(100, self._poll)

    def _worker(self, action, key):
        try:
            fn = model_manager.update if action == "update" else model_manager.download
            fn(key, progress=lambda msg: self._queue.put(("status", msg)))
            self._queue.put(("done", f"{action.capitalize()} complete: {key}"))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "error":
                    self._set_busy(False)
                    self._refresh_rows()
                    messagebox.showerror("Model download failed", payload, parent=self)
                    return
                elif kind == "done":
                    self._set_busy(False)
                    self.status.set(payload)
                    self._refresh_rows()
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _close(self):
        if self._on_close:
            self._on_close()
        self.destroy()


class PatternManagerDialog(tk.Toplevel):
    """Review and delete user-saved OCR time patterns.

    Built-in patterns (``config.BUILTIN_OCR_TIME_PATTERNS``) aren't shown
    here -- they're fixed reference choices, not something to edit or
    delete. Only the ones a user has explicitly named via "Save as..." in
    the review dialog's pattern picker live in this list.
    """

    def __init__(self, parent, on_change=None):
        super().__init__(parent)
        self.title("Manage saved time patterns")
        self.geometry("560x320")
        self.on_change = on_change

        self.tree = ttk.Treeview(self, columns=("name", "pattern"), show="headings")
        self.tree.heading("name", text="Name")
        self.tree.heading("pattern", text="Pattern")
        self.tree.column("name", width=160, anchor="w")
        self.tree.column("pattern", width=360, anchor="w")
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        ttk.Button(footer, text="Delete", command=self._delete_selected).pack(side="left")
        ttk.Button(footer, text="Close", command=self.destroy).pack(side="right")

        self._refresh()
        self.transient(parent)
        self.grab_set()

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for name, pattern in sorted(timestamp_ocr.load_saved_patterns().items()):
            self.tree.insert("", "end", iid=name, values=(name, pattern))

    def _delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        name = selection[0]
        if not messagebox.askyesno("Delete pattern", f"Delete saved pattern \"{name}\"?",
                                   parent=self):
            return
        saved = timestamp_ocr.load_saved_patterns()
        saved.pop(name, None)
        timestamp_ocr.save_patterns(saved)
        self._refresh()
        if self.on_change:
            self.on_change()


class RegionPickerDialog(tk.Toplevel):
    """Click-drag a rectangle around the clock on one frame.

    Existing to fix exactly the failure a mismatched region produces: EasyOCR
    finds a plausible-looking box of text and, if it never happens to contain
    a date/time, ``find_overlay_region`` never anchors on the real clock and
    every reading comes from whatever else has text -- a camera-model
    watermark, a plate. Letting the user point at the clock once sidesteps
    that guess entirely, and is faster too (no full-frame probing pass).

    Result is delivered via ``on_done(region_or_None)`` rather than a return
    value, matching this file's other dialogs -- none of them use
    ``wait_window``, so a plain return value isn't available to the caller.
    """

    MAX_DISPLAY = (1000, 700)

    def __init__(self, parent, image_path: str, on_done):
        super().__init__(parent)
        self.title("Mark the clock's position")
        self.on_done = on_done
        self._start = None
        self._rect_id = None
        self._box = None  # (x1, y1, x2, y2) in canvas/display pixels

        from PIL import Image, ImageTk

        img = Image.open(image_path).convert("RGB")
        max_w, max_h = self.MAX_DISPLAY
        # Only ever shrink to fit -- upscaling would make the drawn box
        # imprecise relative to the actual pixels it needs to map back to.
        self._scale = min(max_w / img.width, max_h / img.height, 1.0)
        if self._scale < 1.0:
            img = img.resize((max(1, int(img.width * self._scale)),
                             max(1, int(img.height * self._scale))))
        self._photo = ImageTk.PhotoImage(img)

        ttk.Label(self, text="Click and drag a box around the on-screen clock, "
                             "then click \"Use this box\".",
                  wraplength=self._photo.width()).pack(padx=8, pady=(8, 4))

        self.canvas = tk.Canvas(self, width=self._photo.width(), height=self._photo.height(),
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=8, pady=8)
        self._use_btn = ttk.Button(footer, text="Use this box", state="disabled",
                                   command=self._confirm)
        self._use_btn.pack(side="right")
        ttk.Button(footer, text="Cancel", command=self._cancel).pack(side="right", padx=4)

        self.transient(parent)
        self.grab_set()

    def _on_press(self, event):
        self._start = (event.x, event.y)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if self._start is None:
            return
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        x0, y0 = self._start
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#ff3030", width=2)

    def _on_release(self, event):
        if self._start is None:
            return
        x0, y0 = self._start
        w, h = self._photo.width(), self._photo.height()
        x1, x2 = sorted((max(0, min(x0, w)), max(0, min(event.x, w))))
        y1, y2 = sorted((max(0, min(y0, h)), max(0, min(event.y, h))))
        if x2 - x1 < 4 or y2 - y1 < 4:
            # Too small to be a deliberate drag (a stray click) -- ignore it
            # rather than accepting a box that can't contain any text.
            self._box = None
            self._use_btn.config(state="disabled")
            return
        self._box = (x1, y1, x2, y2)
        self._use_btn.config(state="normal")

    def _confirm(self):
        if self._box is None:
            return
        x1, y1, x2, y2 = self._box
        scale = self._scale or 1.0
        region = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
        self.on_done(region)
        self.destroy()

    def _cancel(self):
        self.on_done(None)
        self.destroy()


class TimelineReviewDialog(tk.Toplevel):
    """Confirm (and correct) the fitted clock before it is written out.

    Deliberately a summary plus two lists rather than a row per frame: the
    fit reduces 12,000 frames to one rate and a start time per clip, so that
    is what there is to check. A per-frame table would also mean thousands of
    image-bearing widgets, which is exactly what makes the main gallery slow.

    The one thing a robust fit cannot catch is a *systematic* misread -- if
    the overlay font makes every "8" read as "6", the readings stay mutually
    consistent and the residuals look perfect. That is why the samples list
    shows the filename time beside the read time: the gap between the two
    clocks is the thing a human can actually judge.

    Follows this file's dialog convention: Toplevel + _build() + transient +
    grab_set, no wait_window; the result is written to disk on Apply rather
    than returned.
    """

    CLIP_COLUMNS = ("clip", "frames", "samples", "start", "mad", "worst", "offset")
    SAMPLE_COLUMNS = ("n", "clip", "x", "filename", "text", "conf", "read", "residual", "in")

    def __init__(self, parent, point, scan):
        super().__init__(parent)
        self.title(f"Review timestamps - point {point}")
        self.geometry("1180x680")
        self.point = point
        self.scan = scan
        self._app = parent
        self._samples = list(scan.samples)
        self._fit = scan.fit
        self._photo = None  # keeps the detail-pane crop from being GC'd
        self._selected = None

        self.summary = tk.StringVar()
        self.warnings = tk.StringVar()
        self.header = tk.StringVar()
        self.manual = tk.StringVar()
        self.detail_caption = tk.StringVar()
        # Shared with the main window (like VideoExtractDialog's shared vars)
        # via getattr with a fallback, so this dialog stays standalone-testable.
        parent_pattern = getattr(parent, "ocr_time_pattern", None)
        self.pattern_text = tk.StringVar(value=parent_pattern.get() if parent_pattern else "")

        self._build()
        self._refit()
        self.transient(parent)
        self.grab_set()

    # --- construction ------------------------------------------------------

    def _build(self):
        pad = {"padx": 8, "pady": 4}

        ttk.Label(self, textvariable=self.header, font=("TkDefaultFont", 9, "bold")
                  ).pack(anchor="w", **pad)
        ttk.Label(self, textvariable=self.summary).pack(anchor="w", **pad)
        self._warn_label = ttk.Label(self, textvariable=self.warnings,
                                     foreground="#b00000", wraplength=1140, justify="left")
        self._warn_label.pack(anchor="w", **pad)

        # The one place a time pattern is picked, typed, saved or managed --
        # this used to also live as a plain Entry on the main window, which
        # just meant two places to look for the same setting. Editable (not
        # readonly) so a freehand token template still works; picking one of
        # the "pattern   (label)" entries snaps the box back to the bare,
        # compilable pattern immediately (see _on_pattern_selected).
        prow = ttk.Frame(self)
        prow.pack(fill="x", **pad)
        ttk.Label(prow, text="Time pattern:").pack(side="left")
        self.pattern_combo = ttk.Combobox(prow, textvariable=self.pattern_text, width=30)
        self.pattern_combo.pack(side="left", fill="x", expand=True, padx=4)
        self.pattern_combo.bind("<<ComboboxSelected>>", self._on_pattern_selected)
        self._refresh_pattern_choices()
        ttk.Button(prow, text="Re-parse", command=self._apply_custom_pattern).pack(side="left")
        ttk.Button(prow, text="Save as...", command=self._save_pattern_as).pack(
            side="left", padx=(4, 0))
        ttk.Button(prow, text="Manage...", command=self._manage_patterns).pack(
            side="left", padx=(4, 0))
        ttk.Button(prow, text="?", width=2, command=self._show_pattern_help).pack(
            side="left", padx=(4, 0))

        clips_frame = ttk.Labelframe(self, text="Clips")
        clips_frame.pack(fill="x", **pad)
        self.clips_tree = ttk.Treeview(clips_frame, columns=self.CLIP_COLUMNS,
                                       show="headings", height=4)
        for col, title, width in (
                ("clip", "Clip", 60), ("frames", "Frames", 80), ("samples", "Read", 80),
                ("start", "Start time", 190), ("mad", "Typical error", 110),
                ("worst", "Worst", 90), ("offset", "Manual offset", 110)):
            self.clips_tree.heading(col, text=title)
            self.clips_tree.column(col, width=width, anchor="w")
        self.clips_tree.tag_configure("borrowed", foreground="#b06000")
        self.clips_tree.pack(fill="x", padx=4, pady=4)

        nudge = ttk.Frame(clips_frame)
        nudge.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(nudge, text="Shift selected clip",
                   command=lambda: self._nudge(selected_only=True)).pack(side="left")
        ttk.Button(nudge, text="Shift all clips",
                   command=lambda: self._nudge(selected_only=False)).pack(side="left", padx=4)
        self.nudge_seconds = tk.DoubleVar(value=0.0)
        ttk.Spinbox(nudge, from_=-3600.0, to=3600.0, increment=0.5, width=10,
                    textvariable=self.nudge_seconds).pack(side="left", padx=4)
        ttk.Label(nudge, text="seconds").pack(side="left")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, **pad)

        samples_frame = ttk.Labelframe(body, text="Sampled frames")
        samples_frame.pack(side="left", fill="both", expand=True)
        self.samples_tree = ttk.Treeview(samples_frame, columns=self.SAMPLE_COLUMNS,
                                         show="headings")
        for col, title, width in (
                ("n", "#", 40), ("clip", "Clip", 45), ("x", "Frame", 75),
                ("filename", "Filename clock", 150), ("text", "OCR text", 190),
                ("conf", "Conf", 55), ("read", "Read clock", 150),
                ("residual", "Off by", 70), ("in", "Used", 50)):
            self.samples_tree.heading(col, text=title)
            self.samples_tree.column(col, width=width, anchor="w")
        self.samples_tree.tag_configure("outlier", foreground="#b00000")
        self.samples_tree.tag_configure("edited", foreground="#0050b0")
        self.samples_tree.tag_configure("unread", foreground="#909090")
        self.samples_tree.bind("<<TreeviewSelect>>", self._on_select_sample)
        # grid rather than pack, so a horizontal scrollbar can sit under the
        # tree: the columns' combined width (OCR text, filename/read clocks,
        # ...) comfortably exceeds this pane's share of the dialog once the
        # detail pane on the right claims its space, and a Treeview never
        # wraps or reflows -- without this, the right-hand columns were
        # simply unreachable rather than merely narrow.
        vsbar = ttk.Scrollbar(samples_frame, orient="vertical",
                             command=self.samples_tree.yview)
        hsbar = ttk.Scrollbar(samples_frame, orient="horizontal",
                             command=self.samples_tree.xview)
        self.samples_tree.configure(yscrollcommand=vsbar.set, xscrollcommand=hsbar.set)
        self.samples_tree.grid(row=0, column=0, sticky="nsew")
        vsbar.grid(row=0, column=1, sticky="ns")
        hsbar.grid(row=1, column=0, sticky="ew")
        samples_frame.grid_rowconfigure(0, weight=1)
        samples_frame.grid_columnconfigure(0, weight=1)

        detail = ttk.Labelframe(body, text="Selected frame")
        detail.pack(side="right", fill="y", padx=(8, 0))
        self._crop_label = ttk.Label(detail)
        self._crop_label.pack(padx=6, pady=(6, 0))
        self._zoom_btn = ttk.Button(detail, text="Zoom in...", command=self._zoom_selected,
                                    state="disabled")
        self._zoom_btn.pack(padx=6, pady=(2, 6))
        ttk.Label(detail, textvariable=self.detail_caption, wraplength=380,
                  justify="left", font=("TkDefaultFont", 8)).pack(anchor="w", padx=6)

        ttk.Label(detail, text="Readings the OCR considered:").pack(anchor="w", padx=6, pady=(8, 0))
        self.candidate_combo = ttk.Combobox(detail, state="readonly", width=52)
        self.candidate_combo.pack(padx=6, pady=2)

        ttk.Label(detail, text="Or type the correct time (YYYY-MM-DD HH:MM:SS):"
                  ).pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Entry(detail, textvariable=self.manual, width=54).pack(padx=6, pady=2)

        buttons = ttk.Frame(detail)
        buttons.pack(fill="x", padx=6, pady=6)
        ttk.Button(buttons, text="Use this", command=self._use_selection).pack(side="left")
        ttk.Button(buttons, text="Ignore frame", command=self._ignore_selected).pack(side="left", padx=4)
        ttk.Button(buttons, text="Reset", command=self._reset_selected).pack(side="left")

        footer = ttk.Frame(self)
        footer.pack(fill="x", **pad)
        self._apply_btn = ttk.Button(footer, text="Apply and write times", command=self._apply)
        self._apply_btn.pack(side="right")
        ttk.Button(footer, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(footer, text="Read every frame instead (slow)",
                   command=self._full_ocr).pack(side="left")
        self._mark_region_btn = ttk.Button(footer, text="Mark clock position...",
                                           command=self._mark_region)
        self._mark_region_btn.pack(side="left", padx=(4, 0))
        self._resample_btn = ttk.Button(footer, text="Re-sample clock (fresh OCR)...",
                                        command=lambda: self._rescan_with_region(None))
        self._resample_btn.pack(side="left", padx=(4, 0))

    # --- fit + rendering ---------------------------------------------------

    def _refit(self):
        """Recompute the fit from the current readings and repaint.

        Cheap enough to run on every edit (a few hundred pairwise slopes), so
        the user watches the residuals move as they correct a reading --
        which is what makes the review meaningful rather than ceremonial.
        """
        from mash_reid import timeline

        self._fit = timeline.fit_timeline(
            self._samples, x_kind=self.scan.x_kind, all_keys=self.scan.keys)
        self._render()

    def _render(self):
        fit = self._fit
        clips = len({k.clip for k in self.scan.keys})
        axis = ("source frame index" if self.scan.x_kind == "frame_index"
                else "position in folder")
        self.header.set(f"{self.scan.folder}   -   {len(self.scan.keys)} frames, "
                        f"{clips} clip(s), timed by {axis}")

        rate = (f"{fit.implied_fps:.2f} fps" if fit.implied_fps
                else f"{fit.slope:.4f} s/frame")
        read = sum(1 for s in self._samples if s.chosen)
        self.summary.set(
            f"Rate {rate}   |   read {read}/{len(self._samples)} sampled clocks   |   "
            f"{fit.n_inliers}/{fit.n_samples} agree ({fit.inlier_fraction:.0%})   |   "
            f"typical error {fit.residual_mad:.2f} s   |   worst {fit.max_abs_residual:.2f} s")
        self.warnings.set("\n".join(fit.warnings))

        self.clips_tree.delete(*self.clips_tree.get_children())
        frames_per_clip = Counter(k.clip for k in self.scan.keys)
        for clip in sorted(fit.clips):
            cf = fit.clips[clip]
            self.clips_tree.insert(
                "", "end", iid=str(clip), tags=() if cf.fitted else ("borrowed",),
                values=(clip, frames_per_clip.get(clip, 0),
                        f"{cf.n_inliers}/{cf.n_samples}" if cf.fitted else "borrowed",
                        cf.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{cf.residual_mad:.2f} s", f"{cf.max_abs_residual:.2f} s",
                        f"{cf.user_offset_seconds:+.1f} s"))

        self.samples_tree.delete(*self.samples_tree.get_children())
        for i, s in enumerate(self._samples):
            residual = fit.residuals.get(s.name)
            if s.ignored:
                tag = "unread"
            elif s.chosen is None:
                tag = "unread"
            elif s.edited:
                tag = "edited"
            elif s.name not in fit.inliers:
                tag = "outlier"
            else:
                tag = ""
            best = s.candidates[0].text if s.candidates else ""
            conf = f"{s.candidates[0].confidence:.2f}" if s.candidates else ""
            self.samples_tree.insert(
                "", "end", iid=str(i), tags=(tag,) if tag else (),
                values=(i + 1, s.clip, s.x,
                        s.filename_time.strftime("%H:%M:%S") if s.filename_time else "-",
                        best[:40],
                        conf,
                        s.chosen.strftime("%Y-%m-%d %H:%M:%S") if s.chosen else "unread",
                        f"{residual:+.1f} s" if residual is not None else "-",
                        "no" if (s.ignored or s.chosen is None)
                        else ("yes" if s.name in fit.inliers else "no")))

    def _apply_custom_pattern(self):
        """Re-parse every sample's already-OCR'd text with a new pattern.

        No OCR call, no image decode -- the raw text each candidate came
        from is already sitting in ``TimeSample.candidates`` from the scan.
        This is what lets a wrong initial read get fixed by describing the
        overlay's actual layout, without a slow re-read of the clock.

        Samples the user already corrected by hand (``edited``) are left
        alone -- an explicit correction should never be silently overwritten
        by a pattern change.
        """
        from dataclasses import replace

        text = timestamp_ocr.strip_pattern_label(self.pattern_text.get())
        pattern = None
        if text:
            try:
                pattern = timestamp_ocr.compile_custom_pattern(text)
            except ValueError as exc:
                messagebox.showerror("Bad time pattern", str(exc), parent=self)
                return

        updated = []
        for s in self._samples:
            if s.edited:
                updated.append(s)
                continue
            candidates = tuple(timestamp_ocr.reparse_candidates(s.candidates, pattern))
            chosen = next((c.timestamp for c in candidates if c.timestamp), None)
            updated.append(replace(s, candidates=candidates, chosen=chosen))
        self._samples = updated

        # Remembered immediately (not only on Apply) so the *other* point's
        # first scan already benefits, without the user having to redo this.
        self.pattern_text.set(text)
        if hasattr(self._app, "ocr_time_pattern"):
            self._app.ocr_time_pattern.set(text)

        self._refit()

    def _on_pattern_selected(self, _event=None):
        """Snap the box to the bare pattern right after picking a labeled entry.

        The dropdown shows "pattern   (label)" so a choice is identifiable;
        once picked, the box should hold just the compilable text, ready to
        test or edit further -- not the annotation along with it.
        """
        self.pattern_text.set(timestamp_ocr.strip_pattern_label(self.pattern_text.get()))

    def _refresh_pattern_choices(self):
        self.pattern_combo.config(values=timestamp_ocr.pattern_choices())

    def _save_pattern_as(self):
        """Name and persist the box's current pattern for reuse later."""
        text = timestamp_ocr.strip_pattern_label(self.pattern_text.get())
        if not text:
            messagebox.showinfo("Nothing to save", "Type or pick a pattern first.", parent=self)
            return
        try:
            timestamp_ocr.compile_custom_pattern(text)
        except ValueError as exc:
            messagebox.showerror("Bad time pattern", str(exc), parent=self)
            return

        from tkinter import simpledialog
        name = simpledialog.askstring("Save time pattern", "Name for this pattern:", parent=self)
        if not name:
            return
        saved = timestamp_ocr.load_saved_patterns()
        saved[name] = text
        timestamp_ocr.save_patterns(saved)
        self._refresh_pattern_choices()
        self.pattern_text.set(text)

    def _manage_patterns(self):
        PatternManagerDialog(self, on_change=self._refresh_pattern_choices)

    def _show_pattern_help(self):
        messagebox.showinfo(
            "Time pattern",
            "Pick one of the ready-made patterns from the list, or describe your "
            "camera's exact layout using these tokens (any other character, "
            "including spaces and punctuation, is matched literally):\n\n"
            "  YYYY  4-digit year        DD  day\n"
            "  YY    2-digit year        HH  hour (24-hour)\n"
            "  MO    month               hh  hour (12-hour, needs AP)\n"
            "  MI    minute              SS  second\n"
            "  AP    AM/PM\n\n"
            "Example: if your overlay shows \"2026-07-24 17.40.50\", the pattern "
            "is:\n\n    YYYY-MO-DD HH.MI.SS\n\n"
            "Use \"Save as...\" to keep a pattern you've typed for next time; "
            "\"Manage...\" lets you review or delete saved patterns. Leave the "
            "box empty to use only the built-in parser.", parent=self)

    # --- detail pane -------------------------------------------------------

    def _on_select_sample(self, _event=None):
        selection = self.samples_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        self._selected = index
        sample = self._samples[index]

        self._show_crop(sample)
        gap = ""
        if sample.chosen and sample.filename_time:
            delta = (sample.chosen - sample.filename_time).total_seconds()
            gap = f"\nOverlay clock is {delta:+.0f} s from the filename clock."
        self.detail_caption.set(f"{sample.name}{gap}")

        values = [f"{c.timestamp:%Y-%m-%d %H:%M:%S}   ({c.confidence:.2f})   {c.text[:40]}"
                  if c.timestamp else f"(unreadable)   ({c.confidence:.2f})   {c.text[:40]}"
                  for c in sample.candidates]
        self.candidate_combo.config(values=values)
        if values:
            self.candidate_combo.current(0)
        else:
            self.candidate_combo.set("")
        self.manual.set(sample.chosen.strftime("%Y-%m-%d %H:%M:%S") if sample.chosen else "")

    def _show_crop(self, sample):
        """Render the clock region for the selected frame, if we kept it.

        Converted here rather than in the worker because ImageTk needs a live
        Tk root; the raw BGR arrays travel from the scan thread untouched.
        """
        crop = self.scan.crops.get(sample.name)
        if crop is None:
            self._crop_label.config(image="")
            self._photo = None
            self._zoom_btn.config(state="disabled")
            return
        try:
            from PIL import Image, ImageTk

            img = Image.fromarray(crop[:, :, ::-1])  # cv2 gives BGR
            if img.width > 420:
                scale = 420 / img.width
                img = img.resize((420, max(1, int(img.height * scale))))
            self._photo = ImageTk.PhotoImage(img)
            self._crop_label.config(image=self._photo)
            self._zoom_btn.config(state="normal")
        except Exception:
            log.warning("Could not render clock crop for %s", sample.name, exc_info=True)
            self._crop_label.config(image="")
            self._photo = None
            self._zoom_btn.config(state="disabled")

    def _zoom_selected(self):
        """Open the current sample's full frame, scrollable and wheel-zoomable.

        Shows the whole frame (with the detected/marked region outlined)
        rather than just the small crop the inline preview uses -- a clock
        sitting near the frame's edge, or a region drawn a touch too tight,
        would otherwise be invisible no matter how much the crop is enlarged.
        """
        if self._selected is None:
            return
        sample = self._samples[self._selected]
        image_path = os.path.join(self.scan.folder, sample.name)
        if not os.path.isfile(image_path):
            messagebox.showerror("Frame not found", f"Could not find {image_path}", parent=self)
            return
        FrameZoomDialog(self, image_path, f"{self.point}  {sample.name}",
                        region=self.scan.region)

    # --- edits -------------------------------------------------------------

    def _replace_selected(self, **changes):
        from dataclasses import replace

        if self._selected is None:
            messagebox.showinfo("No frame selected",
                                "Pick a row in the sampled frames list first.", parent=self)
            return False
        self._samples[self._selected] = replace(self._samples[self._selected], **changes)
        return True

    def _use_selection(self):
        """Apply the typed time if given, otherwise the chosen candidate."""
        from datetime import datetime  # lazy, matching this module's convention

        if self._selected is None:
            messagebox.showinfo("No frame selected",
                                "Pick a row in the sampled frames list first.", parent=self)
            return

        text = self.manual.get().strip()
        if text:
            try:
                chosen = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                messagebox.showerror("Bad time", f"Could not read '{text}'. "
                                     "Use YYYY-MM-DD HH:MM:SS.", parent=self)
                return
        else:
            index = self.candidate_combo.current()
            candidates = self._samples[self._selected].candidates
            if index < 0 or index >= len(candidates):
                return
            chosen = candidates[index].timestamp
            if chosen is None:
                messagebox.showerror("Unreadable", "That reading did not parse as a time; "
                                     "type the correct one instead.", parent=self)
                return

        if self._replace_selected(chosen=chosen, edited=True, ignored=False):
            self._refit()

    def _ignore_selected(self):
        if self._replace_selected(ignored=True):
            self._refit()

    def _reset_selected(self):
        """Back to whatever the OCR originally picked for this frame."""
        if self._selected is None:
            return
        sample = self._samples[self._selected]
        original = next((c.timestamp for c in sample.candidates if c.timestamp), None)
        if self._replace_selected(chosen=original, edited=False, ignored=False):
            self._refit()

    def _nudge(self, selected_only: bool):
        from mash_reid import timeline

        seconds = float(self.nudge_seconds.get())
        if not seconds:
            return
        clip = None
        if selected_only:
            selection = self.clips_tree.selection()
            if not selection:
                messagebox.showinfo("No clip selected",
                                    "Pick a clip row first, or use 'Shift all clips'.",
                                    parent=self)
                return
            clip = int(selection[0])
        self._fit = timeline.nudge(self._fit, seconds, clip=clip)
        self._render()

    # --- outcomes ----------------------------------------------------------

    def _apply(self):
        from mash_reid import timeline

        if not self._fit.ok and not messagebox.askyesno(
                "Fit looks wrong",
                "\n".join(self._fit.warnings) + "\n\nWrite these times anyway?",
                parent=self):
            return
        try:
            frames = timeline.apply_fit(self._fit, self.scan.keys)
            doc = timeline.build_document(self._fit, self._samples, frames,
                                          confirmed_by_user=True)
            # Carried alongside the fit so a later "Fix times..." click can
            # reopen this exact review (region, pattern, corrections) without
            # any new OCR -- see timestamp_ocr.load_scan_for_review.
            doc["region"] = list(self.scan.region) if self.scan.region else None
            doc["region_is_manual"] = self.scan.region_is_manual
            doc["custom_pattern"] = timestamp_ocr.strip_pattern_label(self.pattern_text.get())
            timestamp_ocr.write_sidecar_doc(self.scan.folder, doc)
        except Exception as exc:
            log.warning("Could not write timestamps for %s", self.scan.folder, exc_info=True)
            messagebox.showerror("Could not write times", str(exc), parent=self)
            return

        self._app.status.set(
            f"Point {self.point}: wrote {len(frames)} timestamp(s). "
            f"Click Process again to use the corrected times.")
        messagebox.showinfo(
            "Times written",
            f"Wrote {len(frames)} timestamp(s) for point {self.point}.\n\n"
            f"Click Process again to use the corrected times.", parent=self)
        self.destroy()

    def _full_ocr(self):
        """Fall back to reading every frame, for footage the line can't describe.

        Kept because the linear model genuinely fails on variable-frame-rate
        video, recordings with dropped frames, or a clock adjusted mid-record
        -- the cases the "worst reading" warning flags.
        """
        if not messagebox.askyesno(
                "Read every frame?",
                f"This reads the clock on all {len(self.scan.keys)} frames and can take "
                f"many minutes. Use it only if the fitted clock above looks wrong.\n\n"
                f"Continue?", parent=self):
            return

        folder = self.scan.folder
        names = [k.name for k in self.scan.keys]
        device = self._app.device.get()
        q: queue.Queue = queue.Queue()
        self._apply_btn.config(state="disabled")

        def worker():
            try:
                def progress(done, total, msg):
                    q.put(("status", f"[{done}/{total}] {msg}"))

                results = timestamp_ocr.ocr_folder(folder, names, device=device,
                                                   progress=progress)
                q.put(("done", len(results)))
            except Exception as exc:
                log.warning("Full-frame OCR failed for %s", folder, exc_info=True)
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "status":
                        self._app.status.set(payload)
                    elif kind == "error":
                        self._apply_btn.config(state="normal")
                        messagebox.showerror("OCR failed", payload, parent=self)
                        return
                    elif kind == "done":
                        self._app.status.set(
                            f"Point {self.point}: read {payload}/{len(names)} timestamp(s). "
                            f"Click Process again to use the corrected times.")
                        messagebox.showinfo(
                            "OCR complete",
                            f"Read {payload}/{len(names)} timestamp(s).", parent=self)
                        self.destroy()
                        return
            except queue.Empty:
                pass
            self.after(100, poll)

        self.after(100, poll)

    def _mark_region(self):
        """Let the user draw a box around the clock, then re-scan with it.

        For when automatic detection locks onto the wrong text -- a camera
        label, a license plate -- instead of the actual clock. That failure
        shows up in the samples list as a confident-looking but unparseable
        reading that's identical across every frame (the same watermark, read
        again and again). A hand-picked region also skips the region-probe
        pass on the next scan entirely, so it's faster as well as reliable.
        """
        if not self._samples:
            messagebox.showinfo("No frame available",
                                "There are no sampled frames to pick from.", parent=self)
            return
        image_path = os.path.join(self.scan.folder, self._samples[0].name)
        if not os.path.isfile(image_path):
            messagebox.showerror("Frame not found", f"Could not find {image_path}", parent=self)
            return

        def on_done(region):
            if region is not None:
                self._rescan_with_region(region)

        RegionPickerDialog(self, image_path, on_done)

    def _rescan_with_region(self, region):
        """Re-run the sampled scan with a fixed region, replacing this review.

        Reuses the exact filenames the current review already sampled from
        (recovered from ``scan.keys``/``scan.unparsed``) so marking a region
        doesn't quietly resample a different set of frames.
        """
        all_names = [k.name for k in self.scan.keys] + list(self.scan.unparsed)
        device = self._app.device.get()
        pattern_text = timestamp_ocr.strip_pattern_label(self.pattern_text.get())
        try:
            custom_pattern = (timestamp_ocr.compile_custom_pattern(pattern_text)
                              if pattern_text else None)
        except ValueError as exc:
            messagebox.showerror("Bad time pattern", str(exc), parent=self)
            return
        sample_count = max(len(self._samples), 1)

        q: queue.Queue = queue.Queue()
        self._apply_btn.config(state="disabled")
        self._mark_region_btn.config(state="disabled")
        self._resample_btn.config(state="disabled")
        verb = "the marked region" if region is not None else "a fresh auto-detected region"
        self._app.status.set(f"Re-reading clock for point {self.point} with {verb}...")

        def worker():
            try:
                def progress(done, total, msg):
                    q.put(("status", f"[{done}/{total}] {msg}"))

                result = timestamp_ocr.scan_folder(
                    self.scan.folder, all_names, device=device, sample_count=sample_count,
                    progress=progress, custom_pattern=custom_pattern, region_override=region)
                q.put(("done", result))
            except Exception as exc:
                log.warning("Re-scan with marked region failed for %s",
                           self.scan.folder, exc_info=True)
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "status":
                        self._app.status.set(payload)
                    elif kind == "error":
                        self._apply_btn.config(state="normal")
                        self._mark_region_btn.config(state="normal")
                        self._resample_btn.config(state="normal")
                        messagebox.showerror("Could not re-read timestamps", payload, parent=self)
                        return
                    elif kind == "done":
                        self.scan = payload
                        self._samples = list(payload.samples)
                        self._apply_btn.config(state="normal")
                        self._mark_region_btn.config(state="normal")
                        self._resample_btn.config(state="normal")
                        self._app.status.set(f"Point {self.point}: re-read the clock.")
                        self._refit()
                        return
            except queue.Empty:
                pass
            self.after(100, poll)

        self.after(100, poll)


def main():
    log_path = logging_setup.setup_logging()
    root = tk.Tk()
    root.title("Match-Vehicle-AI  |  Cross-Point Vehicle Re-ID")
    root.geometry("1100x760")
    # Below this, the control rows (folder pickers, device/model status) no
    # longer have room to lay out without their text running past the
    # window's edge -- resizing smaller should shrink the galleries, not
    # silently clip a status line with no way to see the rest of it.
    root.minsize(900, 600)
    app = ReIDApp(root)
    app.status.set(f"Ready. Logging to {log_path}")

    def on_close():
        app._save_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
