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
    training_export,
    video_extractor,
)
from mash_reid.matcher import VehicleRecord  # noqa: E402

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
                on_confirm=None, on_reject=None):
        """Add a thumbnail card. If ``on_confirm``/``on_reject`` (no-arg
        callables) are given, a "Same"/"Different" button pair is shown below
        the caption for labeling this candidate as training data; both
        disable themselves after either is clicked.
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

        self.dir_a = tk.StringVar()
        self.dir_b = tk.StringVar()
        # Detection model: keep a display<->key mapping so the combobox can show
        # friendly names while we pass the weights key to the pipeline.
        self._model_display_to_key = {m.display_name: m.key for m in model_registry.all_models()}
        self.model_key = tk.StringVar(value=config.YOLO_WEIGHTS)
        self.model_display = tk.StringVar(value=model_registry.default().display_name)
        self.threshold = tk.DoubleVar(value=config.DEFAULT_SIMILARITY_THRESHOLD)
        self.det_conf = tk.DoubleVar(value=config.DEFAULT_DETECTION_CONF)
        self.min_travel = tk.DoubleVar(value=0.0)
        self.max_travel = tk.DoubleVar(value=600.0)
        self.use_gate = tk.BooleanVar(value=True)
        self.one_to_one = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Select folders for point A and B, then Process.")

        self._res_a = None
        self._res_b = None
        self._a_by_id: dict[int, VehicleRecord] = {}
        self._b_by_id: dict[int, VehicleRecord] = {}
        self._a_clusters: dict[int, int] = {}  # record_id -> cluster id (same-point repeats)
        self._b_clusters: dict[int, int] = {}
        self._last_selected_a: VehicleRecord | None = None
        self._queue: queue.Queue = queue.Queue()

        self._build_controls()
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
        self._process_btn = ttk.Button(toggles, text="Process", command=self._on_process)
        self._process_btn.pack(side="right", padx=4)

        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", pady=(4, 6))

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
        thread = threading.Thread(target=self._process_worker, args=(dir_a, dir_b), daemon=True)
        thread.start()
        self.after(100, self._poll_queue)

    def _process_worker(self, dir_a, dir_b):
        try:
            pcfg = config.PipelineConfig(
                yolo_weights=self.model_key.get(), detection_conf=self.det_conf.get())
            detector, embedder = pipeline.build_pipeline(pcfg)

            def progress(done, total, msg):
                self._queue.put(("status", f"[{done}/{total}] {msg}"))

            self._queue.put(("status", "Processing point A..."))
            res_a = pipeline.process_point(dir_a, "A", detector, embedder, pcfg, progress=progress)
            self._queue.put(("status", "Processing point B..."))
            res_b = pipeline.process_point(dir_b, "B", detector, embedder, pcfg, progress=progress)
            self._queue.put(("done", (res_a, res_b)))
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

    def _on_processed(self, res_a, res_b):
        self._res_a, self._res_b = res_a, res_b
        self._a_by_id = {r.record_id: r for r in res_a.records}
        self._b_by_id = {r.record_id: r for r in res_b.records}
        # Group repeat sightings of the same vehicle within each point (e.g. a
        # car circling back past the same camera) so they read as one vehicle
        # rather than several, while every detection still stays visible.
        self._a_clusters = matcher.cluster_same_point(res_a.records)
        self._b_clusters = matcher.cluster_same_point(res_b.records)
        self._process_btn.config(state="normal")
        self.status.set(
            f"A: {len(res_a.records)} vehicles / {res_a.frame_count} frames   |   "
            f"B: {len(res_b.records)} vehicles / {res_b.frame_count} frames"
        )
        self._populate_a()
        self._last_selected_a = None
        self._populate_b_all()

    def _cluster_tag(self, clusters: dict[int, int], record_id: int) -> str:
        """Return a " •GrpN(xK)" suffix when this record shares a same-point
        cluster with other detections; empty string for a singleton.
        """
        cluster_id = clusters.get(record_id)
        if cluster_id is None:
            return ""
        size = sum(1 for c in clusters.values() if c == cluster_id)
        return f"  •Grp{cluster_id}(x{size})" if size > 1 else ""

    def _populate_a(self):
        self.gallery_a.clear()
        if not self._res_a:
            return
        for rec in self._res_a.records:
            cap = f"A#{rec.record_id}  {rec.timestamp:%H:%M:%S}"
            cap += self._cluster_tag(self._a_clusters, rec.record_id)
            self.gallery_a.add_card(rec, cap, self._on_select_a, self._on_double)

    def _show_all_b(self):
        self._last_selected_a = None
        self._populate_b_all()

    def _populate_b_all(self):
        """Default B view: every detected vehicle at point B, grouped by
        same-point cluster tag. Shown before any point-A vehicle is selected,
        and reachable again afterwards via the "Show all B" button.
        """
        self.gallery_b.clear()
        self.b_view_label.set("Showing: all point B vehicles")
        if not self._res_b:
            return
        for rec in self._res_b.records:
            cap = f"B#{rec.record_id}  {rec.timestamp:%H:%M:%S}"
            cap += self._cluster_tag(self._b_clusters, rec.record_id)
            self.gallery_b.add_card(rec, cap, lambda r: None, self._on_double)

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
            cap += self._cluster_tag(self._b_clusters, rec_b.record_id)
            self.gallery_b.add_card(
                rec_b, cap, lambda r: None, self._on_double,
                on_confirm=lambda a=rec_a, b=rec_b, s=cand.similarity: self._label_pair(a, b, s, True),
                on_reject=lambda a=rec_a, b=rec_b, s=cand.similarity: self._label_pair(a, b, s, False),
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
    """Modal dialog: pick a video, extract timestamped frames for one point.

    On success the output folder is written back into the point's folder entry
    so the user can immediately Process it.
    """

    def __init__(self, parent, point, folder_var):
        super().__init__(parent)
        self.title(f"Extract frames for point {point}")
        self.point = point
        self.folder_var = folder_var
        self._queue: queue.Queue = queue.Queue()

        self.video_path = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.interval = tk.DoubleVar(value=1.0)
        self.start_time = tk.StringVar()
        self.status = tk.StringVar(value="Choose a video to begin.")

        self._build()
        self.transient(parent)
        self.grab_set()

    def _build(self):
        pad = {"padx": 8, "pady": 4}

        vrow = ttk.Frame(self)
        vrow.pack(fill="x", **pad)
        ttk.Label(vrow, text="Video", width=10).pack(side="left")
        ttk.Entry(vrow, textvariable=self.video_path, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(vrow, text="Browse...", command=self._pick_video).pack(side="left", padx=4)

        orow = ttk.Frame(self)
        orow.pack(fill="x", **pad)
        ttk.Label(orow, text="Output", width=10).pack(side="left")
        ttk.Entry(orow, textvariable=self.out_dir, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(orow, text="Browse...", command=self._pick_out).pack(side="left", padx=4)

        srow = ttk.Frame(self)
        srow.pack(fill="x", **pad)
        ttk.Label(srow, text="Start time", width=10).pack(side="left")
        ttk.Entry(srow, textvariable=self.start_time, width=24).pack(side="left")
        ttk.Label(srow, text="(auto from filename; edit as YYYY-MM-DD HH:MM:SS)").pack(side="left", padx=6)

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
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path.set(path)
        # Prefill output folder and the start time parsed from the filename.
        self.out_dir.set(video_extractor.default_output_dir(path, self.point))
        start, source = video_extractor.resolve_start_time(path, None)
        self.start_time.set(start.strftime("%Y-%m-%d %H:%M:%S"))
        self.status.set(f"Start time from {source}. Adjust if needed, then Extract.")

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
        video = self.video_path.get().strip()
        out = self.out_dir.get().strip()
        if not os.path.isfile(video):
            messagebox.showerror("No video", "Please choose a valid video file.", parent=self)
            return
        if not out:
            messagebox.showerror("No output", "Please choose an output folder.", parent=self)
            return
        try:
            start_dt = self._parse_start()
        except ValueError as exc:
            messagebox.showerror("Invalid start time", str(exc), parent=self)
            return

        self._extract_btn.config(state="disabled")
        self.status.set("Extracting frames...")
        thread = threading.Thread(
            target=self._worker, args=(video, out, start_dt, self.interval.get()), daemon=True
        )
        thread.start()
        self.after(100, self._poll)

    def _worker(self, video, out, start_dt, interval):
        try:
            def progress(done, total, msg):
                total_str = str(total) if total else "?"
                self._queue.put(("status", f"[{done}/{total_str}] {msg}"))

            written = video_extractor.extract_frames(
                video, out, self.point, interval_seconds=interval,
                start_time=start_dt, progress=progress,
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


def main():
    log_path = logging_setup.setup_logging()
    root = tk.Tk()
    root.title("Match-Vehicle-AI  |  Cross-Point Vehicle Re-ID")
    root.geometry("1100x760")
    app = ReIDApp(root)
    app.status.set(f"Ready. Logging to {log_path}")
    root.mainloop()


if __name__ == "__main__":
    main()
