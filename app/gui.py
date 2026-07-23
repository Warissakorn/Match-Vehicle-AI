"""Simple Tkinter desktop app for cross-point vehicle Re-ID.

Workflow:
    1. Pick the folder of frames for point A and for point B.
    2. Adjust thresholds / travel-time window if needed.
    3. Click "Process" -- detection + embedding run in a background thread.
    4. Click a vehicle in the A gallery -> its best B-matches appear on the
       right with similarity scores. Double-click any thumbnail to see the full
       frame with the bounding box drawn.

Run with:  python app/gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Make ``config`` (project root) and the ``mash_reid`` package importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import config  # noqa: E402
from mash_reid import matcher, pipeline, video_extractor  # noqa: E402
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

    def add_card(self, record: VehicleRecord, caption: str, on_click, on_double):
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


class ReIDApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)

        self.dir_a = tk.StringVar()
        self.dir_b = tk.StringVar()
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
        right = ttk.Labelframe(panes, text="Best matches at point B")
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        self.gallery_a = ScrollableThumbs(left, columns=2)
        self.gallery_a.pack(fill="both", expand=True)
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
            pcfg = config.PipelineConfig(detection_conf=self.det_conf.get())
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
        self._process_btn.config(state="normal")
        self.status.set(
            f"A: {len(res_a.records)} vehicles / {res_a.frame_count} frames   |   "
            f"B: {len(res_b.records)} vehicles / {res_b.frame_count} frames"
        )
        self._populate_a()
        self.gallery_b.clear()

    def _populate_a(self):
        self.gallery_a.clear()
        if not self._res_a:
            return
        for rec in self._res_a.records:
            cap = f"A#{rec.record_id}  {rec.timestamp:%H:%M:%S}"
            self.gallery_a.add_card(rec, cap, self._on_select_a, self._on_double)

    def _on_select_a(self, rec_a: VehicleRecord):
        if not self._res_b:
            return
        results = matcher.match([rec_a], self._res_b.records, self._current_match_config())
        self.gallery_b.clear()
        result = results[0] if results else None
        if not result or not result.candidates:
            self.status.set(f"A#{rec_a.record_id}: no B match above threshold.")
            return
        self.status.set(f"A#{rec_a.record_id}: {len(result.candidates)} candidate(s).")
        for cand in result.candidates:
            rec_b = self._b_by_id[cand.b_record_id]
            dt = (rec_b.timestamp - rec_a.timestamp).total_seconds()
            cap = f"B#{rec_b.record_id}  sim={cand.similarity:.2f}  +{dt:.0f}s"
            self.gallery_b.add_card(rec_b, cap, lambda r: None, self._on_double)
        self._last_selected_a = rec_a

    def _on_double(self, rec: VehicleRecord):
        _show_full_frame(self, rec)

    def _rematch(self):
        # Re-run matching for the currently selected A vehicle when a slider or
        # toggle changes. No model work involved -- purely numeric, so instant.
        rec_a = getattr(self, "_last_selected_a", None)
        if rec_a is not None and self._res_b is not None:
            self._on_select_a(rec_a)


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


def main():
    root = tk.Tk()
    root.title("Mash-Object-AI  |  Cross-Point Vehicle Re-ID")
    root.geometry("1100x760")
    ReIDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
