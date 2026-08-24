"""Command-line smoke test / batch runner for the vehicle Re-ID pipeline.

Example:
    python cli.py --dir-a samples/pointA --dir-b samples/pointB \\
        --threshold 0.6 --max-travel 600

Runs detection + embedding on both folders, matches A->B, and prints the best
B-candidate for each A-vehicle. Handy for verifying the models download and run
before opening the GUI.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make ``config`` (project root) and the ``mash_reid`` package importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import config  # noqa: E402
from mash_reid import (  # noqa: E402
    app_paths,
    logging_setup,
    matcher,
    model_registry,
    pipeline,
)
import mash_reid.version as app_version  # noqa: E402


def _fmt_ts(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _frame_names(folder: str) -> list[str]:
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS
    )


def _ocr_folder(folder: str, point: str, device, show_progress) -> None:
    """Read every frame's clock and write the timestamp sidecar (the slow path)."""
    from mash_reid import timestamp_ocr

    names = _frame_names(folder)
    print(f"OCR-ing timestamps for point {point} ({len(names)} frame(s)) ...")
    results = timestamp_ocr.ocr_folder(folder, names, device=device, progress=show_progress)
    print(f"\n  Read {len(results)}/{len(names)} timestamp(s) from the frame overlay.")


def _fit_folder(folder: str, point: str, device, show_progress,
                sample_count: int, force: bool) -> bool:
    """Read a sample of clocks, fit a timeline, and write it. Returns success.

    The headless counterpart of the GUI's review dialog. Since nobody is
    there to eyeball a suspicious fit, this one *does* refuse to write when
    the sanity checks fail -- ``--force`` overrides, and the warnings are
    printed either way.
    """
    from mash_reid import timeline, timestamp_ocr

    names = _frame_names(folder)
    print(f"Reading clock samples for point {point} "
          f"({len(names)} frame(s), sampling ~{sample_count}) ...")
    scan = timestamp_ocr.scan_folder(folder, names, device=device,
                                     sample_count=sample_count, progress=show_progress)
    fit = scan.fit
    read = sum(1 for s in scan.samples if s.chosen)

    rate = f"{fit.implied_fps:.2f} fps" if fit.implied_fps else f"{fit.slope:.4f} s/frame"
    print(f"\n  Read {read}/{len(scan.samples)} sampled clock(s); {rate}, "
          f"{fit.n_inliers}/{fit.n_samples} agree, "
          f"typical error {fit.residual_mad:.2f} s, worst {fit.max_abs_residual:.2f} s")
    for warning in fit.warnings:
        print(f"  ! {warning}")

    if not fit.ok and not force:
        print("  Refusing to write these timestamps (use --force to override, or "
              "--ocr-every-frame to read every frame instead).")
        return False

    frames = timeline.apply_fit(fit, scan.keys)
    doc = timeline.build_document(fit, scan.samples, frames)
    # Carried alongside the fit so the GUI can later reopen this exact scan
    # for review without any new OCR -- see timestamp_ocr.load_scan_for_review.
    doc["region"] = list(scan.region) if scan.region else None
    doc["region_is_manual"] = scan.region_is_manual
    timestamp_ocr.write_sidecar_doc(folder, doc)
    print(f"  Wrote {len(frames)} timestamp(s).")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-point vehicle Re-ID (A vs B).")
    parser.add_argument("--version", action="version",
                        version=f"MatchVehicleAI {app_version.get_version()}")
    parser.add_argument("--dir-a", required=True, help="Folder of frames from point A")
    parser.add_argument("--dir-b", required=True, help="Folder of frames from point B")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--model", default=config.YOLO_WEIGHTS, metavar="KEY_OR_PATH",
                        help="Detection model: a catalog key (%s) or a custom .pt path. "
                             "See `python models_cli.py list`."
                             % "/".join(model_registry.keys()))
    parser.add_argument("--models-dir", default=None,
                        help="Folder for downloaded weights (default: <project>/models "
                             "or $MASH_MODELS_DIR)")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_DETECTION_CONF,
                        help="YOLO detection confidence")
    parser.add_argument("--device", default=None,
                        help="Compute device: 'auto' (default, uses CUDA if available), "
                             "'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--min-travel", type=float, default=0.0,
                        help="Min seconds between passing A and B")
    parser.add_argument("--max-travel", type=float, default=600.0,
                        help="Max seconds between passing A and B")
    parser.add_argument("--no-time-gate", action="store_true",
                        help="Ignore timestamps when matching")
    parser.add_argument("--one-to-one", action="store_true",
                        help="Force a one-to-one A/B assignment (Hungarian)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not read/write the on-disk detection cache")
    parser.add_argument("--cluster-same-point", action="store_true",
                        default=config.DEFAULT_ENABLE_SAME_POINT_CLUSTERING,
                        help="Group repeat sightings of one vehicle within each point. "
                             "Off by default: it is a full N-squared similarity pass per "
                             "point and only affects the 'distinct' count printed below")
    parser.add_argument("--ocr-time", action="store_true",
                        help="Read the true timestamps off the frames before processing. "
                             "Samples a few frames and fits the clock's rate (writes a "
                             "sidecar, reused on later runs)")
    parser.add_argument("--ocr-samples", type=int,
                        default=config.DEFAULT_TIMELINE_SAMPLE_TARGET,
                        help="How many frames to sample when fitting the clock "
                             "(default: %(default)s)")
    parser.add_argument("--ocr-every-frame", action="store_true",
                        help="With --ocr-time, read every frame's clock instead of "
                             "fitting from a sample. Far slower; use only when the "
                             "footage has gaps or a variable frame rate")
    parser.add_argument("--force", action="store_true",
                        help="Write fitted timestamps even when they fail the sanity checks")
    parser.add_argument("--log-dir", default=logging_setup.DEFAULT_LOG_DIR,
                        help="Folder for run log files (default: logs/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print DEBUG detail to the console")
    args = parser.parse_args(argv)

    app_paths.apply_runtime_env()
    log_dir = args.log_dir
    if log_dir == logging_setup.DEFAULT_LOG_DIR:
        # Only the default is relocated in a frozen build; an explicit
        # --log-dir means the caller chose it deliberately.
        log_dir = app_paths.logs_dir()
    log_path = logging_setup.setup_logging(
        log_dir, console_level=logging.DEBUG if args.verbose else logging.INFO)
    print(f"Logging to {log_path}")

    pcfg = config.PipelineConfig(
        yolo_weights=args.model, detection_conf=args.conf, models_dir=args.models_dir,
        device=args.device)
    print(f"Detection model: {args.model}")
    detector, embedder = pipeline.build_pipeline(pcfg)

    from mash_reid.device import describe_device, diagnose_cuda_unavailable, resolve_device
    resolved = resolve_device(args.device)
    print(f"Device: {describe_device(resolved)}")
    if resolved == "cpu":
        reason = diagnose_cuda_unavailable()
        if reason:
            print(f"  (CUDA not used: {reason})")

    def show_progress(done, total, msg):
        print(f"  [{done}/{total}] {msg}", end="\r", flush=True)

    if args.ocr_time:
        for folder, point in ((args.dir_a, "A"), (args.dir_b, "B")):
            if args.ocr_every_frame:
                _ocr_folder(folder, point, pcfg.device, show_progress)
            else:
                _fit_folder(folder, point, pcfg.device, show_progress,
                            args.ocr_samples, args.force)

    def summarize(res):
        """One line per point; the 'distinct' clause only when we clustered."""
        line = f"\n  {len(res.records)} vehicles across {res.frame_count} frames"
        if args.cluster_same_point:
            clusters = matcher.cluster_same_point(res.records)
            line += (f" ({len(set(clusters.values()))} distinct, "
                     f"after grouping repeat sightings)")
        print(line)

    print("Processing point A ...")
    res_a = pipeline.process_point(args.dir_a, "A", detector, embedder, pcfg,
                                   use_cache=not args.no_cache, progress=show_progress)
    summarize(res_a)

    print("Processing point B ...")
    res_b = pipeline.process_point(args.dir_b, "B", detector, embedder, pcfg,
                                   use_cache=not args.no_cache, progress=show_progress)
    summarize(res_b)

    mcfg = config.MatchConfig(
        similarity_threshold=args.threshold,
        top_k=args.top_k,
        use_time_gate=not args.no_time_gate,
        min_travel_seconds=args.min_travel,
        max_travel_seconds=args.max_travel,
        one_to_one=args.one_to_one,
    )
    results = matcher.match(res_a.records, res_b.records, mcfg)
    b_by_id = {r.record_id: r for r in res_b.records}
    a_by_id = {r.record_id: r for r in res_a.records}

    print("\n=== Matches (A -> best B) ===")
    matched = 0
    for result in results:
        rec_a = a_by_id[result.a_record_id]
        best = result.best
        if best is None:
            print(f"A#{rec_a.record_id} [{_fmt_ts(rec_a.timestamp)}] "
                  f"{os.path.basename(rec_a.frame_path)} -> no match")
            continue
        matched += 1
        rec_b = b_by_id[best.b_record_id]
        dt = (rec_b.timestamp - rec_a.timestamp).total_seconds()
        print(f"A#{rec_a.record_id} [{_fmt_ts(rec_a.timestamp)}] "
              f"{os.path.basename(rec_a.frame_path)} -> "
              f"B#{rec_b.record_id} [{_fmt_ts(rec_b.timestamp)}] "
              f"{os.path.basename(rec_b.frame_path)} "
              f"(sim={best.similarity:.3f}, travel={dt:.0f}s)")

    print(f"\n{matched}/{len(results)} A-vehicles matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
