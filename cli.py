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
from mash_reid import logging_setup, matcher, model_registry, pipeline  # noqa: E402


def _fmt_ts(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-point vehicle Re-ID (A vs B).")
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
    parser.add_argument("--log-dir", default=logging_setup.DEFAULT_LOG_DIR,
                        help="Folder for run log files (default: logs/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print DEBUG detail to the console")
    args = parser.parse_args(argv)

    log_path = logging_setup.setup_logging(
        args.log_dir, console_level=logging.DEBUG if args.verbose else logging.INFO)
    print(f"Logging to {log_path}")

    pcfg = config.PipelineConfig(
        yolo_weights=args.model, detection_conf=args.conf, models_dir=args.models_dir)
    print(f"Detection model: {args.model}")
    detector, embedder = pipeline.build_pipeline(pcfg)

    def show_progress(done, total, msg):
        print(f"  [{done}/{total}] {msg}", end="\r", flush=True)

    print("Processing point A ...")
    res_a = pipeline.process_point(args.dir_a, "A", detector, embedder, pcfg,
                                   use_cache=not args.no_cache, progress=show_progress)
    a_clusters = matcher.cluster_same_point(res_a.records)
    print(f"\n  {len(res_a.records)} vehicles across {res_a.frame_count} frames "
          f"({len(set(a_clusters.values()))} distinct, after grouping repeat sightings)")

    print("Processing point B ...")
    res_b = pipeline.process_point(args.dir_b, "B", detector, embedder, pcfg,
                                   use_cache=not args.no_cache, progress=show_progress)
    b_clusters = matcher.cluster_same_point(res_b.records)
    print(f"\n  {len(res_b.records)} vehicles across {res_b.frame_count} frames "
          f"({len(set(b_clusters.values()))} distinct, after grouping repeat sightings)")

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
