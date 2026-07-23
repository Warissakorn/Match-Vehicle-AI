"""Manage detection-model weights: list / download / update / remove.

Examples:
    python models_cli.py list                 # what's available + installed
    python models_cli.py download yolo11n.pt  # fetch a model
    python models_cli.py update yolo11n.pt    # re-fetch the latest weights
    python models_cli.py remove yolov8x.pt    # delete local weights
    python models_cli.py where                # print the models folder

Weights live in ``<project>/models`` by default (override with $MASH_MODELS_DIR
or --models-dir). Downloads need network access to the ultralytics asset host.
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

from mash_reid import logging_setup, model_manager, model_registry  # noqa: E402


def _cmd_list(args) -> int:
    models_dir = args.models_dir or model_manager.default_models_dir()
    ver = model_manager.ultralytics_version()
    print(f"Models folder : {models_dir}")
    print(f"ultralytics   : {ver or 'not installed'}  "
          f"(update the package, then `update <model>`, for newer weights)")
    print(f"{'':2}{'KEY':16}{'FAMILY':9}{'SIZE':8}{'~MB':>6}  STATUS   DESCRIPTION")
    for row in model_manager.status(models_dir):
        mark = "*" if row["recommended"] else " "
        if row["installed"]:
            state = f"{row['size_mb']:.0f}MB"
        else:
            state = "-"
        print(f"{mark} {row['key']:16}{row['family']:9}{row['size']:8}"
              f"{row['approx_mb']:6.0f}  {state:8} {row['description']}")
    print("\n* = newer generation, recommended.  ~MB = download size.")
    return 0


def _progress(message: str) -> None:
    print(f"  {message}")


def _cmd_download(args) -> int:
    path = model_manager.download(args.model, args.models_dir, progress=_progress)
    print(f"Ready: {path}")
    return 0


def _cmd_update(args) -> int:
    path = model_manager.update(args.model, args.models_dir, progress=_progress)
    print(f"Updated: {path}")
    return 0


def _cmd_remove(args) -> int:
    if model_manager.remove(args.model, args.models_dir):
        print(f"Removed {args.model}.")
    else:
        print(f"{args.model} was not installed.")
    return 0


def _cmd_where(args) -> int:
    print(args.models_dir or model_manager.default_models_dir())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage vehicle-detection models.")
    parser.add_argument("--models-dir", default=None,
                        help="Folder for weights (default: <project>/models or $MASH_MODELS_DIR)")
    parser.add_argument("--log-dir", default=logging_setup.DEFAULT_LOG_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available and installed models")
    sub.add_parser("where", help="Print the models folder path")

    choices = model_registry.keys()
    for name, help_text in (("download", "Download a model if missing"),
                            ("update", "Re-download a model's latest weights"),
                            ("remove", "Delete a model's local weights")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("model", choices=choices, metavar="MODEL",
                       help="Model key, e.g. yolo11n.pt")

    args = parser.parse_args(argv)
    logging_setup.setup_logging(args.log_dir, console_level=logging.WARNING)

    return {
        "list": _cmd_list,
        "where": _cmd_where,
        "download": _cmd_download,
        "update": _cmd_update,
        "remove": _cmd_remove,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
