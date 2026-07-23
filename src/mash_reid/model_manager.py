"""Check, download and update detection-model weights.

The registry (``model_registry``) says *which* models exist; this module deals
with the *files*: where they live on disk, whether they're already downloaded,
and how to fetch or refresh them. Downloading uses Ultralytics' own asset
downloader, so the URLs/versions track whatever ``ultralytics`` version is
installed — that is how "keep models up to date" stays honest: update the
package and re-download to get the latest published weights.

Nothing here imports torch or ultralytics at module load — those are pulled in
only inside the download functions — so ``status()`` and path helpers stay fast
and work with no network. Weights land in one folder (default ``<project>/models``,
override with ``MASH_MODELS_DIR``) instead of the current directory, so runs
share one cache regardless of where they're launched from.
"""

from __future__ import annotations

import logging
import os
import shutil

from mash_reid import model_registry

log = logging.getLogger(__name__)

_ENV_MODELS_DIR = "MASH_MODELS_DIR"


def default_models_dir() -> str:
    """Folder where weights are stored. ``$MASH_MODELS_DIR`` overrides the default."""
    env = os.environ.get(_ENV_MODELS_DIR)
    if env:
        return env
    # src/mash_reid/model_manager.py -> mash_reid -> src -> project root
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "models")


def _key_of(model) -> str:
    return model.key if isinstance(model, model_registry.ModelInfo) else str(model)


def local_path(model, models_dir: str | None = None) -> str:
    """Absolute path where ``model`` (key or ``ModelInfo``) would be stored."""
    models_dir = models_dir or default_models_dir()
    return os.path.join(models_dir, _key_of(model))


def is_installed(model, models_dir: str | None = None) -> bool:
    """True if the model's weights file already exists locally."""
    return os.path.exists(local_path(model, models_dir))


def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def ultralytics_version() -> str | None:
    """Installed ultralytics version (drives which weights 'latest' means), or None."""
    try:
        import ultralytics
    except Exception:
        return None
    return getattr(ultralytics, "__version__", "unknown")


def status(models_dir: str | None = None) -> list[dict]:
    """Per-model install status for the whole catalog (no network)."""
    models_dir = models_dir or default_models_dir()
    rows: list[dict] = []
    for m in model_registry.all_models():
        path = os.path.join(models_dir, m.key)
        installed = os.path.exists(path)
        rows.append(
            {
                "key": m.key,
                "display_name": m.display_name,
                "family": m.family,
                "size": m.size,
                "approx_mb": m.approx_mb,
                "recommended": m.recommended,
                "description": m.description,
                "installed": installed,
                "path": path if installed else None,
                "size_mb": round(file_size_mb(path), 1) if installed else None,
            }
        )
    return rows


def _emit(progress, message: str) -> None:
    log.info(message)
    if progress:
        progress(message)


def _fetch_asset(filename: str, target: str) -> None:
    """Download the named ultralytics asset to ``target`` (overridable in tests)."""
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception as exc:  # ultralytics missing / API moved
        raise RuntimeError(
            "ultralytics is required to download model weights. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    got = attempt_download_asset(target)
    # Some ultralytics versions drop the file in CWD or their own weights dir
    # rather than at ``target``; relocate it so our models dir stays the cache.
    if not os.path.exists(target):
        for cand in (got, filename, os.path.join(os.getcwd(), filename)):
            if (
                cand
                and os.path.exists(cand)
                and os.path.abspath(cand) != os.path.abspath(target)
            ):
                shutil.move(cand, target)
                break


def download(key: str, models_dir: str | None = None, progress=None) -> str:
    """Ensure ``key`` is downloaded; return its local path. No-op if present.

    ``progress`` is an optional ``callable(message: str)`` for UI status lines.
    """
    info = model_registry.get(key)  # validates the key
    models_dir = models_dir or default_models_dir()
    os.makedirs(models_dir, exist_ok=True)
    target = os.path.join(models_dir, info.key)

    if os.path.exists(target):
        _emit(progress, f"{info.key} already downloaded ({file_size_mb(target):.1f} MB)")
        return target

    _emit(progress, f"Downloading {info.key} (~{info.approx_mb:.0f} MB)...")
    try:
        _fetch_asset(info.key, target)
    except Exception:
        log.exception("Failed to download model '%s'", info.key)
        raise
    if not os.path.exists(target):
        raise RuntimeError(
            f"Download of '{info.key}' finished but {target} is missing."
        )
    _emit(progress, f"Downloaded {info.key} ({file_size_mb(target):.1f} MB)")
    return target


def update(key: str, models_dir: str | None = None, progress=None) -> str:
    """Refresh a model to the latest published weights (removes then re-downloads).

    "Latest" is whatever the installed ultralytics version points at, so pair
    this with upgrading the ``ultralytics`` package to get newer weights.
    """
    models_dir = models_dir or default_models_dir()
    target = local_path(key, models_dir)
    if os.path.exists(target):
        os.remove(target)
        _emit(progress, f"Removed cached {key}, fetching latest...")
    return download(key, models_dir, progress)


def remove(key: str, models_dir: str | None = None) -> bool:
    """Delete a model's local weights. Returns True if a file was removed."""
    target = local_path(key, models_dir)
    if os.path.exists(target):
        os.remove(target)
        log.info("Removed model weights %s", target)
        return True
    return False


def resolve_weights(weights: str, models_dir: str | None = None, progress=None) -> str:
    """Turn a selected model into a path to hand to ``YOLO(...)``.

    Known catalog keys are stored in (and downloaded to, if missing) the shared
    models dir. Anything else — a custom ``.pt`` path or an ultralytics name we
    don't catalog — is returned unchanged so YOLO handles it as before.
    """
    if not model_registry.is_known(weights):
        return weights
    models_dir = models_dir or default_models_dir()
    target = os.path.join(models_dir, weights)
    if os.path.exists(target):
        return target
    return download(weights, models_dir, progress)
