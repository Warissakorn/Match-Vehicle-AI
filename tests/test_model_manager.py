"""Tests for model_manager: paths, install status, download/update/resolve.

Downloads are faked by monkeypatching ``_fetch_asset`` so nothing hits the
network — the tests exercise the filesystem logic around it.
"""

import os

import pytest

from mash_reid import model_manager, model_registry

KEY = model_registry.DEFAULT_KEY


def _fake_fetch(path_writes: dict):
    """Return a _fetch_asset stand-in that writes a dummy weights file."""
    def _fetch(filename, target):
        with open(target, "wb") as fh:
            fh.write(b"fake-weights")
        path_writes["target"] = target
    return _fetch


def test_default_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MASH_MODELS_DIR", str(tmp_path))
    assert model_manager.default_models_dir() == str(tmp_path)
    monkeypatch.delenv("MASH_MODELS_DIR")
    # Falls back to <project>/models
    assert model_manager.default_models_dir().endswith(os.path.join("", "models"))


def test_local_path_and_is_installed(tmp_path):
    path = model_manager.local_path(KEY, str(tmp_path))
    assert path == os.path.join(str(tmp_path), KEY)
    assert not model_manager.is_installed(KEY, str(tmp_path))
    open(path, "wb").close()
    assert model_manager.is_installed(KEY, str(tmp_path))


def test_status_marks_installed(tmp_path):
    open(os.path.join(str(tmp_path), KEY), "wb").close()
    rows = {r["key"]: r for r in model_manager.status(str(tmp_path))}
    assert rows[KEY]["installed"] is True
    assert rows[KEY]["path"] is not None
    # Some other model should be reported as not installed.
    other = next(k for k in rows if k != KEY)
    assert rows[other]["installed"] is False
    assert rows[other]["path"] is None


def test_download_missing_then_noop(monkeypatch, tmp_path):
    writes: dict = {}
    monkeypatch.setattr(model_manager, "_fetch_asset", _fake_fetch(writes))
    messages: list = []

    path = model_manager.download(KEY, str(tmp_path), progress=messages.append)
    assert os.path.exists(path)
    assert writes["target"] == path
    assert any("Downloading" in m for m in messages)

    # Second call must not re-fetch.
    writes.clear()
    messages.clear()
    path2 = model_manager.download(KEY, str(tmp_path), progress=messages.append)
    assert path2 == path
    assert writes == {}  # _fetch_asset not invoked
    assert any("already downloaded" in m for m in messages)


def test_download_rejects_unknown_key(tmp_path):
    with pytest.raises(KeyError):
        model_manager.download("nope.pt", str(tmp_path))


def test_update_refetches(monkeypatch, tmp_path):
    writes: dict = {}
    monkeypatch.setattr(model_manager, "_fetch_asset", _fake_fetch(writes))
    first = model_manager.download(KEY, str(tmp_path))
    os.utime(first, (0, 0))
    writes.clear()

    updated = model_manager.update(KEY, str(tmp_path))
    assert updated == first
    assert writes["target"] == first  # re-fetched


def test_remove(tmp_path):
    path = model_manager.local_path(KEY, str(tmp_path))
    open(path, "wb").close()
    assert model_manager.remove(KEY, str(tmp_path)) is True
    assert not os.path.exists(path)
    assert model_manager.remove(KEY, str(tmp_path)) is False


def test_resolve_weights_custom_path_passthrough(tmp_path):
    custom = "/some/custom/weights.pt"
    assert model_manager.resolve_weights(custom, str(tmp_path)) == custom


def test_resolve_weights_known_downloads_when_missing(monkeypatch, tmp_path):
    writes: dict = {}
    monkeypatch.setattr(model_manager, "_fetch_asset", _fake_fetch(writes))
    resolved = model_manager.resolve_weights(KEY, str(tmp_path))
    assert resolved == os.path.join(str(tmp_path), KEY)
    assert os.path.exists(resolved)


def test_resolve_weights_known_uses_existing_without_fetch(monkeypatch, tmp_path):
    def _boom(filename, target):
        raise AssertionError("_fetch_asset must not be called when file exists")
    monkeypatch.setattr(model_manager, "_fetch_asset", _boom)
    path = model_manager.local_path(KEY, str(tmp_path))
    open(path, "wb").close()
    assert model_manager.resolve_weights(KEY, str(tmp_path)) == path
