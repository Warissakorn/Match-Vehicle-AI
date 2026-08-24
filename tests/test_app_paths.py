"""Tests for the user-data path resolution (``mash_reid.app_paths``).

The two modes must stay sharply separated: a source checkout keeps every
historical location byte-for-byte (tests elsewhere assert settings sits next
to ``config.py``), while a frozen build roots everything writable under one
data directory so an installed exe works from read-only locations like
Program Files.
"""

import os
import sys

import pytest

import config
from mash_reid import app_paths


@pytest.fixture
def project_root():
    """The root the unfrozen path arithmetic derives (src/mash_reid/ -> up 3)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(app_paths.__file__))))


@pytest.fixture
def unfrozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    for var in ("MASH_DATA_DIR", "MASH_MODELS_DIR", "TORCH_HOME", "YOLO_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def frozen_windows(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", os.path.join("C:", "Users", "someone", "AppData", "Local"))
    for var in ("MASH_DATA_DIR", "MASH_MODELS_DIR", "TORCH_HOME", "YOLO_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)


# --- unfrozen: historical behaviour, unchanged -------------------------------


def test_unfrozen_data_root_is_project_root(unfrozen, project_root):
    assert app_paths.data_root() == project_root


def test_unfrozen_settings_stays_next_to_config(unfrozen, project_root):
    expected = os.path.join(project_root, config.DEFAULT_SETTINGS_FILE)
    assert app_paths.settings_file() == expected


def test_unfrozen_models_dir_under_project(unfrozen, project_root):
    assert app_paths.models_dir() == os.path.join(project_root, "models")


def test_unfrozen_logs_dir_is_relative_logs(unfrozen):
    assert app_paths.logs_dir() == "logs"


def test_unfrozen_third_party_cache_locations_untouched(unfrozen):
    # None means "leave easyocr/torch on their own historical defaults".
    assert app_paths.easyocr_model_dir() is None
    assert app_paths.torch_home() is None


# --- frozen: everything writable under one data root -------------------------


def test_frozen_data_root_is_localappdata(frozen_windows):
    assert app_paths.data_root() == os.path.join(
        "C:", "Users", "someone", "AppData", "Local", "MatchVehicleAI")


def test_frozen_state_lives_under_data_root(frozen_windows):
    root = app_paths.data_root()
    assert app_paths.models_dir() == os.path.join(root, "models")
    assert app_paths.settings_file() == os.path.join(root, "settings.json")
    assert app_paths.ocr_patterns_file() == os.path.join(root, "ocr_patterns.json")
    assert app_paths.logs_dir() == os.path.join(root, "logs")
    assert app_paths.easyocr_model_dir() == os.path.join(root, "easyocr")


def test_frozen_apply_runtime_env_points_torch_at_data_root(frozen_windows):
    app_paths.apply_runtime_env()
    root = app_paths.data_root()
    assert os.environ["TORCH_HOME"] == os.path.join(root, "torch")
    assert os.environ["YOLO_CONFIG_DIR"] == os.path.join(root, "ultralytics")


# --- overrides ---------------------------------------------------------------


def test_mash_data_dir_overrides_everything(monkeypatch, tmp_path, unfrozen):
    monkeypatch.setenv("MASH_DATA_DIR", str(tmp_path))
    assert app_paths.data_root() == str(tmp_path)
    assert app_paths.settings_file() == os.path.join(str(tmp_path), "settings.json")
    assert app_paths.logs_dir() == os.path.join(str(tmp_path), "logs")


def test_explicit_torch_home_wins_over_app_choice(frozen_windows, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str("D:/custom/torch"))
    app_paths.apply_runtime_env()
    assert os.environ["TORCH_HOME"] == "D:/custom/torch"


def test_mash_models_dir_beats_the_data_root(monkeypatch, tmp_path, frozen_windows):
    monkeypatch.setenv("MASH_MODELS_DIR", str(tmp_path / "weights"))
    assert app_paths.models_dir() == str(tmp_path / "weights")


def test_is_frozen_reflects_sys_flag(unfrozen, monkeypatch):
    assert not app_paths.is_frozen()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert app_paths.is_frozen()
