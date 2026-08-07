"""paths.py picks a different home once compiled -- this guards the switch.

The bug this exists to prevent: a --onefile build's bundle is a temp folder,
recreated and discarded on every launch. If settings.json or the models/
folder resolved into that bundle, the app would forget every setting and
re-download every model weight on every single run, silently, with no error
to point at the cause.
"""

from __future__ import annotations

import os
import sys

from mash_reid import paths


def test_unfrozen_resource_and_writable_dirs_are_the_project_root():
    assert not paths.is_frozen()
    root = paths._project_root()
    assert paths.resource_dir() == root
    assert paths.writable_dir() == root
    assert os.path.isfile(os.path.join(root, "config.py"))


# Plain POSIX-style stand-ins for an install folder -- the tests exercise
# paths.py's *logic* (dirname-of-executable, MEIPASS precedence), which is
# the same on every platform; os.path itself handles the separator, and this
# sandbox has no Windows to run a real "C:\\App\\..." path through.
_EXE = os.path.join(os.sep, "app", "MatchVehicleAI.exe")
_APP_DIR = os.path.dirname(_EXE)


def test_frozen_writable_dir_is_next_to_the_executable(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    assert paths.is_frozen()
    assert paths.writable_dir() == _APP_DIR


def test_frozen_resource_dir_prefers_meipass_over_the_executable_folder(monkeypatch):
    meipass = os.path.join(os.sep, "tmp", "_MEI12345")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", meipass, raising=False)
    assert paths.resource_dir() == meipass


def test_frozen_resource_dir_falls_back_to_executable_folder_without_meipass(monkeypatch):
    """--onedir doesn't always set _MEIPASS on every PyInstaller version;
    the bundle then lives next to the exe itself."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert paths.resource_dir() == _APP_DIR


def test_models_dir_env_override_still_wins_when_frozen(monkeypatch):
    from mash_reid import model_manager
    override = os.path.join(os.sep, "data", "weights")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    monkeypatch.setenv("MASH_MODELS_DIR", override)
    assert model_manager.default_models_dir() == override


def test_models_dir_is_next_to_the_executable_when_frozen(monkeypatch):
    from mash_reid import model_manager
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    monkeypatch.delenv("MASH_MODELS_DIR", raising=False)
    assert model_manager.default_models_dir() == os.path.join(_APP_DIR, "models")


def test_settings_path_is_next_to_the_executable_when_frozen(monkeypatch):
    from mash_reid import settings
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", _EXE, raising=False)
    assert settings.default_path() == os.path.join(_APP_DIR, "settings.json")
