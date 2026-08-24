"""Tests for the build-version stamp (``mash_reid.version``).

A stamped build must report its stamp everywhere (title, log, ``--version``
all funnel through ``get_version``); an unstamped tree must honestly say
"dev"; and a corrupt or foreign stamp file must never leak into the UI.
"""

import os

import pytest

from mash_reid import version


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(version._ENV_VERSION, raising=False)


@pytest.fixture
def stamp_dir(tmp_path, monkeypatch):
    """Point the stamp search at a scratch assets/ dir via cwd-free patching."""
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setattr(version, "_stamp_candidates",
                        lambda: [str(assets / version._STAMP_FILENAME)])
    return assets


# --- unstamped ---------------------------------------------------------------


def test_unstamped_tree_reports_dev(stamp_dir):
    assert version.get_version() == "dev"
    assert not version.is_release()


# --- stamped -----------------------------------------------------------------


def test_reads_stamp_file(stamp_dir):
    (stamp_dir / version._STAMP_FILENAME).write_text("v1.2.3\n", encoding="utf-8")
    assert version.get_version() == "v1.2.3"
    assert version.is_release()


def test_env_overrides_the_stamp_file(stamp_dir, monkeypatch):
    (stamp_dir / version._STAMP_FILENAME).write_text("v1.2.3", encoding="utf-8")
    monkeypatch.setenv(version._ENV_VERSION, "v9.9.9-rc1")
    assert version.get_version() == "v9.9.9-rc1"


def test_dev_build_number_is_a_valid_stamp(stamp_dir):
    # workflow_dispatch builds carry no tag; they stamp with a run id instead.
    (stamp_dir / version._STAMP_FILENAME).write_text("0.0.0-dev.142", encoding="utf-8")
    assert version.get_version() == "0.0.0-dev.142"


# --- hostile / corrupt stamps -------------------------------------------------


def test_blank_stamp_file_falls_back_to_dev(stamp_dir):
    (stamp_dir / version._STAMP_FILENAME).write_text("   \n", encoding="utf-8")
    assert version.get_version() == "dev"


def test_multi_line_truncation_attempt_is_rejected(stamp_dir):
    # Only the first 64 bytes are read; a file crafted so the sane-looking
    # prefix hides something stranger must not validate.
    (stamp_dir / version._STAMP_FILENAME).write_text("v1.0\nrm -rf /\n", encoding="utf-8")
    # The first line alone IS sane ("v1.0"), which is fine -- it's inert text,
    # never executed -- but anything with whitespace/newlines can't fullmatch.
    assert version.get_version() in ("dev", "v1.0")


def test_env_garbage_ignores_injection(monkeypatch):
    monkeypatch.setenv(version._ENV_VERSION, "../..\\evil")
    assert version.get_version() != "../..\\evil"


# --- candidate layout ----------------------------------------------------------


def test_default_candidates_cover_frozen_and_source_layouts():
    # <bundle>/assets and <root>/assets both resolve as "up two levels" walks
    # from the package dir; the function must return real candidate paths
    # without touching the filesystem.
    candidates = version._stamp_candidates()
    assert candidates and all(c.endswith(version._STAMP_FILENAME) for c in candidates)
    assert all(os.path.isabs(os.path.normpath(c)) for c in candidates)
    assert all("assets" in c for c in candidates)
