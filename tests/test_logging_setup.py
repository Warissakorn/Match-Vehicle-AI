"""Tests for logging setup: a run must produce a readable log file."""

import logging
import os

from mash_reid import logging_setup


def test_setup_logging_creates_file_and_records_messages(tmp_path):
    log_dir = str(tmp_path / "logs")
    log_path = logging_setup.setup_logging(log_dir, console=False)

    assert os.path.isfile(log_path)
    assert log_path.startswith(log_dir)

    # A message from any mash_reid child logger must land in the file.
    logging.getLogger("mash_reid.pipeline").info("hello-from-test-42")
    for handler in logging.getLogger("mash_reid").handlers:
        handler.flush()

    with open(log_path, encoding="utf-8") as fh:
        contents = fh.read()
    assert "hello-from-test-42" in contents
    assert "run started" in contents


def test_setup_logging_handles_unicode_messages(tmp_path):
    log_path = logging_setup.setup_logging(str(tmp_path / "logs"), console=False)
    logging.getLogger("mash_reid.video_extractor").info("จุด A ทดสอบ")
    for handler in logging.getLogger("mash_reid").handlers:
        handler.flush()
    with open(log_path, encoding="utf-8") as fh:
        assert "จุด A ทดสอบ" in fh.read()
