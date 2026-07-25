"""Tests for GUI settings persistence -- pure stdlib json, no GUI needed."""

import json
import os

from mash_reid import settings


def _some_values():
    return {
        "dir_a": "/data/A",
        "dir_b": "/data/B",
        "model_key": "yolov8n.pt",
        "device": "auto",
        "threshold": 0.6,
        "det_conf": 0.35,
        "min_travel": 0.0,
        "max_travel": 600.0,
        "use_gate": True,
        "one_to_one": False,
        "extract_interval": 1.0,
        "last_video_dir": "/videos",
    }


def test_round_trip(tmp_path):
    path = str(tmp_path / "settings.json")
    values = _some_values()
    settings.save(values, path)
    assert settings.load(path) == values


def test_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "does_not_exist.json")
    assert settings.load(path) == {}


def test_corrupt_json_returns_empty(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        fh.write("{not valid json")
    assert settings.load(path) == {}


def test_non_object_json_returns_empty(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump([1, 2, 3], fh)
    assert settings.load(path) == {}


def test_unknown_keys_are_dropped(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump({"dir_a": "/x", "evil_key": "rm -rf /"}, fh)
    result = settings.load(path)
    assert result == {"dir_a": "/x"}
    assert "evil_key" not in result


def test_wrong_type_values_are_dropped(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump({
            "dir_a": 12345,          # should be str
            "threshold": "high",     # should be float
            "use_gate": "yes",       # should be bool, not a truthy string
            "det_conf": 0.5,         # valid
        }, fh)
    result = settings.load(path)
    assert result == {"det_conf": 0.5}


def test_int_is_accepted_for_float_fields(tmp_path):
    # JSON doesn't distinguish "1" from "1.0" on the wire in every writer;
    # an int value for a float-typed field should coerce, not drop.
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump({"threshold": 1}, fh)
    result = settings.load(path)
    assert result == {"threshold": 1.0}
    assert isinstance(result["threshold"], float)


def test_bool_is_not_accepted_for_float_fields(tmp_path):
    # bool is a subclass of int in Python -- must not silently pass as a
    # float-typed setting (True == 1 would otherwise sneak through).
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump({"threshold": True}, fh)
    result = settings.load(path)
    assert result == {}


def test_save_only_writes_whitelisted_keys(tmp_path):
    path = str(tmp_path / "settings.json")
    settings.save({"dir_a": "/x", "not_a_real_setting": 42}, path)
    with open(path) as fh:
        raw = json.load(fh)
    assert raw == {"dir_a": "/x"}


def test_save_creates_readable_json(tmp_path):
    path = str(tmp_path / "settings.json")
    settings.save({"dir_a": "/x"}, path)
    with open(path) as fh:
        text = fh.read()
    assert '"dir_a"' in text
    json.loads(text)  # must parse cleanly


def test_save_failure_does_not_raise(tmp_path):
    # Write to a path whose parent directory doesn't exist -> open() raises
    # inside save(), which must be caught rather than propagated.
    bad_path = str(tmp_path / "no_such_dir" / "settings.json")
    settings.save({"dir_a": "/x"}, bad_path)  # must not raise
    assert not os.path.exists(bad_path)


def test_default_path_is_next_to_config():
    path = settings.default_path()
    assert os.path.basename(path) == "settings.json"
    import config
    assert os.path.dirname(path) == os.path.dirname(os.path.abspath(config.__file__))
