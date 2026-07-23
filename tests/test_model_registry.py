"""Tests for the detection-model catalog. Pure data — no network or torch."""

import pytest

import config
from mash_reid import model_registry


def test_default_key_is_in_catalog_and_matches_config():
    assert model_registry.is_known(model_registry.DEFAULT_KEY)
    # config and the registry must agree on the default model.
    assert config.YOLO_WEIGHTS == model_registry.DEFAULT_KEY
    assert model_registry.default().key == model_registry.DEFAULT_KEY


def test_keys_are_unique_and_look_like_weights():
    keys = model_registry.keys()
    assert len(keys) == len(set(keys))
    assert keys, "catalog must not be empty"
    for key in keys:
        assert key.endswith(".pt")


def test_get_valid_and_invalid():
    info = model_registry.get(model_registry.DEFAULT_KEY)
    assert info.key == model_registry.DEFAULT_KEY
    assert info.approx_mb > 0
    with pytest.raises(KeyError):
        model_registry.get("does-not-exist.pt")


def test_is_known_rejects_custom_paths():
    assert not model_registry.is_known("/home/me/custom_weights.pt")
    assert not model_registry.is_known("yolov9e.pt")


def test_recommended_models_exist():
    # "Keep up to date" relies on there being a newer, recommended family.
    recommended = [m for m in model_registry.all_models() if m.recommended]
    assert recommended
    assert all("★" in m.display_name for m in recommended)
