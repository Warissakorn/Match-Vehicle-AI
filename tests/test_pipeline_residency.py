"""Tests for the model residency cache (``pipeline.get_pipeline``).

The GUI re-runs Process with tweaked thresholds constantly; the models must
stay resident across those runs, rebuild only when a constructor-shaping
setting changes, and always track the caller's per-call knobs (confidence,
class ids, box area) even on a cache hit.
"""

import threading

import pytest

from mash_reid import pipeline


@pytest.fixture(autouse=True)
def clean_cache():
    """Start and end every test with an empty slot."""
    pipeline.drop_pipeline()
    yield
    pipeline.drop_pipeline()


def _cfg(**overrides):
    import config

    return config.PipelineConfig(**overrides)


def test_same_settings_return_the_same_objects():
    first = pipeline.get_pipeline(_cfg())
    second = pipeline.get_pipeline(_cfg())
    assert first is second
    assert first[0] is second[0] and first[1] is second[1]


def test_per_call_knobs_do_not_trigger_a_reload():
    # detection_conf / class ids / min box area are read at detect time;
    # tweaking them must reuse the loaded models.
    first = pipeline.get_pipeline(_cfg(detection_conf=0.25))
    second = pipeline.get_pipeline(_cfg(detection_conf=0.6, min_box_area=999))
    assert first is second
    # ...and the cached detector must now carry the FRESH config, not the one
    # it was built with.
    assert second[0].cfg.detection_conf == 0.6
    assert second[0].cfg.min_box_area == 999


def test_changing_weights_or_device_rebuilds():
    a = pipeline.get_pipeline(_cfg(yolo_weights="yolo11n.pt"))
    b = pipeline.get_pipeline(_cfg(yolo_weights="yolov8n.pt"))
    assert a is not b and a[0] is not b[0]

    c = pipeline.get_pipeline(_cfg(device="cpu"))
    d = pipeline.get_pipeline(_cfg(device="cuda"))
    assert c is not d and c[1] is not d[1]


def test_changing_batch_size_rebuilds_embedder():
    a = pipeline.get_pipeline(_cfg(embed_batch_size=16))
    b = pipeline.get_pipeline(_cfg(embed_batch_size=64))
    assert a[1] is not b[1]
    assert a[0] is not b[0]  # whole pair is rebuilt together


def test_drop_forces_a_rebuild():
    a = pipeline.get_pipeline(_cfg())
    pipeline.drop_pipeline()
    b = pipeline.get_pipeline(_cfg())
    assert a is not b


def test_concurrent_callers_never_crash_and_settle_consistently():
    # The GUI disables Process while running, but the prefetcher (and any
    # future background user) shares this cache; a torn read would hand back
    # a half-built pair. Hammer it from several threads with mixed keys.
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def run(device):
        try:
            barrier.wait()
            for _ in range(20):
                pair = pipeline.get_pipeline(_cfg(device=device))
                detector, embedder = pair
                assert embedder._batch_size >= 1
                assert hasattr(detector, "detect")
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(dev,), daemon=True)
               for dev in ("cpu", "cpu", "cpu", "cuda")]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)
    assert errors == []


def test_none_uses_default_config():
    import config

    pair = pipeline.get_pipeline()
    assert pair[0].cfg.yolo_weights == config.PipelineConfig().yolo_weights
