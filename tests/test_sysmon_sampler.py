"""Tests for ``sysmon.Sampler`` -- the background resource sampler.

The GUI must never run the probes on the Tk thread (the GPU ones pull torch
and NVML init), so the sampler owns them off-thread and hands over plain
dicts. These tests pin: the pre-start shape, snapshot publication, clean
stopping, single-thread discipline, and the permanent give-up when pynvml is
missing. No test may require an actual GPU or even psutil.
"""

import sys
import threading
import time

from mash_reid import sysmon


def test_latest_before_start_is_all_none_with_known_fields():
    s = sysmon.Sampler(interval_ms=1000)
    snap = s.latest()
    assert set(snap) == set(sysmon.Sampler._FIELDS)
    assert all(v is None for v in snap.values())
    # And it renders as the all-n/a status line rather than crashing.
    text = sysmon.format_sample(snap)
    assert text.count("n/a") == 3


def test_sampler_publishes_probe_results_and_stops_cleanly(monkeypatch):
    monkeypatch.setattr(sysmon, "_cpu_percent", lambda: 42.0)
    monkeypatch.setattr(sysmon, "_ram_gb", lambda: (2.0, 8.0))
    monkeypatch.setattr(sysmon, "_gpu_memory_gb", lambda: (1.5, 6.0))

    s = sysmon.Sampler(interval_ms=25)
    s._gpu_util = lambda: 77.0  # bypass the NVML path entirely
    s.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if s.latest()["cpu_percent"] == 42.0:
            break
        time.sleep(0.01)
    snap = s.latest()
    assert snap == {
        "cpu_percent": 42.0,
        "ram_used_gb": 2.0,
        "ram_total_gb": 8.0,
        "gpu_util_percent": 77.0,
        "gpu_mem_used_gb": 1.5,
        "gpu_mem_total_gb": 6.0,
    }

    s.stop()
    s._thread.join(timeout=5)
    assert not s._thread.is_alive()


def test_start_twice_keeps_a_single_thread():
    s = sysmon.Sampler(interval_ms=1000)
    s.start()
    try:
        first = s._thread
        s.start()
        assert s._thread is first
        alive = [t for t in threading.enumerate() if t.name == "sysmon-sampler"]
        assert len(alive) == 1
    finally:
        s.stop()
        s._thread.join(timeout=5)


def test_missing_pynvml_gives_up_permanently(monkeypatch):
    # A None entry in sys.modules makes `import pynvml` raise ImportError --
    # the CI/GPU-less path. The sampler must mark NVML dead and stop holding
    # out for a driver that will not appear mid-session.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    s = sysmon.Sampler(interval_ms=1000)

    assert s._gpu_util() is None
    assert s._gpu_util() is None
    assert s._nvml_dead
    assert s._nvml_handle is None


def test_nvml_failure_degrades_only_the_utilization_field(monkeypatch):
    monkeypatch.setattr(sysmon, "_cpu_percent", lambda: 11.0)
    monkeypatch.setattr(sysmon, "_ram_gb", lambda: (1.0, 4.0))
    monkeypatch.setattr(sysmon, "_gpu_memory_gb", lambda: (None, None))
    monkeypatch.setitem(sys.modules, "pynvml", None)

    s = sysmon.Sampler(interval_ms=1000)
    snap = s._sample_once()
    assert snap["cpu_percent"] == 11.0
    assert snap["gpu_util_percent"] is None


def test_interval_floor():
    # Absurdly small intervals are clamped so a typo can't turn the sampler
    # into a busy loop.
    assert sysmon.Sampler(interval_ms=0)._interval_s >= 0.2
