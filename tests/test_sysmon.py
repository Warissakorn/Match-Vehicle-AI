"""Tests for the resource-usage sampler behind the GUI's status bar.

The "everything degrades to None/n/a without the optional dependency"
tests simulate a missing psutil/torch/pynvml via a faked import, so they
run the same whether or not those packages happen to be installed (CI
installs none of them). Tests exercising real values are gated with
``pytest.importorskip``.
"""

import builtins

import pytest

from mash_reid import sysmon


def _block_imports(monkeypatch, *names):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in names:
            raise ImportError(f"no {name} for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_sample_has_all_expected_keys():
    s = sysmon.sample()
    assert set(s) == {
        "cpu_percent", "ram_used_gb", "ram_total_gb",
        "gpu_util_percent", "gpu_mem_used_gb", "gpu_mem_total_gb",
    }


def test_sample_degrades_fully_without_any_optional_dependency(monkeypatch):
    _block_imports(monkeypatch, "psutil", "torch", "pynvml")
    s = sysmon.sample()
    assert s == {
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "gpu_util_percent": None,
        "gpu_mem_used_gb": None,
        "gpu_mem_total_gb": None,
    }


def test_cpu_percent_none_without_psutil(monkeypatch):
    _block_imports(monkeypatch, "psutil")
    assert sysmon._cpu_percent() is None


def test_ram_gb_none_without_psutil(monkeypatch):
    _block_imports(monkeypatch, "psutil")
    assert sysmon._ram_gb() == (None, None)


def test_gpu_memory_none_without_torch(monkeypatch):
    _block_imports(monkeypatch, "torch")
    assert sysmon._gpu_memory_gb() == (None, None)


def test_gpu_util_none_without_pynvml(monkeypatch):
    _block_imports(monkeypatch, "pynvml")
    assert sysmon._gpu_util_percent() is None


def test_cpu_percent_degrades_when_psutil_call_itself_raises(monkeypatch):
    # Not "psutil missing" (ImportError) but "psutil present, call broke" --
    # a different failure mode, also expected to degrade rather than raise.
    pytest.importorskip("psutil")
    import psutil

    def boom(interval=None):
        raise RuntimeError("simulated psutil failure")

    monkeypatch.setattr(psutil, "cpu_percent", boom)
    assert sysmon._cpu_percent() is None


def test_real_cpu_percent_when_psutil_installed():
    pytest.importorskip("psutil")
    value = sysmon._cpu_percent()
    assert value is None or value >= 0.0


def test_real_ram_when_psutil_installed():
    pytest.importorskip("psutil")
    used, total = sysmon._ram_gb()
    assert used is not None and total is not None
    assert 0 <= used <= total


# --- format_sample -----------------------------------------------------------

def test_format_sample_all_present():
    s = {
        "cpu_percent": 42.0,
        "ram_used_gb": 4.1,
        "ram_total_gb": 16.0,
        "gpu_util_percent": 12.0,
        "gpu_mem_used_gb": 1.2,
        "gpu_mem_total_gb": 8.0,
    }
    text = sysmon.format_sample(s)
    assert "CPU 42%" in text
    assert "RAM 4.1/16.0 GB" in text
    assert "GPU 12% 1.2/8.0 GB" in text


def test_format_sample_all_missing():
    s = {
        "cpu_percent": None, "ram_used_gb": None, "ram_total_gb": None,
        "gpu_util_percent": None, "gpu_mem_used_gb": None, "gpu_mem_total_gb": None,
    }
    text = sysmon.format_sample(s)
    assert "CPU n/a" in text
    assert "RAM n/a" in text
    assert "GPU n/a" in text


def test_format_sample_gpu_memory_without_util():
    # e.g. torch present (memory numbers) but pynvml absent (no utilization).
    s = {
        "cpu_percent": 10.0, "ram_used_gb": 2.0, "ram_total_gb": 8.0,
        "gpu_util_percent": None, "gpu_mem_used_gb": 0.5, "gpu_mem_total_gb": 4.0,
    }
    text = sysmon.format_sample(s)
    assert "GPU 0.5/4.0 GB" in text
