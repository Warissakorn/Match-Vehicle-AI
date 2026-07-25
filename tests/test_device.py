"""Tests for device resolution (no real GPU or torch installation needed --
the CUDA probe is injectable so these run the same in CI as anywhere else).
"""

import sys
import types

import pytest

from mash_reid import device


def _fake_torch(cuda_available, cuda_built, raise_on_is_available=False):
    """A minimal stand-in for the ``torch`` module, just deep enough for
    ``diagnose_cuda_unavailable`` -- ``torch.cuda.is_available()`` and
    ``torch.backends.cuda.is_built()``.
    """
    mod = types.ModuleType("torch")

    def is_available():
        if raise_on_is_available:
            raise RuntimeError("simulated CUDA init failure")
        return cuda_available

    mod.cuda = types.SimpleNamespace(is_available=is_available)
    mod.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_built=lambda: cuda_built))
    return mod


def test_auto_none_resolves_to_cpu_when_no_cuda():
    assert device.resolve_device(None, cuda_available=False) == "cpu"


def test_auto_none_resolves_to_cuda_when_available():
    assert device.resolve_device(None, cuda_available=True) == "cuda"


def test_auto_string_resolves_same_as_none():
    assert device.resolve_device("auto", cuda_available=True) == "cuda"
    assert device.resolve_device("AUTO", cuda_available=False) == "cpu"


def test_empty_string_is_treated_as_auto():
    assert device.resolve_device("", cuda_available=True) == "cuda"


def test_explicit_cpu_is_never_overridden():
    assert device.resolve_device("cpu", cuda_available=True) == "cpu"


def test_explicit_cuda_index_passes_through():
    assert device.resolve_device("cuda:1", cuda_available=False) == "cuda:1"


def test_requested_device_is_stripped():
    assert device.resolve_device("  cuda  ", cuda_available=False) == "cuda"


def test_describe_device_cpu():
    assert device.describe_device("cpu") == "CPU"


def test_describe_device_empty_is_cpu():
    assert device.describe_device("") == "CPU"


def test_describe_device_falls_back_to_raw_string_without_torch(monkeypatch):
    # Simulate "no torch installed" by making the import fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device.describe_device("cuda") == "cuda"


def test_list_available_devices_always_has_auto_and_cpu(monkeypatch):
    monkeypatch.setattr(device, "_torch_cuda_available", lambda: False)
    assert device.list_available_devices() == ["auto", "cpu"]


def test_list_available_devices_includes_cuda_when_present(monkeypatch):
    monkeypatch.setattr(device, "_torch_cuda_available", lambda: True)
    assert device.list_available_devices() == ["auto", "cpu", "cuda"]


def test_torch_cuda_available_false_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device._torch_cuda_available() is False


def test_torch_cuda_available_reflects_real_torch_when_installed():
    torch = pytest.importorskip("torch")
    assert device._torch_cuda_available() == bool(torch.cuda.is_available())


# --- diagnose_cuda_unavailable: distinguishes *why* CUDA isn't available ----

def test_diagnose_returns_empty_when_cuda_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, cuda_built=True))
    assert device.diagnose_cuda_unavailable() == ""


def test_diagnose_no_torch_installed(monkeypatch):
    import builtins

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device.diagnose_cuda_unavailable() == "PyTorch is not installed"


def test_diagnose_cpu_only_wheel(monkeypatch):
    # torch installed, but never compiled with CUDA support -- the gotcha a
    # plain `pip install torch` produces on some platforms.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, cuda_built=False))
    reason = device.diagnose_cuda_unavailable()
    assert "CPU-only wheel" in reason


def test_diagnose_cuda_build_but_no_gpu(monkeypatch):
    # torch has real CUDA support, but no GPU/driver on this machine.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, cuda_built=True))
    reason = device.diagnose_cuda_unavailable()
    assert reason == "No CUDA-capable GPU or driver detected"


def test_diagnose_handles_unexpected_probe_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(cuda_available=False, cuda_built=True, raise_on_is_available=True))
    reason = device.diagnose_cuda_unavailable()
    assert "Could not determine CUDA availability" in reason


def test_diagnose_real_torch_does_not_raise():
    # Whatever the real installed torch build reports, this must never raise.
    pytest.importorskip("torch")
    reason = device.diagnose_cuda_unavailable()
    assert isinstance(reason, str)
