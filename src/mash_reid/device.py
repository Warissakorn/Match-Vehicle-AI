"""Resolve and describe the compute device (CPU/CUDA) for detection + embedding.

``config.PipelineConfig.device`` has existed since the start ("None -> auto
(cuda if available else cpu)"), but nothing actually set it -- the GUI never
built a selector, the CLI never had a flag, and the detector only called
``.to(device)`` when a device string happened to be truthy, which it never
was. GPUs went unused even when present. This module centralizes the "auto"
resolution (previously duplicated ad hoc inside the embedder) so the
detector, embedder, and OCR reader all agree on the same device string.

``torch`` is a deferred import -- probing CUDA is the only thing here that
needs it, and ``resolve_device`` accepts an injectable override so tests can
exercise the resolution logic without torch installed (matching the
deferred-import discipline used by ``detector.py``/``embedder.py``).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_AUTO_VALUES = (None, "", "auto")


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        log.warning("torch.cuda.is_available() raised; treating CUDA as unavailable", exc_info=True)
        return False


def resolve_device(requested: str | None = None, cuda_available: bool | None = None) -> str:
    """Resolve a device string for torch/ultralytics.

    ``requested`` is passed straight through when it names a specific device
    (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...); ``None``/``""``/``"auto"``
    (case-insensitive) probes CUDA and picks the best available. Probing
    happens once per call -- callers that resolve per-model (detector,
    embedder) should cache the result rather than calling this per frame.

    ``cuda_available`` overrides the CUDA probe -- tests use this to exercise
    both branches without torch installed.
    """
    if requested is not None and requested.strip().lower() not in ("", "auto"):
        return requested.strip()

    if cuda_available is None:
        cuda_available = _torch_cuda_available()
    return "cuda" if cuda_available else "cpu"


def describe_device(device: str) -> str:
    """Human-readable description of a resolved device, for status displays.

    Falls back to the raw device string whenever torch isn't installed or
    the device index can't be queried -- this is a display aid, not a
    correctness-critical path, so it never raises.
    """
    if not device or device == "cpu":
        return "CPU"
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            return f"CUDA ({torch.cuda.get_device_name(idx)})"
    except Exception:
        log.debug("Could not describe device %r", device, exc_info=True)
    return device


def list_available_devices() -> list[str]:
    """Devices worth offering in a picker: always Auto/CPU, plus CUDA if present."""
    devices = ["auto", "cpu"]
    if _torch_cuda_available():
        devices.append("cuda")
    return devices


def diagnose_cuda_unavailable() -> str:
    """One-line, human-readable reason CUDA isn't available -- or ``""`` if it is.

    ``torch.cuda.is_available() == False`` conflates two very different
    situations: no CUDA-capable GPU/driver, versus a torch build that was
    never compiled with CUDA support at all (the default on some platforms
    for a plain ``pip install torch`` -- see the README's GPU section).
    ``torch.backends.cuda.is_built()`` distinguishes them: it's about the
    installed binary, not the hardware, so it's true even on a machine with
    no GPU as long as the *build* supports CUDA. Meant to be shown to a user
    who expected to see "cuda" in the device list and didn't.
    """
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    try:
        if torch.cuda.is_available():
            return ""
        if not torch.backends.cuda.is_built():
            return ("This PyTorch build has no CUDA support (CPU-only wheel). "
                     "Reinstall with a CUDA index -- see README's GPU section.")
        return "No CUDA-capable GPU or driver detected"
    except Exception as exc:
        log.warning("Could not diagnose CUDA availability", exc_info=True)
        return f"Could not determine CUDA availability ({exc})"
