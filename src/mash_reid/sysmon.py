"""Lightweight CPU/RAM/GPU usage sampling for the GUI's resource bar.

Real runs take minutes at real-world gallery sizes (thousands of vehicles
per point) with no visibility into whether the bottleneck is CPU, RAM, or
GPU -- this gives the GUI a cheap, non-blocking snapshot to display.

Every probe is independently wrapped so a missing optional dependency
(``psutil``, ``torch``, ``pynvml``) or an environment without a GPU degrades
just that one field to ``None`` rather than raising -- callers always get a
complete, displayable dict back, matching the deferred-import discipline
used throughout this package.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_GB = 1024 ** 3


def _cpu_percent() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        # interval=None -> non-blocking, compares against the last call
        # (or the process start on the first call) instead of sleeping.
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        log.debug("psutil.cpu_percent failed", exc_info=True)
        return None


def _ram_gb() -> tuple[float | None, float | None]:
    try:
        import psutil
    except ImportError:
        return None, None
    try:
        vm = psutil.virtual_memory()
        return vm.used / _GB, vm.total / _GB
    except Exception:
        log.debug("psutil.virtual_memory failed", exc_info=True)
        return None, None


def _gpu_memory_gb() -> tuple[float | None, float | None]:
    """(used_gb, total_gb) for the default CUDA device, via torch."""
    try:
        import torch
    except ImportError:
        return None, None
    try:
        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / _GB, total / _GB
    except Exception:
        log.debug("torch.cuda.mem_get_info failed", exc_info=True)
        return None, None


def _gpu_util_percent() -> float | None:
    """GPU compute utilization, via the optional pynvml dependency.

    torch has no built-in utilization query (only memory), so this is
    genuinely optional -- a system without pynvml still gets GPU memory
    numbers, just not a utilization percentage.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        log.debug("pynvml utilization query failed", exc_info=True)
        return None


def sample() -> dict:
    """A single point-in-time snapshot. Every value is a float or ``None``."""
    ram_used, ram_total = _ram_gb()
    gpu_used, gpu_total = _gpu_memory_gb()
    return {
        "cpu_percent": _cpu_percent(),
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "gpu_util_percent": _gpu_util_percent(),
        "gpu_mem_used_gb": gpu_used,
        "gpu_mem_total_gb": gpu_total,
    }


def format_sample(s: dict) -> str:
    """Compact one-line rendering of ``sample()``'s output for a status bar."""
    cpu = s.get("cpu_percent")
    cpu_part = f"CPU {cpu:.0f}%" if cpu is not None else "CPU n/a"

    ram_used, ram_total = s.get("ram_used_gb"), s.get("ram_total_gb")
    ram_part = (f"RAM {ram_used:.1f}/{ram_total:.1f} GB"
                if ram_used is not None and ram_total is not None else "RAM n/a")

    gpu_used, gpu_total = s.get("gpu_mem_used_gb"), s.get("gpu_mem_total_gb")
    if gpu_used is not None and gpu_total is not None:
        util = s.get("gpu_util_percent")
        util_part = f"{util:.0f}% " if util is not None else ""
        gpu_part = f"GPU {util_part}{gpu_used:.1f}/{gpu_total:.1f} GB"
    else:
        gpu_part = "GPU n/a"

    return "  |  ".join((cpu_part, ram_part, gpu_part))
