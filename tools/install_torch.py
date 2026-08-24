"""Install a CUDA-enabled PyTorch when this machine has a usable NVIDIA GPU.

Why this exists: a plain ``pip install torch`` (what ``requirements.txt``
does) resolves to a **CPU-only wheel** on Windows, even with a real NVIDIA GPU
and driver present. ``torch.cuda.is_available()`` is then always ``False`` and
the app reports "no GPU" -- indistinguishable, from the app's side, from
actually having no GPU. This is by far the most common cause of "GPU not
detected" reports.

The launchers (``run.sh`` / ``run.bat``) call this *before* installing
``requirements.txt``, so the CUDA build is already in place and pip's plain
``torch>=2.0.0`` requirement is satisfied without replacing it.

Two things this does that the previous inline shell version could not:

1. **Picks the wheel that matches the installed driver.** The old launchers
   hardcoded ``cu121``. On a machine whose driver predates CUDA 12.1 that
   install *succeeds* but ``torch.cuda.is_available()`` still comes back
   ``False`` -- the same dead end, one step further along.
2. **Verifies afterwards and says what happened.** Whatever the outcome, the
   user gets a single line telling them whether the GPU will actually be used,
   instead of finding out later from a greyed-out dropdown.

Runs on stdlib only and must work *before* torch exists in the environment.
Deliberately never fails the launcher: a GPU is an optimization, so every
error path degrades to "carry on and let requirements.txt install the CPU
build". Exit status is always 0.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys

# Minimum NVIDIA driver version per CUDA wheel line we install, from NVIDIA's
# CUDA release notes ("CUDA Toolkit and Corresponding Driver Versions").
#
# One tier per platform these days, and that is deliberate: current PyTorch
# publishes its CUDA-12 wheels as ``cu126`` only (plus ``cu130`` on the
# CUDA-13 driver generation). Through CUDA's minor-version compatibility a
# 12.6 build runs on *any* driver from the published 12.x floor upward --
# including the newest -- so one entry covers every machine that can run a
# CUDA-12 wheel at all, and it is the same wheel line the released
# ``-windows-cuda.zip`` build ships with. A driver older than the floor gets
# no wheel: PyTorch no longer publishes the old cu118 line those machines
# would once have fallen back to, so installing anything would produce the
# exact "installs fine, CUDA unavailable" failure this helper exists to
# prevent.
_CUDA_TIERS = {
    "Windows": [
        ((527, 41), "cu126"),
    ],
    "Linux": [
        ((525, 60), "cu126"),
    ],
}

_INDEX_URL = "https://download.pytorch.org/whl/{tag}"


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Parse a dotted numeric version (``"551.86"``, ``"525.60.13"``).

    Returns ``None`` for anything that doesn't start with a number, which
    covers every way ``nvidia-smi`` can hand back something unusable (an error
    string, an empty line, ``"N/A"`` from a driver that can't report itself).
    """
    if not text:
        return None
    match = re.match(r"\s*(\d+(?:\.\d+)*)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def select_wheel_tag(driver_version: str | None, system: str | None = None) -> str | None:
    """Pick the PyTorch CUDA wheel tag (``"cu121"``) for a driver version.

    ``None`` means "no CUDA wheel is appropriate here" -- either the platform
    has no CUDA builds (macOS), the driver couldn't be read, or it is older
    than every CUDA release we'd install. In all three cases the caller should
    quietly leave the CPU build alone.
    """
    version = parse_version(driver_version)
    if version is None:
        return None

    tiers = _CUDA_TIERS.get(system or platform.system())
    if not tiers:
        return None

    for floor, tag in tiers:
        # Compare on the leading components only, so a two-part driver string
        # ("551.86", what Windows reports) and a three-part one ("525.60.13",
        # Linux) both compare correctly against a two-part floor.
        if version[: len(floor)] >= floor:
            return tag
    return None


def detect_driver_version() -> str | None:
    """The installed NVIDIA driver version, or ``None`` if there isn't one.

    A missing ``nvidia-smi`` is the normal "no NVIDIA GPU" case, not an error.
    A present-but-failing one usually means the driver is installed but not
    working (common in VMs and after a driver update without a reboot) -- also
    not something to install a CUDA build for.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    # Multi-GPU machines print one line per GPU; they share a driver, so the
    # first line is the answer.
    first_line = result.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def cuda_check_command(tag: str) -> list[str]:
    """The pip command that installs the CUDA build for ``tag``."""
    return [sys.executable, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", _INDEX_URL.format(tag=tag)]


def verify_cuda() -> tuple[bool, str]:
    """Ask the freshly installed torch whether CUDA actually works.

    Run in a subprocess rather than importing torch here: this process started
    before the install, and importing a package that appeared mid-run is
    exactly the situation where a stale import cache gives a wrong answer.
    """
    # One delimited line rather than three printed ones: the interesting
    # fields (CUDA version, GPU name) are *empty* in precisely the failure
    # cases this function exists to explain, and blank trailing lines don't
    # survive a round trip through stdout reliably.
    code = (
        "import torch;"
        "ok = torch.cuda.is_available();"
        "print('RESULT|%s|%s|%s' % ("
        "  ok, torch.version.cuda or '', torch.cuda.get_device_name(0) if ok else ''))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run the verification ({exc})"
    if result.returncode != 0:
        return False, "importing torch failed after the install"

    # Ignore anything torch or its dependencies logged before our line.
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("RESULT|")), None)
    if line is None:
        return False, "the verification returned unexpected output"
    _, available, cuda_version, gpu_name = (line.split("|", 3) + ["", "", ""])[:4]
    available, cuda_version, gpu_name = available.strip(), cuda_version.strip(), gpu_name.strip()

    if available == "True":
        return True, f"{gpu_name or 'CUDA device'} (CUDA {cuda_version})"
    if not cuda_version:
        return False, "the installed torch is still a CPU-only build"
    return False, (f"torch was built for CUDA {cuda_version} but no GPU is usable "
                   f"-- the driver may be too old, or need a reboot")


def main() -> int:
    driver = detect_driver_version()
    if driver is None:
        print("No usable NVIDIA GPU detected -- installing the CPU build of PyTorch.")
        return 0

    tag = select_wheel_tag(driver)
    if tag is None:
        print(f"NVIDIA driver {driver} is older than any CUDA build we install "
              f"-- using the CPU build of PyTorch. Update your driver to enable GPU support.")
        return 0

    print(f"NVIDIA driver {driver} detected -- installing PyTorch for {tag} ...")
    try:
        result = subprocess.run(cuda_check_command(tag), check=False)
        installed = result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Warning: could not run pip ({exc}); falling back to the CPU build.")
        return 0

    if not installed:
        print("Warning: the CUDA PyTorch install failed; falling back to the CPU build.")
        return 0

    ok, detail = verify_cuda()
    if ok:
        print(f"GPU ready: {detail}")
    else:
        print(f"Warning: GPU still unavailable after installing {tag} -- {detail}.")
        print("         The app will run on CPU. See the README's \"GPU not detected?\" section.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
