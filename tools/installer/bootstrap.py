"""Provision the installed app's Python environment, at install time.

This is the half of the thin installer that runs *after* Inno Setup has
copied the source tree and uv.exe into place, and after uv has created an
empty virtual environment. It runs under that environment's interpreter, so
by the time this file executes there is a real CPython (with tkinter) on the
machine even if the user had no Python at all.

Why a thin installer exists alongside the packaged builds
--------------------------------------------------------
The released ``MatchVehicleAI-windows.zip`` / ``-cuda.7z`` carry every
dependency, which makes them self-contained but large -- the CUDA one is
2.7 GiB of mostly cuDNN/cuBLAS. This path ships the ~1 MB of application
source instead and fetches the dependencies from PyPI/pytorch.org during
setup. The bytes that land on disk are the same either way; what changes is
that the download is no longer a release asset, and that the machine's own
GPU decides which torch it gets rather than the user picking an archive.

The trade is real and worth stating: this route needs working internet
during setup and resolves its wheels on the user's machine rather than in
CI. That is exactly why the versions below are pinned to the same values the
packaged builds use rather than left to pip -- an install that drifts from
the tested set is the failure mode this design is most exposed to, and
``tests/test_installer_pins.py`` fails the build if the two ever disagree.

uv rather than pip
------------------
The environment is created and populated with uv, for two reasons that both
matter here rather than being a preference:

* ``uv python install`` fetches a full python-build-standalone CPython,
  which **includes tkinter and the Tcl/Tk runtime**. Python's own
  "embeddable" Windows distribution -- the obvious choice for bundling an
  interpreter into an installer -- ships neither, and this application is a
  Tkinter GUI, so that route cannot work without hand-grafting Tcl/Tk in.
* ``uv venv`` deliberately creates environments without pip in them, so
  installs go through ``uv pip install --python <env>`` rather than
  ``python -m pip``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Same pins as .github/workflows/build-windows.yml, so a machine that
# installs through setup ends up with the environment the packaged builds
# were tested as. tests/test_installer_pins.py fails if these drift apart.
PYTHON_VERSION = "3.14"
TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu126"

# The CUDA wheel line CUDA_TORCH_INDEX points at. install_torch.py maps
# driver versions to exactly these tags, so asking it for a tag and then
# checking it against this is what keeps "the driver supports CUDA" and "we
# have an index URL for that CUDA" from drifting into disagreement.
CUDA_TAG = "cu126"


def _log(message: str) -> None:
    """Print immediately.

    Setup shows this console while it works, and the pip/uv output it
    interleaves with is the only progress indication during a download that
    can run to gigabytes. Buffered output would leave that console blank for
    minutes at a time, which reads as a hung installer.
    """
    print(message, flush=True)


def torch_requirements(cuda_tag: str | None) -> tuple[list[str], list[str]]:
    """``(packages, extra_uv_args)`` for the torch install.

    Mirrors the two matrix legs in the build workflow: default PyPI wheels
    are CPU-only, and the CUDA build comes from PyTorch's own index with a
    ``+cuXXX`` local version on each pin.
    """
    if cuda_tag is None:
        return (
            [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"],
            [],
        )
    return (
        [
            f"torch=={TORCH_VERSION}+{cuda_tag}",
            f"torchvision=={TORCHVISION_VERSION}+{cuda_tag}",
        ],
        ["--index-url", CUDA_TORCH_INDEX],
    )


def _run(command: list[str]) -> int:
    _log("> " + " ".join(command))
    try:
        return subprocess.run(command, check=False).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"could not run the command: {exc}")
        return 1


def _select_cuda_tag() -> str | None:
    """The CUDA wheel tag this machine should get, or None for the CPU build.

    Reuses install_torch's driver probing rather than repeating it: that
    module already encodes which driver versions each CUDA line needs, and
    the launchers have been relying on it for source installs. A tag it
    returns that this installer has no index URL for is treated as "no CUDA"
    -- installing a wheel line we cannot point pip at would fail the whole
    setup over what is only ever an optimization.
    """
    from install_torch import detect_driver_version, select_wheel_tag

    driver = detect_driver_version()
    if driver is None:
        _log("No usable NVIDIA GPU detected -- installing the CPU build of PyTorch.")
        return None

    tag = select_wheel_tag(driver, system="Windows")
    if tag is None:
        _log(f"NVIDIA driver {driver} predates every CUDA build we install -- "
             f"installing the CPU build. Update the driver to enable GPU support.")
        return None
    if tag != CUDA_TAG:
        _log(f"NVIDIA driver {driver} maps to {tag}, which this installer has no "
             f"index for -- installing the CPU build.")
        return None

    _log(f"NVIDIA driver {driver} detected -- installing PyTorch for {tag}.")
    return tag


def provision(uv: str, python: str, root: str) -> int:
    """Install the dependency set into the environment at ``python``."""
    lock = os.path.join(root, "requirements-lock.txt")
    if not os.path.isfile(lock):
        _log(f"error: {lock} is missing; the installed tree is incomplete.")
        return 1

    cuda_tag = _select_cuda_tag()
    packages, index_args = torch_requirements(cuda_tag)

    # Torch first and on its own, exactly as the build workflow does it: with
    # the CUDA wheels already satisfying the requirement, installing the lock
    # file afterwards leaves them alone instead of pulling the CPU wheels in
    # as an easyocr dependency.
    _log("Installing PyTorch ...")
    if _run([uv, "pip", "install", "--python", python, *packages, *index_args]):
        _log("error: installing PyTorch failed.")
        return 1

    _log("Installing the remaining dependencies ...")
    if _run([uv, "pip", "install", "--python", python, "-r", lock]):
        _log("error: installing the dependencies failed.")
        return 1

    return _verify(python, cuda_tag)


def _verify(python: str, cuda_tag: str | None) -> int:
    """Prove the freshly built environment can actually run the app.

    The packaged builds get this from the workflow's ``--selftest`` step,
    which imports every lazily-imported heavy dependency while the build can
    still be failed. An install resolved on the user's machine has no such
    gate, so it runs the equivalent here: a dependency that failed to install
    must surface now, while setup is still on screen and can say so, rather
    than the first time someone clicks Process.
    """
    _log("Verifying the installation ...")
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, "pyinstaller_entry.py")
    if _run([python, os.path.normpath(entry), "--selftest"]):
        _log("error: the installed environment failed its self-test.")
        return 1

    if cuda_tag is None:
        return 0

    # Only reached when a CUDA build was requested. A GPU that is still
    # unusable is not a failed install -- the app falls back to CPU on its
    # own -- but it is the single most common surprise this project gets
    # reported, so it is said out loud rather than discovered later from a
    # greyed-out dropdown.
    from install_torch import verify_cuda

    ok, detail = verify_cuda()
    _log(f"GPU ready: {detail}" if ok else
         f"Warning: GPU unavailable after installing {cuda_tag} -- {detail}. "
         f"The app will run on CPU.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", required=True, help="path to uv.exe")
    parser.add_argument("--python", required=True,
                        help="path to the target environment's python.exe")
    parser.add_argument("--root", required=True, help="the installed app directory")
    args = parser.parse_args(argv)

    # install_torch lives in tools/; this file is tools/installer/.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return provision(args.uv, args.python, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
