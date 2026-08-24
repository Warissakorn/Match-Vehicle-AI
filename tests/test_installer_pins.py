"""Guard that the thin installer provisions the tested environment.

The packaged ``.zip``/``.7z`` builds and the thin installer are meant to put
the *same* dependency set on a user's machine -- the only intended difference
is when the bytes are fetched. But they get there by different routes: the
workflow installs from its own ``env:`` block, and the installer from pins
compiled into ``bootstrap.py`` and ``provision.cmd``. Nothing links those
three, so a version bumped in the workflow alone would silently ship an
installer that resolves a set nobody ever tested.

That failure is invisible in CI by construction: both routes would still
build, still pass, and still run. Hence this test, which is the only thing
that makes the duplication safe. The alternative -- having the workflow read
its versions out of ``bootstrap.py`` -- founders on the interpreter pin,
since the matrix needs it before any Python exists to read it with.
"""

from __future__ import annotations

import ast
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools", "installer"))

import bootstrap  # noqa: E402

_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "build-windows.yml")
_BOOTSTRAP = os.path.join(_ROOT, "tools", "installer", "bootstrap.py")
_PROVISION = os.path.join(_ROOT, "tools", "installer", "provision.cmd")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _workflow_env() -> dict[str, str]:
    """The workflow's top-level ``env:`` mapping.

    Parsed with a regex rather than PyYAML on purpose: CI installs only
    numpy/scipy/pytest, so a test that imported yaml would skip in exactly
    the environment this guard is meant to run in.
    """
    text = _read(_WORKFLOW)
    block = re.search(r"^env:\n((?:[ \t]+.*\n|\n)+)", text, re.MULTILINE)
    assert block, "build-windows.yml no longer has a top-level env: block"
    pairs = re.findall(r'^\s+([A-Z_]+):\s*"([^"]*)"\s*$', block.group(1), re.MULTILINE)
    return dict(pairs)


def _bootstrap_constants() -> dict[str, str]:
    """Module-level string constants, read without importing."""
    tree = ast.parse(_read(_BOOTSTRAP))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


@pytest.mark.parametrize(
    ("workflow_key", "bootstrap_key"),
    [
        ("PYTHON_VERSION", "PYTHON_VERSION"),
        ("TORCH_VERSION", "TORCH_VERSION"),
        ("TORCHVISION_VERSION", "TORCHVISION_VERSION"),
        ("CUDA_TORCH_INDEX", "CUDA_TORCH_INDEX"),
    ],
)
def test_bootstrap_pins_match_the_workflow(workflow_key: str, bootstrap_key: str) -> None:
    workflow = _workflow_env()
    constants = _bootstrap_constants()
    assert workflow_key in workflow, f"{workflow_key} vanished from the workflow env"
    assert bootstrap_key in constants, f"{bootstrap_key} vanished from bootstrap.py"
    assert constants[bootstrap_key] == workflow[workflow_key], (
        f"{bootstrap_key} in bootstrap.py is {constants[bootstrap_key]!r} but the "
        f"workflow installs {workflow[workflow_key]!r} -- the installer would "
        f"provision an environment the packaged builds were never tested as"
    )


def test_provision_cmd_python_pin_matches_bootstrap() -> None:
    """provision.cmd downloads the interpreter bootstrap.py's pins assume.

    It cannot import the constant -- the interpreter that would do the
    importing is what that very line downloads -- so the value is repeated
    there and checked here.
    """
    match = re.search(r'^set "PYTHON_VERSION=([^"]+)"', _read(_PROVISION), re.MULTILINE)
    assert match, "provision.cmd no longer sets PYTHON_VERSION"
    assert match.group(1) == _bootstrap_constants()["PYTHON_VERSION"]


def test_cuda_tag_matches_the_index_url() -> None:
    """The wheel tag and the index URL must name the same CUDA line.

    bootstrap.py chooses the CUDA build by comparing install_torch's tag
    against CUDA_TAG, then installs from CUDA_TORCH_INDEX. If those two named
    different CUDA lines, a machine would be told it qualifies for one build
    and handed another.
    """
    assert bootstrap.CUDA_TORCH_INDEX.rstrip("/").endswith(bootstrap.CUDA_TAG)


def test_cpu_requirements_carry_no_local_version_and_no_index() -> None:
    packages, index_args = bootstrap.torch_requirements(None)
    assert index_args == [], "the CPU build comes from plain PyPI"
    assert packages == [
        f"torch=={bootstrap.TORCH_VERSION}",
        f"torchvision=={bootstrap.TORCHVISION_VERSION}",
    ]


def test_cuda_requirements_pin_the_local_version_and_index() -> None:
    packages, index_args = bootstrap.torch_requirements("cu126")
    assert index_args == ["--index-url", bootstrap.CUDA_TORCH_INDEX]
    assert packages == [
        f"torch=={bootstrap.TORCH_VERSION}+cu126",
        f"torchvision=={bootstrap.TORCHVISION_VERSION}+cu126",
    ]


def test_installer_ships_every_module_the_app_imports() -> None:
    """The .iss must carry the entry point bootstrap.py self-tests against.

    A [Files] list is a hand-maintained parallel to the repository layout,
    which is precisely the kind of thing that rots silently: a missing entry
    produces an installer that builds, installs, and only fails once the
    launcher runs.
    """
    iss = _read(os.path.join(_ROOT, "tools", "installer", "MatchVehicleAI.iss"))
    for required in (
        r"tools\install_torch.py",
        r"tools\pyinstaller_entry.py",
        r"tools\installer\bootstrap.py",
        r"tools\installer\provision.cmd",
        r"requirements-lock.txt",
        r"config.py",
    ):
        assert required in iss, f"{required} is not in the installer's [Files] list"
