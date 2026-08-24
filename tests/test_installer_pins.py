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


_ISS = os.path.join(_ROOT, "tools", "installer", "MatchVehicleAI.iss")


def test_installer_ships_every_module_the_app_imports() -> None:
    """The .iss must carry the entry point bootstrap.py self-tests against.

    A [Files] list is a hand-maintained parallel to the repository layout,
    which is precisely the kind of thing that rots silently: a missing entry
    produces an installer that builds, installs, and only fails once the
    launcher runs.
    """
    iss = _read(_ISS)
    for required in (
        r"tools\install_torch.py",
        r"tools\pyinstaller_entry.py",
        r"tools\installer\bootstrap.py",
        r"tools\installer\provision.cmd",
        r"requirements-lock.txt",
        r"config.py",
    ):
        assert required in iss, f"{required} is not in the installer's [Files] list"


def test_every_installer_source_path_exists() -> None:
    """Every [Files] entry points at something that is actually here.

    ISCC catches this too, but only on a Windows runner several minutes into
    a build, and only for whichever entry it reaches first. Checking it here
    turns "the release job failed" into a local test failure naming the path
    -- which matters most for the rename that moves a file and leaves the
    installer silently shipping an incomplete tree.
    """
    root_token = "{#RepoRoot}"
    # Fetched into the repository root by the workflow's "Fetch uv" step and
    # git-ignored, so it is legitimately absent from a checkout. Listed here
    # rather than skipped silently: it is the one [Files] entry whose absence
    # locally says nothing about whether the build will find it.
    build_time_inputs = {"uv.exe"}
    missing = []
    for raw in re.findall(r'^Source:\s*"([^"]+)"', _read(_ISS), re.MULTILINE):
        assert raw.startswith(root_token), f"{raw} is not relative to RepoRoot"
        relative = raw[len(root_token):].lstrip("\\").replace("\\", os.sep)
        if relative in build_time_inputs:
            continue
        # A trailing \* is Inno's "everything in this directory"; the check
        # that matters for it is that the directory itself is there.
        target = os.path.join(_ROOT, relative)
        if target.endswith(os.sep + "*"):
            target = os.path.dirname(target)
        if not os.path.exists(target):
            missing.append(relative)
    assert not missing, f"the installer's [Files] list points at missing paths: {missing}"


def test_setup_section_uses_no_invented_directives() -> None:
    """Guard against a [Setup] directive that simply does not exist.

    Inno rejects an unknown directive outright, so one invented name fails
    the whole release job with a message no local run would have produced.
    The allow-list is deliberately just the directives this script uses,
    checked off against Inno's documentation once -- it is here to notice a
    *new* invented name, not to mirror Inno's full grammar.
    """
    known = {
        "AppId", "AppName", "AppVersion", "AppPublisher", "AppSupportURL",
        "DefaultDirName", "DefaultGroupName", "OutputBaseFilename", "OutputDir",
        "SetupIconFile", "Compression", "SolidCompression", "WizardStyle",
        "ArchitecturesAllowed", "ArchitecturesInstallIn64BitMode",
        "PrivilegesRequired", "PrivilegesRequiredOverridesAllowed",
    }
    section = re.search(r"^\[Setup\]\n(.*?)(?=^\[)", _read(_ISS), re.MULTILINE | re.DOTALL)
    assert section, "the .iss no longer has a [Setup] section"
    used = set(re.findall(r"^([A-Za-z][A-Za-z0-9]*)=", section.group(1), re.MULTILINE))
    assert used <= known, f"unrecognised [Setup] directive(s): {sorted(used - known)}"


def test_code_section_concatenates_strings_explicitly() -> None:
    """No two string literals sit adjacent in the [Code] section.

    [Code] is Pascal, which -- unlike C or Python -- does not join adjacent
    string literals. A wrapped message written without a '+' compiles to a
    syntax error rather than a long string, and the only place that shows up
    is ISCC on a Windows runner, minutes into a release build. The mistake is
    easy to make precisely because the broken form looks correct in every
    other language this repository uses.

    A '#13#10' between two literals is fine: a character code and a quoted
    part do form one constant, which is why only quote-to-quote pairs are
    rejected here.
    """
    code = re.search(r"^\[Code\]\n(.*)", _read(_ISS), re.MULTILINE | re.DOTALL)
    assert code, "the .iss no longer has a [Code] section"
    # Brace comments would otherwise contribute stray apostrophes ("uv's").
    body = re.sub(r"\{[^}]*\}", "", code.group(1))
    offenders = re.findall(r"'[ \t]*\r?\n[ \t]*'", body)
    assert not offenders, (
        f"{len(offenders)} pair(s) of adjacent string literals in [Code] -- "
        f"Pascal needs an explicit '+' between them"
    )


def test_no_line_starts_with_a_character_code() -> None:
    """No line begins with '#' unless it is a real ISPP directive.

    ISPP runs over the whole .iss before the Pascal compiler and reads any
    line whose first non-blank character is '#' as one of its directives. So
    a '#13#10' wrapped onto its own continuation line is not a character
    code -- it is a directive named "13", and the compile aborts during
    preprocessing. The two hazards compound: fixing an adjacent-literal
    error by breaking the line at the character code trades one compile
    failure for the other, which is exactly how this arose.
    """
    directives = {
        "define", "undef", "include", "if", "elif", "else", "endif",
        "ifdef", "ifndef", "ifexist", "ifnexist", "error", "pragma",
        "emit", "expr", "insert", "append", "sub", "endsub", "for",
    }
    bad = []
    for number, line in enumerate(_read(_ISS).splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        word = re.match(r"#\s*([A-Za-z_]\w*)", stripped)
        if not word or word.group(1) not in directives:
            bad.append(f"line {number}: {stripped[:40]}")
    assert not bad, (
        "line(s) starting with '#' that ISPP will not recognise as a "
        f"directive: {bad}"
    )
