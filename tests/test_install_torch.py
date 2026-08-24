"""Tests for the launcher's CUDA-wheel picker.

The value here is in ``select_wheel_tag``/``parse_version``: they decide which
PyTorch build lands on a user's machine, and getting that wrong is invisible
until the app reports "no GPU" -- the exact failure this helper exists to
prevent. Both are pure functions, so they're tested directly for every
platform and driver tier without needing an NVIDIA GPU (this repo's CI has
none, and neither does any dev sandbox).

The subprocess-backed parts are covered for their failure modes only, since
those are what must never take the launcher down with them.
"""

import subprocess

import install_torch


# --- parse_version -----------------------------------------------------------

def test_parse_two_part_version():
    assert install_torch.parse_version("551.86") == (551, 86)


def test_parse_three_part_version():
    assert install_torch.parse_version("525.60.13") == (525, 60, 13)


def test_parse_strips_surrounding_whitespace():
    assert install_torch.parse_version("  537.13 \n") == (537, 13)


def test_parse_single_number():
    assert install_torch.parse_version("470") == (470,)


def test_parse_none_is_none():
    assert install_torch.parse_version(None) is None


def test_parse_empty_is_none():
    assert install_torch.parse_version("") is None


def test_parse_non_numeric_is_none():
    # What a broken driver reports instead of a version.
    assert install_torch.parse_version("N/A") is None


def test_parse_error_text_is_none():
    assert install_torch.parse_version("Failed to initialize NVML") is None


# --- select_wheel_tag: Windows tiers ----------------------------------------
# Current PyTorch publishes one CUDA-12 line (cu126) whose minor-version
# compatibility covers every driver from the 12.x floor upward, so there is a
# single qualifying tier and everything below it gets nothing.


def test_windows_modern_driver_gets_cu126():
    assert install_torch.select_wheel_tag("551.61", "Windows") == "cu126"
    assert install_torch.select_wheel_tag("580.10", "Windows") == "cu126"


def test_windows_floor_driver_gets_cu126():
    assert install_torch.select_wheel_tag("527.41", "Windows") == "cu126"
    assert install_torch.select_wheel_tag("537.13", "Windows") == "cu126"


def test_windows_pre_floor_driver_gets_nothing():
    # PyTorch no longer publishes the old cu118 line these machines once fell
    # back to -- installing anything would yield a wheel that cannot init.
    assert install_torch.select_wheel_tag("452.39", "Windows") is None
    assert install_torch.select_wheel_tag("472.12", "Windows") is None
    assert install_torch.select_wheel_tag("527.40", "Windows") is None


def test_windows_ancient_driver_gets_nothing():
    # Older than every CUDA build we'd install -> stay on the CPU wheel
    # rather than install something that can't initialize.
    assert install_torch.select_wheel_tag("451.00", "Windows") is None


def test_windows_tier_boundary_is_inclusive():
    # Exactly at the published floor must qualify, not fall to nothing.
    assert install_torch.select_wheel_tag("527.41", "Windows") == "cu126"


# --- select_wheel_tag: Linux tiers ------------------------------------------

def test_linux_modern_driver_gets_cu126():
    assert install_torch.select_wheel_tag("550.54.14", "Linux") == "cu126"


def test_linux_floor_driver_gets_cu126():
    assert install_torch.select_wheel_tag("525.60.13", "Linux") == "cu126"


def test_linux_pre_floor_driver_gets_nothing():
    assert install_torch.select_wheel_tag("470.82.01", "Linux") is None


def test_linux_ancient_driver_gets_nothing():
    assert install_torch.select_wheel_tag("450.51.06", "Linux") is None


def test_three_part_driver_compares_against_two_part_floor():
    # The floors are two-part; a three-part driver string must not compare as
    # "longer therefore greater" and skip a tier.
    assert install_torch.select_wheel_tag("525.59.99", "Linux") is None


# --- select_wheel_tag: platforms and bad input -------------------------------

def test_macos_never_gets_a_cuda_wheel():
    # No CUDA builds are published for macOS at all.
    assert install_torch.select_wheel_tag("551.86", "Darwin") is None


def test_unknown_platform_gets_nothing():
    assert install_torch.select_wheel_tag("551.86", "SunOS") is None


def test_no_driver_version_gets_nothing():
    assert install_torch.select_wheel_tag(None, "Windows") is None


def test_unparseable_driver_version_gets_nothing():
    assert install_torch.select_wheel_tag("N/A", "Windows") is None


# --- cuda_check_command ------------------------------------------------------

def test_install_command_targets_the_running_interpreter_and_index():
    cmd = install_torch.cuda_check_command("cu126")
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "torch" in cmd and "torchvision" in cmd
    assert "https://download.pytorch.org/whl/cu126" in cmd


# --- detect_driver_version: must degrade, never raise ------------------------

def _fake_run(monkeypatch, *, returncode=0, stdout="", raises=None):
    def fake(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(install_torch.subprocess, "run", fake)


def test_detect_returns_version_from_nvidia_smi(monkeypatch):
    _fake_run(monkeypatch, stdout="551.86\n")
    assert install_torch.detect_driver_version() == "551.86"


def test_detect_uses_first_line_on_multi_gpu(monkeypatch):
    # One line per GPU, but they share a driver.
    _fake_run(monkeypatch, stdout="551.86\n551.86\n")
    assert install_torch.detect_driver_version() == "551.86"


def test_detect_none_when_nvidia_smi_missing(monkeypatch):
    _fake_run(monkeypatch, raises=FileNotFoundError("no nvidia-smi"))
    assert install_torch.detect_driver_version() is None


def test_detect_none_when_nvidia_smi_fails(monkeypatch):
    # Driver installed but not working -- e.g. needs a reboot.
    _fake_run(monkeypatch, returncode=9, stdout="")
    assert install_torch.detect_driver_version() is None


def test_detect_none_when_nvidia_smi_times_out(monkeypatch):
    _fake_run(monkeypatch, raises=subprocess.TimeoutExpired("nvidia-smi", 30))
    assert install_torch.detect_driver_version() is None


def test_detect_none_on_empty_output(monkeypatch):
    _fake_run(monkeypatch, stdout="\n")
    assert install_torch.detect_driver_version() is None


# --- verify_cuda -------------------------------------------------------------

def test_verify_reports_success_with_gpu_name(monkeypatch):
    _fake_run(monkeypatch, stdout="RESULT|True|12.1|NVIDIA GeForce RTX 3060\n")
    ok, detail = install_torch.verify_cuda()
    assert ok is True
    assert "RTX 3060" in detail and "12.1" in detail


def test_verify_detects_cpu_only_wheel(monkeypatch):
    # torch.version.cuda empty -> the CUDA install didn't take effect.
    _fake_run(monkeypatch, stdout="RESULT|False||\n")
    ok, detail = install_torch.verify_cuda()
    assert ok is False
    assert "CPU-only build" in detail


def test_verify_detects_cuda_build_that_cannot_see_the_gpu(monkeypatch):
    # The failure mode the old hardcoded cu121 launcher produced silently.
    _fake_run(monkeypatch, stdout="RESULT|False|12.4|\n")
    ok, detail = install_torch.verify_cuda()
    assert ok is False
    assert "12.4" in detail and "driver" in detail


def test_verify_handles_torch_import_failure(monkeypatch):
    _fake_run(monkeypatch, returncode=1, stdout="")
    ok, detail = install_torch.verify_cuda()
    assert ok is False
    assert "importing torch failed" in detail


def test_verify_handles_truncated_output(monkeypatch):
    _fake_run(monkeypatch, stdout="some unrelated warning\n")
    ok, detail = install_torch.verify_cuda()
    assert ok is False
    assert "unexpected output" in detail


def test_verify_handles_subprocess_error(monkeypatch):
    _fake_run(monkeypatch, raises=OSError("exec failed"))
    ok, detail = install_torch.verify_cuda()
    assert ok is False
    assert "could not run the verification" in detail


# --- main: always exits 0 so the launcher continues --------------------------

def test_main_succeeds_without_a_gpu(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: None)
    assert install_torch.main() == 0
    assert "No usable NVIDIA GPU" in capsys.readouterr().out


def test_main_skips_install_for_too_old_a_driver(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: "400.00")
    monkeypatch.setattr(install_torch, "select_wheel_tag", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(install_torch.subprocess, "run", lambda *a, **k: calls.append(a))

    assert install_torch.main() == 0
    assert calls == []  # nothing installed
    assert "older than any CUDA build" in capsys.readouterr().out


def test_main_reports_success_after_a_working_install(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: "551.86")
    monkeypatch.setattr(install_torch.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(install_torch, "verify_cuda", lambda: (True, "RTX 4070 (CUDA 12.4)"))

    assert install_torch.main() == 0
    assert "GPU ready: RTX 4070" in capsys.readouterr().out


def test_main_survives_a_failed_pip_install(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: "551.86")
    monkeypatch.setattr(install_torch.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 1))

    assert install_torch.main() == 0
    assert "falling back to the CPU build" in capsys.readouterr().out


def test_main_survives_pip_being_unrunnable(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: "551.86")

    def boom(*a, **k):
        raise OSError("pip exploded")

    monkeypatch.setattr(install_torch.subprocess, "run", boom)
    assert install_torch.main() == 0
    assert "could not run pip" in capsys.readouterr().out


def test_main_warns_when_gpu_still_unavailable_after_install(monkeypatch, capsys):
    monkeypatch.setattr(install_torch, "detect_driver_version", lambda: "527.41")
    monkeypatch.setattr(install_torch.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(install_torch, "verify_cuda",
                        lambda: (False, "the installed torch is still a CPU-only build"))

    assert install_torch.main() == 0
    out = capsys.readouterr().out
    assert "GPU still unavailable" in out
    assert "run on CPU" in out
