"""A broken optional runtime means "backend unavailable"; it never breaks silkern.

An installed but unusable torch or Triton -- a missing CUDA shared library,
say -- raises ``ImportError`` or ``OSError`` at import, not
``ModuleNotFoundError``. Each test runs a fresh interpreter in this checkout
with stub packages that fail that way ahead of any real install on
``sys.path``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BROKEN_TRITON = 'raise ImportError("libcuda.so.1: cannot open shared object file")'
BROKEN_TORCH = 'raise OSError("libcudart.so.12: cannot open shared object file")'
TRITON_REASON = (
    "triton failed to import (ImportError: libcuda.so.1: cannot open shared object file)"
)
TORCH_REASON = (
    "torch failed to import (OSError: libcudart.so.12: cannot open shared object file)"
)


def _python(tmp_path: pathlib.Path, packages: dict[str, str], *args: str):
    """Run this checkout's Python with stub ``packages`` first on ``sys.path``."""
    stubs = tmp_path / "stubs"
    for name, source in packages.items():
        (stubs / name).mkdir(parents=True)
        (stubs / name / "__init__.py").write_text(source + "\n")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(stubs), str(ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_broken_triton_leaves_import_and_the_oracle_working(tmp_path) -> None:
    code = """
import silkern
from silkern import kernels
assert kernels.triton is None and kernels.tl is None
assert silkern.localize_reference(
    [0], [[3]], [[0, 1]], block_size=2, dcp_size=2, dcp_rank=0
) == ([[6, -1]], [1])
try:
    silkern.localize_rowwise(None, None, None, None, None,
                             block_size=2, dcp_size=1, dcp_rank=0)
except silkern.LocalizationError as exc:
    print(exc)
"""
    result = _python(tmp_path, {"triton": BROKEN_TRITON}, "-c", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        f"triton is required for stable localization: {TRITON_REASON}"
    )


def test_broken_triton_skips_conformance_and_names_the_failure(tmp_path) -> None:
    code = "import silkern; print(silkern.conformance().skipped)"
    # An importable torch, so the sweep gets as far as importing Triton.
    result = _python(tmp_path, {"triton": BROKEN_TRITON, "torch": ""}, "-c", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == TRITON_REASON


def test_broken_torch_skips_conformance_and_names_the_failure(tmp_path) -> None:
    code = "import silkern; print(silkern.conformance().skipped)"
    result = _python(tmp_path, {"torch": BROKEN_TORCH}, "-c", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == TORCH_REASON


@pytest.mark.parametrize("flags, status", [([], 0), (["--require-device"], 2)])
def test_cli_reports_a_broken_torch_as_a_skip(tmp_path, flags, status) -> None:
    result = _python(tmp_path, {"torch": BROKEN_TORCH}, "-m", "silkern", "--json", *flags)
    assert result.returncode == status, result.stderr
    report = json.loads(result.stdout)
    assert report["skipped"] == TORCH_REASON
    assert report["ok"] is False and report["cells"] == []
    assert "Traceback" not in result.stderr


def test_broken_triton_leaves_the_benchmarks_importable(tmp_path) -> None:
    code = (
        "import bench.atomic_baseline as atomic, bench.bench_converter, bench.order_instability\n"
        "print(atomic.triton)"
    )
    result = _python(tmp_path, {"triton": BROKEN_TRITON}, "-c", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"


@pytest.mark.parametrize("module", ["bench.bench_converter", "bench.order_instability"])
def test_benchmarks_report_a_broken_torch(tmp_path, module) -> None:
    result = _python(tmp_path, {"torch": BROKEN_TORCH}, "-m", module)
    assert result.returncode == 2, result.stderr
    assert f"{TORCH_REASON}; install silkern[gpu]" in result.stderr
    assert "Traceback" not in result.stderr
