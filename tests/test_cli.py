"""Every command-line entry point stays usable as typed.

Usage lines name the command a reader can type back, help works when Python
strips docstrings (``-OO``), and the examples' self-checks survive ``-O``.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys

import pytest

from silkern._cli import program_name

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMANDS = [
    ("silkern", ()),
    ("silkern.mlx_verify", ()),
    ("bench.bench_mlx", ()),
    ("bench.bench_mlx_attention", ()),
    ("bench.bench_converter", ()),
    ("bench.order_instability", ()),
    ("bench.summarize_mlx", ()),
    ("examples.mlx_sparse_attention", ("mlx",)),
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300
    )


@pytest.mark.parametrize("module, needs", COMMANDS, ids=[module for module, _ in COMMANDS])
def test_help_works_with_docstrings_stripped(module: str, needs: tuple[str, ...]) -> None:
    for package in needs:
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not installed")
    result = _run("-OO", "-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage:")


@pytest.mark.parametrize("module", ["silkern", "silkern.mlx_verify"])
def test_usage_names_the_command_a_user_typed(module: str) -> None:
    result = _run("-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"usage: python -m {module} ")


def test_program_name_keeps_a_console_script_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["/venv/bin/silkern-verify-mlx"])
    assert program_name("silkern.mlx_verify") == "silkern-verify-mlx"
    monkeypatch.setattr(sys, "argv", ["/checkout/silkern/__main__.py"])
    assert program_name("silkern") == "python -m silkern"


def test_examples_do_not_check_themselves_with_assert() -> None:
    """``python -O`` removes assert statements, and with them an example's self-check."""
    for path in sorted((ROOT / "examples").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        found = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        assert not found, f"{path.name}: assert on lines {found}"
