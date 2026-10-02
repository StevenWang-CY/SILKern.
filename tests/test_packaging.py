"""Checks on the distribution itself, all runnable without a GPU.

These exist because the failure they catch is invisible to the rest of the
suite: the GPU modules are skipped wherever CUDA is absent, so a name that goes
stale in one of them survives every CPU run and only explodes on the machine
that matters.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tomllib
from importlib import import_module
from urllib.parse import urlsplit

import pytest

import silkern

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("tests", "bench", "examples")


def _silkern_imports(path: pathlib.Path) -> list[tuple[str, str]]:
    """Every ``from silkern[.x] import name`` in one file, as (module, name)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module == "silkern" or module.startswith("silkern."):
                found.extend((module, alias.name) for alias in node.names)
    return found


def test_every_name_the_suite_imports_from_silkern_still_exists() -> None:
    """Catch a renamed export before it reaches a machine that can run it.

    The GPU suites are skipped without CUDA, so a dangling ``from silkern
    import ...`` in one of them passes CI forever and then fails at collection
    on the one box that could have verified the kernels. This resolves those
    names on the CPU, where every module in the package imports fine.
    """
    checked = 0
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            for module, name in _silkern_imports(path):
                # Match ``from package import submodule`` semantics too. A
                # submodule need not already be bound on its package; relying
                # on that makes this test depend on earlier test collection.
                imported = __import__(module, fromlist=[name])
                assert hasattr(imported, name), (
                    f"{path.relative_to(ROOT)} imports {name!r} from {module}, "
                    f"which no longer defines it"
                )
                checked += 1
    assert checked > 0, "the import scan found nothing; the walk is broken"


def test_public_surface_is_declared_and_resolvable() -> None:
    assert len(silkern.__all__) == len(set(silkern.__all__)), "duplicate in __all__"
    for name in silkern.__all__:
        assert hasattr(silkern, name), f"__all__ names {name!r}, which is not exported"


@pytest.mark.parametrize(
    "name",
    ["silkern.contract", "silkern.errors", "silkern.kernels", "silkern.verify",
     "silkern.workspace", "silkern.integrations", "silkern.integrations.vllm",
     "silkern.mlx", "silkern.mlx_verify"],
)
def test_every_module_imports_without_a_gpu_stack(name: str) -> None:
    """All modules import when optional runtimes are absent, including guarded Triton."""
    import_module(name)


def test_version_is_the_same_in_every_place_that_states_it() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text()
    cited = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', citation, re.MULTILINE)
    assert cited is not None, "CITATION.cff has no version field"
    assert silkern.__version__ == declared == cited.group(1).strip()


def test_typing_marker_ships_with_the_package() -> None:
    """PEP 561: without this file the annotations are invisible to consumers."""
    assert (pathlib.Path(silkern.__file__).parent / "py.typed").is_file()


def test_distribution_description_links_work_without_the_checkout() -> None:
    """Index descriptions cannot resolve repository-relative files or images."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    description = (ROOT / pyproject["project"]["readme"]).read_text()
    targets = re.findall(r"\]\(([^\s)]+)\)", description)
    targets += re.findall(r'''\b(?:src|href)=["']([^"']+)["']''', description)
    assert targets, "the package description should link to its complete documentation"
    assert all(urlsplit(target).scheme == "https" for target in targets), targets


def test_base_import_and_oracle_do_not_load_optional_backends() -> None:
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'torch', 'triton'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
import silkern
assert silkern.localize_reference([0], [[3]], [[0, 1]],
    block_size=2, dcp_size=2, dcp_rank=0) == ([[6, -1]], [1])
assert not any(name.split('.')[0] in ('mlx', 'torch', 'triton') for name in sys.modules)
try:
    silkern.localize_mlx(None, None, None, block_size=2, dcp_size=2, dcp_rank=0)
except silkern.LocalizationError as exc:
    assert 'silkern[mlx]' in str(exc)
else:
    raise AssertionError('missing MLX must raise an actionable error')
"""],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
