"""The committed figures are exactly what their generators produce.

Every SVG under ``assets/`` that a generator writes is regenerated into a scratch
directory and compared with the checked-in file, so a hand edit, a stale export,
or a generator change that was not re-run fails here. Embedded font subsets are
compared by presence only: their bytes depend on the fontTools release.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import pytest

# The generators run in subprocesses; this module never imports fontTools itself.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fontTools") is None,
    reason="figure generation needs fonttools (the dev or docs extra)",
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATORS = ("render_diagrams.py", "render_figures.py", "render_brand.py")
FONT_DATA = re.compile(r"base64,[A-Za-z0-9+/=]+")


def _normalized(text: str) -> str:
    return FONT_DATA.sub("base64,<font>", text)


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("assets")
    for generator in GENERATORS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / generator)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
            env={**os.environ, "SILKERN_ASSETS_DIR": str(out)},
        )
        assert result.returncode == 0, f"{generator}: {result.stderr}"
    return out


def _svgs(root: pathlib.Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*.svg")}


def test_every_committed_svg_has_a_generator_and_none_is_missing(regenerated) -> None:
    assert _svgs(ROOT / "assets") == _svgs(regenerated)


def test_committed_svgs_match_a_fresh_generation(regenerated) -> None:
    stale = [
        name for name in sorted(_svgs(regenerated))
        if _normalized((ROOT / "assets" / name).read_text())
        != _normalized((regenerated / name).read_text())
    ]
    assert not stale, f"regenerate and commit: {stale}"
