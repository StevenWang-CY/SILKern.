"""The documentation must stay true: links resolve and examples run.

Every relative link, image source, and heading anchor in the Markdown files is
resolved against the checkout. Every ``python`` block in the README and guides
is executed in file order, sharing one namespace per file the way a reader
would paste them. Blocks that need CUDA run only where PyTorch with CUDA is
present; MLX blocks run wherever MLX is installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
from urllib.parse import unquote

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCUMENTS = sorted(
    [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md"]
    + list((ROOT / "docs").glob("*.md"))
    + [ROOT / "assets" / "README.md", ROOT / "evidence" / "README.md",
       ROOT / "tools" / "fonts" / "README.md"]
    + list((ROOT / "evidence").glob("*/README.md"))
)
FENCE = re.compile(r"^(```|~~~)(\w*)[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S)
MARKDOWN_LINK = re.compile(r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HTML_LINK = re.compile(r"""<(?:a|img|source)\b[^>]*?\b(?:href|src|srcset)=["']([^"']+)["']""")


def _prose(text: str) -> str:
    """Markdown with fenced code removed, so shell comments are not headings."""
    return FENCE.sub("", text)


def _slug(heading: str) -> str:
    """GitHub's heading anchor: github-slugger semantics on the rendered text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = re.sub(r"[`*]", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: pathlib.Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for line in _prose(path.read_text()).splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = _slug(match.group(1))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _links(path: pathlib.Path) -> list[str]:
    prose = _prose(path.read_text())
    found = MARKDOWN_LINK.findall(prose) + IMAGE_LINK.findall(prose) + HTML_LINK.findall(prose)
    return [link for link in found if not re.match(r"^[a-z][a-z0-9+.-]*:", link, re.I)]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(document: pathlib.Path) -> None:
    problems = []
    for link in _links(document):
        target, _, anchor = link.partition("#")
        resolved = (document.parent / unquote(target)).resolve() if target else document
        if not resolved.exists():
            problems.append(f"{link}: missing {resolved.relative_to(ROOT)}")
            continue
        if anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            problems.append(f"{link}: no heading anchor #{anchor}")
    assert not problems, "\n".join(problems)


def _python_blocks(path: pathlib.Path) -> list[str]:
    return [body for _, lang, body in FENCE.findall(path.read_text()) if lang == "python"]


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cuda_available() -> bool:
    if not _available("torch"):
        return False
    import torch

    return torch.cuda.is_available() and _available("triton")


EXAMPLE_DOCUMENTS = [path for path in DOCUMENTS if _python_blocks(path)]


@pytest.mark.parametrize("document", EXAMPLE_DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_examples_run(document: pathlib.Path) -> None:
    namespace: dict = {"__name__": "__docs__"}
    ran = 0
    for index, block in enumerate(_python_blocks(document)):
        needs_cuda = "import torch" in block or "integrations.vllm" in block
        needs_mlx = "mlx" in block or re.search(r"\bmx\.", block) is not None
        if needs_cuda and not _cuda_available():
            continue
        if needs_mlx and not _available("mlx"):
            continue
        try:
            exec(compile(block, f"{document.name}[block {index}]", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the message is the diagnosis
            pytest.fail(f"{document.relative_to(ROOT)} block {index} failed: {exc!r}\n{block}")
        ran += 1
    if ran == 0:
        pytest.skip("every example here needs an unavailable optional backend")
