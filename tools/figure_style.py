"""Typography, theme, and drawing primitives for SILKern's documentation figures.

Every figure is a self-contained SVG whose text remains selectable SVG text.
Text is set in STIX Two (SIL Open Font License 1.1; see ``tools/fonts``). Each
export embeds a WOFF subset of exactly the glyphs it uses, so GitHub renders the
same typography on every platform without fetching anything. Layout code
measures the same font files, which lets ``Figure.save`` reject overlapping
labels, labels crossed by connectors, and text outside the canvas.

Wide figures use a 1120-unit canvas, displayed at about 840 CSS pixels in a
README (0.75 px per unit). Narrow variants restack the same panels on a
420-unit canvas for phone-width pages.
"""

from __future__ import annotations

import base64
import io
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(__file__).resolve().parent / "fonts"

WIDTH = 1120
NARROW = 420

# Type scale in canvas units; at README width one unit is about 0.75 px.
CAPTION = 21  # panel subcaptions
LABEL = 21  # labels, operations, equations
VALUE = 20  # tensor entries and category labels
NOTE = 18  # headers, ticks, bar values, quiet annotations

# Stroke scale in canvas units.
HAIR = 1.0
THIN = 1.2
WIRE = 1.35
TRACE = 1.75

FONT_NOTICE = (
    " Text is set in subsets of STIX Two Text and STIX Two Math, "
    "SIL Open Font License 1.1 (tools/fonts/OFL.txt). "
)
TEXT_FAMILY = "SILKern Text"
MATH_FAMILY = "SILKern Math"
FACES = {
    "regular": ("STIXTwoText-Regular.ttf", TEXT_FAMILY, "normal", 400),
    "italic": ("STIXTwoText-Italic.ttf", TEXT_FAMILY, "italic", 400),
    "semibold": ("STIXTwoText-SemiBold.ttf", TEXT_FAMILY, "normal", 600),
    "math": ("STIXTwoMath-Regular.ttf", MATH_FAMILY, "normal", 400),
}

# Colors follow GitHub's own neutrals so figures sit naturally in either theme.
# Survivor hues derive from the identity's blue and orange threads.
LIGHT = {
    "ink": "#1f2328",
    "soft": "#3d444d",
    "mute": "#656d76",
    "faint": "#8c959f",
    "frame": "#57606a",
    "wire": "#6e7781",
    "rule": "#d0d7de",
    "grid": "#e8ebef",
    "chip": "#eff2f5",
    "zone": "#f6f8fa",
    "paper": "#ffffff",
    "blue": "#2563b0",
    "blue-w": "#e5eefa",
    "orange": "#bd5521",
    "orange-w": "#fbeadf",
    "green": "#257a58",
    "green-w": "#e1f1e9",
    "violet": "#6e52a3",
    "violet-w": "#ede8f6",
    "accent": "#2563b0",
    "compare": "#c5cbd2",
}
DARK = {
    "ink": "#e6edf3",
    "soft": "#c9d1d9",
    "mute": "#9198a1",
    "faint": "#6e7681",
    "frame": "#8b949e",
    "wire": "#7d8590",
    "rule": "#3d444d",
    "grid": "#262c34",
    "chip": "#1c222a",
    "zone": "#151b23",
    "paper": "#0d1117",
    "blue": "#6fa8ee",
    "blue-w": "#132840",
    "orange": "#ef9a63",
    "orange-w": "#3a2417",
    "green": "#5ec49a",
    "green-w": "#11301f",
    "violet": "#b49ce6",
    "violet-w": "#29203f",
    "accent": "#6fa8ee",
    "compare": "#4f5761",
}
HUES = ("blue", "orange", "green", "violet", "accent")


def _variables(palette: dict[str, str]) -> str:
    return ";".join(f"--{name}:{color}" for name, color in palette.items())


THEME_CSS = (
    f"svg{{{_variables(LIGHT)}}}@media(prefers-color-scheme:dark){{svg{{{_variables(DARK)}}}}}"
)
BASE_CSS = (
    f"text{{font-family:'{TEXT_FAMILY}','{MATH_FAMILY}',serif;fill:var(--ink);"
    "font-synthesis:none;font-kerning:normal;white-space:pre}"
    ".i{font-style:italic}.b{font-weight:600}"
    ".mute{fill:var(--mute)}.faint{fill:var(--faint)}.soft{fill:var(--soft)}"
    + "".join(f".{h}{{--c:var(--{h});--w:var(--{h}-w)}}" for h in HUES)
    + ".tint{fill:var(--c)}.wash{fill:var(--w)}"
    f".wire{{fill:none;stroke:var(--wire);stroke-width:{WIRE};"
    "stroke-linecap:round;stroke-linejoin:round}"
    ".head{fill:var(--wire)}"
    f".trace{{fill:none;stroke:var(--c);stroke-width:{TRACE};"
    "stroke-linecap:round;stroke-linejoin:round}"
    ".trace-head{fill:var(--c)}"
    f".mask{{fill:none;stroke:var(--wire);stroke-width:{WIRE};stroke-dasharray:3.5 3;"
    "stroke-linecap:butt}"
    f".frame{{fill:none;stroke:var(--frame);stroke-width:{THIN}}}"
    f".divider{{fill:none;stroke:var(--rule);stroke-width:{HAIR}}}"
    f".rule{{fill:none;stroke:var(--rule);stroke-width:{THIN}}}"
    f".grid{{fill:none;stroke:var(--grid);stroke-width:{THIN}}}"
    f".axis{{fill:none;stroke:var(--frame);stroke-width:{THIN}}}"
    ".chip{fill:var(--chip)}.zone{fill:var(--zone)}.paper{fill:var(--paper)}"
    ".compare{fill:var(--compare)}"
    f".ring{{fill:var(--paper);stroke:var(--wire);stroke-width:{WIRE}}}"
    ".span{fill:none;stroke:var(--compare);stroke-width:2.5;stroke-linecap:round}"
    ".dot{fill:var(--wire)}"
    ".frac{fill:none;stroke:var(--ink);stroke-width:1}.frac.tint{stroke:var(--c)}"
)

# Function names that remain upright inside math, as in LaTeX's \operatorname.
FUNCTIONS = ("divmod", "cumsum", "where", "count", "max", "min", "exp")
# Commands accepted inside $...$; longer names are matched first.
COMMANDS = {
    r"\,": "\u2006",
    r"\lfloor": "⌊",
    r"\rfloor": "⌋",
    r"\bmod": "mod",
    r"\ell": "ℓ",
    r"\cdots": "⋯",
    r"\cdot": "·",
    r"\delta": "δ",
    r"\sum": "∑",
    r"\sqrt": "√",
    r"\infty": "∞",
    r"\leftarrow": "←",
    r"\gets": "←",
    r"\Rightarrow": "⇒",
    r"\to": "→",
    r"\times": "×",
    r"\neq": "≠",
    r"\le": "≤",
    r"\ge": "≥",
    r"\in": "∈",
    r"\mu": "µ",
    r"\max": "max",
    r"\min": "min",
    r"\exp": "exp",
}
RELATIONS = set("=←→⇒≤≥≠<>∈")
BINARY = set("+−·×")
OPENING = set("([⌊{")
# TeX spacing: thick around relations, medium around binary operators,
# thin between an operator name or sum and its operand.
REL_SPACE, BIN_SPACE, THIN_SPACE = "\u2005", "\u2006", "\u2006"
_TOKEN = re.compile(r"\$[^$]*\$|\*\*[^*]+\*\*|[^$*]+|\*")


@cache
def _font(face: str) -> TTFont:
    return TTFont(FONT_DIR / FACES[face][0], recalcTimestamp=False, lazy=False)


@cache
def _advances(face: str) -> tuple[dict[int, float], set[int]]:
    font = _font(face)
    scale = 1 / font["head"].unitsPerEm
    metrics = font["hmtx"].metrics
    cmap = font.getBestCmap()
    return {code: metrics[name][0] * scale for code, name in cmap.items()}, set(cmap)


def _resolve(face: str, char: str) -> str:
    """Return the face a browser uses for ``char``, mirroring the CSS fallback."""
    if ord(char) in _advances(face)[1]:
        return face
    if ord(char) in _advances("math")[1]:
        return "math"
    raise ValueError(f"No embedded glyph for {char!r} (U+{ord(char):04X})")


@dataclass(frozen=True)
class Span:
    text: str
    face: str = "regular"
    scale: float = 1.0
    shift: float = 0.0  # baseline shift in em; positive moves up


def _group(source: str, i: int) -> tuple[str, int]:
    """Return a braced group or a single character starting at ``i``."""
    if source[i] != "{":
        return source[i], i + 1
    depth, k = 0, i
    while True:
        depth += {"{": 1, "}": -1}.get(source[k], 0)
        if depth == 0:
            return source[i + 1 : k], k + 1
        k += 1


def _math_spans(source: str, scale: float = 1.0, shift: float = 0.0) -> list[Span]:
    """Typeset a small LaTeX subset: italic letters, upright functions, TeX spacing."""
    for command in sorted(COMMANDS, key=len, reverse=True):
        source = source.replace(command, COMMANDS[command])
    source = source.replace("-", "−")
    atoms: list[tuple[str, list[Span]]] = []  # (kind, spans)
    i = 0
    while i < len(source):
        char = source[i]
        if char == " ":
            i += 1
        elif char in "_^" and i + 1 < len(source):
            inner, i = _group(source, i + 1)
            lift = -0.2 if char == "_" else 0.36
            nested = _math_spans(inner, scale * 0.72, shift + lift * scale)
            if atoms:
                atoms[-1] = (atoms[-1][0], atoms[-1][1] + nested)
            else:
                atoms.append(("ord", nested))
        elif char == "√" and source[i + 1 : i + 2] == "{":
            inner, i = _group(source, i + 1)
            radicand = _math_spans(inner, scale, shift)
            atoms.append(("ord", [Span("√", "regular", scale, shift), *radicand]))
        elif source.startswith("\\mathrm{", i):
            inner, i = _group(source, i + 7)
            atoms.append(("ord", [Span(inner, "regular", scale, shift)]))
        elif source.startswith("mod", i) and not (i and source[i - 1].isalpha()):
            atoms.append(("bin-word", [Span("mod", "regular", scale, shift)]))
            i += 3
        elif word := next((w for w in FUNCTIONS if source.startswith(w, i)), None):
            atoms.append(("op", [Span(word, "regular", scale, shift)]))
            i += len(word)
        else:
            italic = char.isalpha() and char not in "∑"
            kind = (
                "rel"
                if char in RELATIONS
                else "bin"
                if char in BINARY
                else "punct"
                if char == ","
                else "open"
                if char in OPENING
                else "op"
                if char == "∑"
                else "ord"
            )
            atoms.append((kind, [Span(char, "italic" if italic else "regular", scale, shift)]))
            i += 1
    spans: list[Span] = []
    previous = None
    for k, (kind, parts) in enumerate(atoms):
        following = atoms[k + 1][0] if k + 1 < len(atoms) else None
        if kind == "bin" and previous in (None, "rel", "bin", "open", "punct", "bin-word", "op"):
            kind = "ord"  # a unary sign
        space = {"rel": REL_SPACE, "bin": BIN_SPACE, "bin-word": REL_SPACE}.get(kind)
        if space and previous is not None:
            spans.append(Span(space, "regular", scale, shift))
        elif kind == "op" and previous == "ord":
            spans.append(Span(THIN_SPACE, "regular", scale, shift))
        spans += parts
        if space and following is not None:
            spans.append(Span(space, "regular", scale, shift))
        elif kind == "op" and following == "ord":
            spans.append(Span(THIN_SPACE, "regular", scale, shift))
        elif kind == "punct" and following is not None:
            spans.append(Span(THIN_SPACE, "regular", scale, shift))
        previous = kind
    return spans


def parse(markup: str, weight: str = "regular") -> list[Span]:
    """Parse ``$math$`` and ``**semibold**`` markup into measured spans."""
    spans: list[Span] = []
    for token in _TOKEN.findall(str(markup)):
        if token.startswith("$") and token.endswith("$") and len(token) > 1:
            spans += _math_spans(token[1:-1])
        elif token.startswith("**") and token.endswith("**") and len(token) > 4:
            spans.append(Span(token[2:-2], "semibold"))
        else:
            spans.append(Span(token, weight))
    merged: list[Span] = []
    for span in spans:
        last = merged[-1] if merged else None
        if last and (last.face, last.scale, last.shift) == (span.face, span.scale, span.shift):
            merged[-1] = Span(last.text + span.text, span.face, span.scale, span.shift)
        else:
            merged.append(span)
    return merged


def measure(markup: str, size: float, weight: str = "regular") -> float:
    """Advance width of ``markup`` at ``size``, from the embedded fonts' metrics."""
    width = 0.0
    for span in parse(markup, weight):
        for char in span.text:
            face = _resolve(span.face, char)
            width += _advances(face)[0][ord(char)] * size * span.scale
    return width


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str

    def overlaps(self, other: Box, pad: float = 0.0) -> bool:
        return not (
            self.x1 + pad <= other.x0
            or other.x1 + pad <= self.x0
            or self.y1 + pad <= other.y0
            or other.y1 + pad <= self.y0
        )


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _segment_hits_box(a, b, box: Box, pad: float) -> bool:
    """Liang-Barsky clip of segment ab against a padded box."""
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
            if lo > hi:
                return False
    return True


def _cubic(p0, p1, p2, p3, steps=24):
    points = []
    for k in range(steps + 1):
        t = k / steps
        u = 1 - t
        points.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return points


class Figure:
    """An SVG canvas that records text and connector geometry for clearance checks."""

    def __init__(self, width: int, height: int, title: str, description: str):
        self.width, self.height = width, height
        self.title, self.description = title, description
        self.items: list[str] = []
        self.boxes: list[Box] = []
        self.wires: list[tuple[list[tuple[float, float]], tuple[str, ...]]] = []
        self.glyphs: dict[str, set[str]] = {face: set() for face in FACES}

    # Text -----------------------------------------------------------------

    def text(
        self,
        x: float,
        y: float,
        markup,
        *,
        size: float = LABEL,
        anchor: str = "start",
        cls: str = "",
        weight: str = "regular",
        collide: bool = True,
        name: str | None = None,
    ) -> Box:
        spans = parse(str(markup), weight)
        width = measure(str(markup), size, weight)
        parts, level = [], 0.0
        for span in spans:
            for char in span.text:
                self.glyphs[_resolve(span.face, char)].add(char)
            attrs = []
            if span.face == "italic":
                attrs.append('class="i"')
            elif span.face == "semibold":
                attrs.append('class="b"')
            if span.scale != 1:
                attrs.append(f'font-size="{_fmt(size * span.scale)}"')
            if span.shift != level:
                attrs.append(f'dy="{_fmt((level - span.shift) * size)}"')
                level = span.shift
            parts.append(
                f"<tspan {' '.join(attrs)}>{escape(span.text)}</tspan>"
                if attrs
                else escape(span.text)
            )
        anchor_attr = "" if anchor == "start" else f' text-anchor="{anchor}"'
        cls_attr = f' class="{cls}"' if cls else ""
        size_attr = "" if size == LABEL else f' font-size="{_fmt(size)}"'
        self.items.append(
            f'<text x="{_fmt(x)}" y="{_fmt(y)}"{anchor_attr}{cls_attr}{size_attr}>'
            + "".join(parts)
            + "</text>"
        )
        left = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
        low = min((s.shift for s in spans), default=0)
        high = max((s.shift + 0.7 * s.scale for s in spans), default=0.7)
        descends = any(c in "gjpqyQ(),;[]{}/|ℓ" for s in spans for c in s.text)
        box = Box(
            left,
            y - max(high, 0.68) * size,
            left + width,
            y + (0.24 if descends else 0.04) * size - low * size,
            name or str(markup),
        )
        if collide:
            self.boxes.append(box)
        return box

    # Geometry -------------------------------------------------------------

    def raw(self, svg: str) -> None:
        self.items.append(svg)

    def rect(self, x, y, w, h, cls, rx: float = 0) -> None:
        radius = f' rx="{_fmt(rx)}"' if rx else ""
        self.items.append(
            f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}"'
            f'{radius} class="{cls}"/>'
        )

    def circle(self, x, y, r, cls) -> None:
        self.items.append(f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(r)}" class="{cls}"/>')

    def path(self, d: str, cls: str) -> None:
        self.items.append(f'<path d="{d}" class="{cls}"/>')

    def _head(self, tip, direction, cls, scale=1.0) -> None:
        dx, dy = direction
        norm = math.hypot(dx, dy) or 1
        ux, uy = dx / norm, dy / norm
        nx, ny = -uy, ux
        length, half, notch = 7.2 * scale, 2.9 * scale, 4.6 * scale
        x, y = tip
        pts = [
            (x, y),
            (x - ux * length + nx * half, y - uy * length + ny * half),
            (x - ux * notch, y - uy * notch),
            (x - ux * length - nx * half, y - uy * length - ny * half),
        ]
        self.items.append(
            '<path d="M'
            + "L".join(f"{_fmt(px)} {_fmt(py)}" for px, py in pts)
            + f'Z" class="{cls}"/>'
        )

    def line(
        self,
        points,
        *,
        cls: str = "wire",
        arrow: bool = False,
        radius: float = 0,
        collide: bool = True,
        ignore: tuple[str, ...] = (),
    ) -> None:
        """A polyline with optional rounded corners and a dart arrowhead."""
        points = [tuple(map(float, p)) for p in points]
        trace = "trace" in cls.split()
        head_cls = (
            cls.replace("trace", "trace-head")
            if trace
            else cls.replace("wire", "head").replace("mask", "head")
        )
        draw = list(points)
        if arrow:
            (ax, ay), (bx, by) = draw[-2], draw[-1]
            seg = math.hypot(bx - ax, by - ay)
            back = min(4.0, seg * 0.5)
            draw[-1] = (bx - (bx - ax) / seg * back, by - (by - ay) / seg * back)
        d = f"M{_fmt(draw[0][0])} {_fmt(draw[0][1])}"
        for k in range(1, len(draw)):
            if radius and 0 < k < len(draw) - 1:
                (px, py), (cx, cy), (nx, ny) = draw[k - 1], draw[k], draw[k + 1]
                l1, l2 = math.hypot(cx - px, cy - py), math.hypot(nx - cx, ny - cy)
                r = min(radius, l1 / 2, l2 / 2)
                ax, ay = cx - (cx - px) / l1 * r, cy - (cy - py) / l1 * r
                bx, by = cx + (nx - cx) / l2 * r, cy + (ny - cy) / l2 * r
                d += f"L{_fmt(ax)} {_fmt(ay)}Q{_fmt(cx)} {_fmt(cy)} {_fmt(bx)} {_fmt(by)}"
            else:
                d += f"L{_fmt(draw[k][0])} {_fmt(draw[k][1])}"
        self.path(d, cls)
        if arrow:
            (ax, ay), (bx, by) = points[-2], points[-1]
            self._head((bx, by), (bx - ax, by - ay), head_cls, 1.08 if trace else 1.0)
        if collide:
            self.wires.append((points, ignore))

    def curve(
        self,
        start,
        end,
        *,
        cls: str = "wire",
        arrow: bool = True,
        bend: float = 0.5,
        horizontal: bool = False,
        collide: bool = True,
        ignore: tuple[str, ...] = (),
    ) -> None:
        """A cubic S-curve between two ports, leaving and entering on the axis."""
        (x0, y0), (x1, y1) = start, end
        if horizontal:
            c1, c2 = (x0 + (x1 - x0) * bend, y0), (x1 - (x1 - x0) * bend, y1)
        else:
            c1, c2 = (x0, y0 + (y1 - y0) * bend), (x1, y1 - (y1 - y0) * bend)
        trace = "trace" in cls.split()
        tip = (x1, y1)
        if arrow:
            dx, dy = x1 - c2[0], y1 - c2[1]
            norm = math.hypot(dx, dy) or 1
            end_draw = (x1 - dx / norm * 4.0, y1 - dy / norm * 4.0)
        else:
            end_draw = tip
        d = (
            f"M{_fmt(x0)} {_fmt(y0)}C{_fmt(c1[0])} {_fmt(c1[1])} "
            f"{_fmt(c2[0])} {_fmt(c2[1])} {_fmt(end_draw[0])} {_fmt(end_draw[1])}"
        )
        self.path(d, cls)
        if arrow:
            head = cls.replace("trace", "trace-head") if trace else cls.replace("wire", "head")
            self._head(tip, (x1 - c2[0], y1 - c2[1]), head, 1.08 if trace else 1.0)
        if collide:
            self.wires.append((_cubic((x0, y0), c1, c2, (x1, y1)), ignore))

    # Composite marks --------------------------------------------------------

    def strip(
        self,
        x: float,
        y: float,
        values,
        *,
        cell: float,
        height: float = 30,
        hues: dict[int, str] | None = None,
        size: float = VALUE,
        quiet: tuple[int, ...] = (),
        bold: bool = False,
        group: int = 0,
        open_end: bool = False,
        name: str = "strip",
    ) -> list[float]:
        """A continuous tensor strip; tinted cells carry the survivor hue.

        ``group`` draws a frame-weight divider every ``group`` cells, for tiles.
        ``open_end`` leaves the right edge open and marks the strip as continuing.
        """
        hues = hues or {}
        n = len(values)
        for j, hue in hues.items():
            self.rect(x + j * cell, y, cell, height, f"{hue} wash")
        for j in range(1, n):
            major = group and j % group == 0
            self.path(
                f"M{_fmt(x + j * cell)} {_fmt(y)}v{_fmt(height)}", "frame" if major else "divider"
            )
        if open_end:
            right = x + n * cell
            self.path(
                f"M{_fmt(right)} {_fmt(y)}H{_fmt(x)}V{_fmt(y + height)}H{_fmt(right)}", "frame"
            )
            for k in (-1, 0, 1):
                self.circle(right + 7 + 5 * k, y + height / 2, 1.1, "dot")
        else:
            self.rect(x, y, n * cell, height, "frame")
        centers = []
        for j, value in enumerate(values):
            cx = x + (j + 0.5) * cell
            centers.append(cx)
            hue = hues.get(j)
            cls = f"{hue} tint" if hue else ("faint" if j in quiet else "mute")
            weight = "semibold" if (bold and hue) else "regular"
            self.text(
                cx,
                y + height / 2 + size * 0.34,
                value,
                size=size,
                anchor="middle",
                cls=cls,
                weight=weight,
                name=f"{name}[{j}]",
            )
        return centers

    def column(
        self,
        x: float,
        y: float,
        values,
        *,
        width: float = 40,
        cell: float = 40,
        hues: dict[int, str] | None = None,
        size: float = VALUE,
        name: str = "column",
    ) -> list[float]:
        """A vertical tensor strip, for tables indexed down the page."""
        hues = hues or {}
        for k, hue in hues.items():
            self.rect(x, y + k * cell, width, cell, f"{hue} wash")
        for k in range(1, len(values)):
            self.path(f"M{_fmt(x)} {_fmt(y + k * cell)}h{_fmt(width)}", "divider")
        self.rect(x, y, width, len(values) * cell, "frame")
        centers = []
        for k, value in enumerate(values):
            cy = y + (k + 0.5) * cell
            centers.append(cy)
            hue = hues.get(k)
            self.text(
                x + width / 2,
                cy + size * 0.34,
                value,
                size=size,
                anchor="middle",
                cls=f"{hue} tint" if hue else "mute",
                name=f"{name}[{k}]",
            )
        return centers

    def band(
        self, x: float, y: float, width: float, markup, *, height: float = 26, size: float = LABEL
    ) -> None:
        """An operation applied across a strip; connectors pass beneath it."""
        self.rect(x, y, width, height, "chip", rx=3)
        self.text(x + width / 2, y + height / 2 + size * 0.33, markup, size=size, anchor="middle")

    def brace(self, x0, x1, y, *, depth: float = 7, cls: str = "wire", up: bool = False) -> None:
        """A horizontal curly brace whose point faces down (or up)."""
        q = depth / 2 * (-1 if up else 1)
        xm = (x0 + x1) / 2
        d = (
            f"M{_fmt(x0)} {_fmt(y)}q0 {_fmt(q)} {_fmt(abs(q))} {_fmt(q)}"
            f"H{_fmt(xm - abs(q))}q{_fmt(abs(q))} 0 {_fmt(abs(q))} {_fmt(q)}"
            f"q0 {_fmt(-q)} {_fmt(abs(q))} {_fmt(-q)}H{_fmt(x1 - abs(q))}"
            f"q{_fmt(abs(q))} 0 {_fmt(abs(q))} {_fmt(-q)}"
        )
        self.path(d, cls)

    def equations(self, x_eq: float, y: float, pitch: float, rows, *, size: float = LABEL) -> None:
        """Display equations aligned on their relation sign."""
        for k, (lhs, rhs) in enumerate(rows):
            self.text(
                x_eq - measure(f"${lhs}$", size), y + k * pitch, f"${lhs} = {rhs}$", size=size
            )

    def rule(self, x0: float, x1: float, y: float, *, weight: float = HAIR, cls: str = "axis"):
        self.items.append(
            f'<path d="M{_fmt(x0)} {_fmt(y)}H{_fmt(x1)}" class="{cls}" '
            f'style="stroke-width:{_fmt(weight)}"/>'
        )

    def fraction(
        self, x: float, y: float, top, bottom, *, size: float = LABEL, cls: str = ""
    ) -> float:
        """A built-up fraction whose bar sits on the math axis at baseline ``y``."""
        width = max(measure(top, size), measure(bottom, size)) + 6
        axis = y - 0.25 * size
        self.text(x + width / 2, axis - 0.28 * size, top, size=size, anchor="middle", cls=cls)
        self.text(x + width / 2, axis + 0.95 * size, bottom, size=size, anchor="middle", cls=cls)
        self.items.append(f'<path d="M{_fmt(x)} {_fmt(axis)}h{_fmt(width)}" class="frac {cls}"/>')
        return width

    def subcaption(self, cx: float, y: float, letter: str, title: str) -> None:
        self.text(cx, y, f"({letter}) {title}", size=CAPTION, anchor="middle")

    # Export ------------------------------------------------------------------

    def check(self) -> None:
        """Reject text outside the canvas, overlapping text, and connectors across text."""
        for k, box in enumerate(self.boxes):
            if box.x0 < 1 or box.y0 < 1 or box.x1 > self.width - 1 or box.y1 > self.height - 1:
                raise ValueError(f"Text leaves the canvas: {box.label!r} {box}")
            for other in self.boxes[k + 1 :]:
                if box.overlaps(other, pad=1.0):
                    raise ValueError(f"Overlapping text: {box.label!r} and {other.label!r}")
        for points, ignore in self.wires:
            for box in self.boxes:
                if box.label in ignore:
                    continue
                for a, b in zip(points, points[1:], strict=False):
                    if _segment_hits_box(a, b, box, pad=1.5):
                        raise ValueError(f"Connector crosses text {box.label!r} near {a}")

    def _font_css(self) -> str:
        rules = []
        for face, chars in self.glyphs.items():
            if not chars:
                continue
            file, family, style, weight = FACES[face]
            options = subset.Options()
            options.flavor = "woff"
            options.layout_features = ["kern", "liga"]
            options.name_IDs = [0, 1, 2, 13, 14]
            options.notdef_outline = True
            options.hinting = False
            options.recalc_timestamp = False
            font = TTFont(FONT_DIR / file, recalcTimestamp=False)
            subsetter = subset.Subsetter(options)
            subsetter.populate(text="".join(sorted(chars)))
            subsetter.subset(font)
            buffer = io.BytesIO()
            font.flavor = "woff"
            font.save(buffer)
            data = base64.b64encode(buffer.getvalue()).decode()
            rules.append(
                f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
                f"src:url(data:font/woff;base64,{data}) format('woff')}}"
            )
        return "".join(rules)

    def render(self) -> str:
        css = self._font_css() + THEME_CSS + BASE_CSS
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" '
                'aria-labelledby="title desc">',
                f'<title id="title">{escape(self.title)}</title>',
                f'<desc id="desc">{escape(self.description)}</desc>',
                f"<!--{FONT_NOTICE}-->",
                f"<style>{css}</style>",
                *self.items,
                "</svg>",
                "",
            ]
        )

    def save(self, name: str, *, panels=(), gap: int = 28, narrow_width: int = NARROW) -> None:
        self.check()
        path = ROOT / "assets" / name
        path.write_text(self.render())
        if panels:
            stack_panels(path, panels, gap=gap, width=narrow_width)


def stack_panels(path: Path, panels, *, gap: int, width: int) -> None:
    """Restack wide panels in one column, reusing the exact vector content."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    source = ET.parse(path).getroot()
    panels = [(*panel, gap) if len(panel) == 4 else tuple(panel) for panel in panels]
    height = sum(panel[3] + panel[4] for panel in panels)
    root = ET.Element(
        f"{{{ns}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-labelledby": "title desc",
        },
    )
    defs = ET.Element(f"{{{ns}}}defs")
    content = ET.SubElement(defs, f"{{{ns}}}g", {"id": "figure-content"})
    for child in source:
        local = child.tag.rsplit("}", 1)[-1]
        if local == "style":
            root.append(ET.Comment(FONT_NOTICE))
        (root if local in {"title", "desc", "style"} else content).append(deepcopy(child))
    root.append(defs)
    y = 0
    for left, top, panel_width, panel_height, after in panels:
        scale = min(1.0, width / panel_width)
        shown_w, shown_h = panel_width * scale, panel_height * scale
        panel = ET.SubElement(
            root,
            f"{{{ns}}}svg",
            {
                "x": _fmt((width - shown_w) / 2),
                "y": _fmt(y),
                "width": _fmt(shown_w),
                "height": _fmt(shown_h),
                "viewBox": f"{_fmt(left)} {_fmt(top)} {_fmt(panel_width)} {_fmt(panel_height)}",
                "overflow": "hidden",
            },
        )
        ET.SubElement(panel, f"{{{ns}}}use", {"href": "#figure-content"})
        y += shown_h + after
    y -= panels[-1][4]
    root.set("height", _fmt(y))
    root.set("viewBox", f"0 0 {width} {_fmt(y)}")
    text = ET.tostring(root, encoding="unicode")
    path.with_name(path.name.replace(".svg", "-narrow.svg")).write_text(text + "\n")
