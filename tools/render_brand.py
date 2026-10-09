"""Render SILKern's brand assets and README icons as self-contained SVGs.

The mark is three index threads that enter spread out and leave packed against
a rail: scattered selections become dense slots. The blue thread passes over the
ink thread and under the orange one, so the threads read as woven rather than
merely crossing; the gaps are masks, so they work on any background. The
wordmark is Manrope (SIL Open Font License 1.1, ``tools/fonts``) converted to
outlines, so no font is fetched or embedded. The icons reuse the figure glyphs
of ``figure_style``, so the README and the figures share one line grammar.

Run from the repository root:

    python tools/render_brand.py
"""

from __future__ import annotations

from figure_style import (
    ASSETS,
    GLYPHS,
    NOTE,
    ROOT,
    VALUE,
    Figure,
    _fmt,
    glyph_element,
    measure,
)
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

from silkern import localize_reference

FONT = ROOT / "tools" / "fonts" / "Manrope-SILKern.ttf"

# Brand hues: the figures' blue and orange, lifted for a mark seen at icon size.
LIGHT = {"ink": "#1f2328", "blue": "#2a6fd1", "orange": "#de6129", "paper": "#ffffff"}
DARK = {"ink": "#e6edf3", "blue": "#6aa6f0", "orange": "#f09a5f", "paper": "#0d1117"}

# The mark on a 64-unit grid: threads run from x 6 to 50, the rail stands at 53.5.
SPREAD = {"orange": 14, "ink": 32, "blue": 50}  # entry heights, spread out
PACKED = {"blue": 20, "orange": 32, "ink": 44}  # exit heights, packed
STROKE, GAP = 5.5, 2.2
RAIL = (53.5, 14, 5.0, 36)  # x, y, width, height


def _css(palette: dict[str, str]) -> str:
    return (
        f".brand-ink{{fill:{palette['ink']}}}.thread-ink{{stroke:{palette['ink']}}}"
        f".thread-blue{{stroke:{palette['blue']}}}.thread-orange{{stroke:{palette['orange']}}}"
        f".brand-paper{{fill:{palette['paper']}}}"
    )


def _icon_css(palette: dict[str, str]) -> str:
    return f".icon{{stroke:{palette['ink']}}}.icon .solid{{fill:{palette['ink']}}}"


STYLE = (
    "<style>.thread-ink,.thread-blue,.thread-orange{fill:none;stroke-linecap:round}"
    f"{_css(LIGHT)}@media (prefers-color-scheme:dark){{{_css(DARK)}}}</style>"
)
ICON_STYLE = (
    "<style>.icon{fill:none;stroke-linecap:round;stroke-linejoin:round}"
    ".icon .solid{stroke:none}"
    f"{_icon_css(LIGHT)}@media (prefers-color-scheme:dark){{{_icon_css(DARK)}}}</style>"
)
FONT_NOTICE = (
    "<!-- Wordmark outlines from Manrope, SIL Open Font License 1.1 "
    "(tools/fonts/Manrope-OFL.txt). -->"
)


def _thread(name: str) -> str:
    y0, y1 = SPREAD[name], PACKED[name]
    return f"M6 {y0}H14C28 {y0} 28 {y1} 42 {y1}H50"


def mark(prefix: str) -> str:
    """The woven mark's elements in its 64-unit box; ``prefix`` keeps mask ids unique."""
    cut = _fmt(STROKE + 2 * GAP)

    def mask(name: str, over: str) -> str:
        return (
            f'<mask id="{prefix}-{name}" maskUnits="userSpaceOnUse" x="0" y="0" width="64" '
            f'height="64"><rect width="64" height="64" fill="#fff"/><path d="{_thread(over)}" '
            f'fill="none" stroke="#000" stroke-width="{cut}"/></mask>'
        )

    x, y, w, h = RAIL
    return (
        f"<defs>{mask('under-blue', 'blue')}{mask('under-orange', 'orange')}</defs>"
        f'<path class="thread-ink" d="{_thread("ink")}" stroke-width="{STROKE}" '
        f'mask="url(#{prefix}-under-blue)"/>'
        f'<path class="thread-blue" d="{_thread("blue")}" stroke-width="{STROKE}" '
        f'mask="url(#{prefix}-under-orange)"/>'
        f'<path class="thread-orange" d="{_thread("orange")}" stroke-width="{STROKE}"/>'
        f'<rect class="brand-ink" x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
        f'rx="{_fmt(w / 2)}"/>'
    )


_INSTANCES: dict[int, TTFont] = {}


def _instance(weight: int) -> TTFont:
    if weight not in _INSTANCES:
        source = TTFont(FONT, recalcTimestamp=False)
        _INSTANCES[weight] = instancer.instantiateVariableFont(source, {"wght": weight})
    return _INSTANCES[weight]


def _kern(font: TTFont, left: str, right: str) -> int:
    """The first pair adjustment GPOS gives for two glyphs (class or glyph pairs)."""
    for lookup in font["GPOS"].table.LookupList.Lookup:
        for table in lookup.SubTable:
            table = getattr(table, "ExtSubTable", table)
            if table.__class__.__name__ != "PairPos" or left not in table.Coverage.glyphs:
                continue
            if table.Format == 1:
                pairs = table.PairSet[table.Coverage.glyphs.index(left)].PairValueRecord
                for record in pairs:
                    if record.SecondGlyph == right:
                        return getattr(record.Value1, "XAdvance", 0) or 0
            else:
                first = table.ClassDef1.classDefs.get(left, 0)
                second = table.ClassDef2.classDefs.get(right, 0)
                value = table.Class1Record[first].Class2Record[second].Value1
                if advance := getattr(value, "XAdvance", 0) or 0:
                    return advance
    return 0


def outline(text: str, weight: int, size: float, x: float, y: float) -> tuple[str, float]:
    """Kerned Manrope outlines for ``text`` with its baseline at ``y``; returns (d, advance)."""
    font = _instance(weight)
    names = [font.getBestCmap()[ord(char)] for char in text]
    glyphs, metrics = font.getGlyphSet(), font["hmtx"].metrics
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs, ntos=_fmt)
    advance = 0.0
    for k, name in enumerate(names):
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + advance, y)))
        step = metrics[name][0] + (_kern(font, name, names[k + 1]) if k + 1 < len(names) else 0)
        advance += step * scale
    return pen.getCommands(), advance


def cap_height(weight: int) -> float:
    font = _instance(weight)
    return font["OS/2"].sCapHeight / font["head"].unitsPerEm


def lockup(prefix: str, size: float, x: float, baseline: float) -> tuple[str, float]:
    """Mark and wordmark, "SIL" heavy and "Kern" medium; returns (svg, right edge)."""
    cap = cap_height(800) * size
    scale = cap * 1.25 / RAIL[3]  # the rail spans 1.25 cap heights
    top = baseline - cap / 2 - (RAIL[1] + RAIL[3] / 2) * scale
    left = x - 3.25 * scale  # the threads' round caps start at the box's x 3.25
    parts = [
        f'<g transform="translate({_fmt(left)} {_fmt(top)}) scale({scale:.5f})">{mark(prefix)}</g>'
    ]
    pen_x = left + (RAIL[0] + RAIL[2]) * scale + size * 0.30
    for text, weight in (("SIL", 800), ("Kern", 500)):
        d, advance = outline(text, weight, size, pen_x, baseline)
        parts.append(f'<path class="brand-ink" d="{d}"/>')
        pen_x += advance
    return "".join(parts), pen_x


def _svg(width: float, height: float, label: str, body: str, *, head: str = FONT_NOTICE + STYLE):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" height="{_fmt(height)}" '
        f'viewBox="0 0 {_fmt(width)} {_fmt(height)}" role="img" aria-label="{label}">'
        f"{head}{body}</svg>\n"
    )


def render_logo() -> None:
    size, pad = 64, 4
    cap = cap_height(800) * size
    scale = cap * 1.25 / RAIL[3]
    # Half the mark's height around the cap centre: rail centre to a thread's outer edge.
    half = (RAIL[1] + RAIL[3] / 2 - (SPREAD["orange"] - STROKE / 2)) * scale
    baseline = pad + half + cap / 2
    body, right = lockup("logo-text", size, pad, baseline)
    height = 2 * (pad + half)
    paper = f'<rect class="brand-paper" width="{_fmt(right + pad)}" height="{_fmt(height)}"/>'
    (ASSETS / "logo-text.svg").write_text(
        _svg(right + pad, height, "SILKern: woven index threads beside the wordmark",
             paper + body)
    )
    (ASSETS / "logo.svg").write_text(
        _svg(64, 64, "SILKern mark: three woven index threads packed against a rail",
             mark("logo"))
    )


def render_icons() -> None:
    """README icons: the figure glyphs alone on their 16-unit canvas, in ink."""
    (ASSETS / "icons").mkdir(exist_ok=True)
    labels = {
        "cpu": "Processor",
        "soc": "System on a chip",
        "gpu": "Accelerator module",
    }
    for name, label in labels.items():
        parts = "".join(glyph_element(element, attrs) for element, attrs in GLYPHS[name])
        body = f'<g class="icon" stroke-width="1.5">{parts}</g>'
        (ASSETS / "icons" / f"{name}.svg").write_text(_svg(16, 16, label, body, head=ICON_STYLE))


def render_banner() -> None:
    """The 1280 x 640 social card; ``cover.png`` is its raster."""
    s = Figure(1280, 640, "SILKern", "")
    s.description = (
        "SILKern social card: the woven mark and wordmark; the line 'Deterministic "
        "sparse-index localization for context-parallel attention'; a selection row "
        "[8, 5, 130, 7, -1, 262] becoming rank 0's slots [708, 129, 451, -1, -1, -1]; and "
        "the three platforms: the Python reference, Apple silicon with MLX and Metal, and "
        "NVIDIA GPUs with Triton."
    )
    body, right = lockup("banner", 112, 0, 0)
    width = right
    x0 = (1280 - width) / 2
    s.raw(STYLE + f'<g transform="translate({_fmt(x0)} 214)">{body}</g>')
    s.text(640, 290, "Deterministic sparse-index localization for context-parallel attention",
           size=34, anchor="middle")
    row = [8, 5, 130, 7, "−1", 262]
    slots, counts = localize_reference([0], [[11, 2, 7, 5]], [[8, 5, 130, 7, -1, 262]],
                                       block_size=64, dcp_size=2, dcp_rank=0)
    assert slots == [[708, 129, 451, -1, -1, -1]] and counts == [3]
    cell, y = 58, 360
    left_x, right_x = 640 - 52 - 6 * cell, 640 + 52
    s.strip(left_x, y, row, cell=cell, height=40, size=VALUE + 4,
            hues={0: "blue", 1: "gray", 2: "orange", 3: "gray", 5: "green"}, quiet=(4,))
    s.strip(right_x, y, [708, 129, 451, "−1", "−1", "−1"], cell=cell, height=40, size=VALUE + 4,
            hues={0: "blue", 1: "orange", 2: "green"}, bold=True, quiet=(3, 4, 5))
    s.line([(640 - 34, y + 20), (640 + 34, y + 20)], cls="flow", arrow=True)
    s.text(left_x + 3 * cell, y + 70, "selected positions", size=NOTE + 4, anchor="middle",
           cls="soft")
    s.text(right_x + 3 * cell, y + 70, "rank 0's KV-cache slots", size=NOTE + 4,
           anchor="middle", cls="soft")
    s.rule(64, 1216, 532, cls="rule")
    items = (
        ("cpu", "Python reference"),
        ("soc", "Apple silicon · MLX and Metal"),
        ("gpu", "NVIDIA GPUs · Triton"),
    )
    gap, size = 64, 26
    widths = [38 + measure(label, size) for _, label in items]
    x = (1280 - sum(widths) - gap * (len(items) - 1)) / 2
    for (glyph, label), width in zip(items, widths, strict=True):
        s.glyph(glyph, x, 559, 28)
        s.text(x + 38, 581, label, size=size)
        x += width + gap
    s.save("banner.svg")


def main() -> None:
    render_logo()
    render_icons()
    render_banner()


if __name__ == "__main__":
    main()
