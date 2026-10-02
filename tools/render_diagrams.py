"""Render the explanatory SVGs without raster assets or hardware execution.

The worked examples are checked against the pure-Python contract before export.
Semantic colors follow survivor identity; neutral canvas and adaptive ink keep
the figures legible when embedded in light or dark Markdown pages.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from silkern import localize_reference

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("blue", "teal", "copper", "violet")
STYLE = """text{font-family:Arial,Helvetica,sans-serif;font-size:16px;font-variant-numeric:tabular-nums;fill:var(--ink)}
svg{--ink:#202832;--muted:#58616c;--line:#84909b;--rule:#d4dbe1;--paper:#fff;--neutral:#f1f3f5;--blue:#32669b;--blue-wash:#e8f0f8;--teal:#287c72;--teal-wash:#e7f2ef;--copper:#a16a3b;--copper-wash:#f7eddf;--violet:#786a9e;--violet-wash:#eeebf5}
.heading{font-size:19px;font-weight:600}.subhead{font-size:16px;font-weight:600}
.small{font-size:13px}.label{font-size:14px}.muted{fill:var(--muted)}
.mono{font-family:Menlo,Consolas,monospace;font-size:15px}
.mono.small{font-size:13px}.data{font-size:16px;font-weight:500}.data.knockout{font-weight:600}
.math{font-family:Georgia,'Times New Roman',serif;font-size:17px;font-style:italic}
.rule{fill:none;stroke:var(--rule);stroke-width:1}
.edge{fill:none;stroke:var(--line);stroke-width:1.25;stroke-linejoin:round;stroke-linecap:round}
.outline{fill:var(--paper);stroke:var(--line);stroke-width:1.2}
.node{fill:var(--neutral);stroke:var(--line);stroke-width:1}
.grid-cell{fill:var(--neutral);stroke:var(--rule);stroke-width:.7}
.grid-frame{fill:none;stroke:var(--line);stroke-width:1}
.route-cutout{fill:none;stroke:var(--paper);stroke-width:5;stroke-linecap:round}
.blue{--color:var(--blue);--wash:var(--blue-wash)}
.teal{--color:var(--teal);--wash:var(--teal-wash)}
.copper{--color:var(--copper);--wash:var(--copper-wash)}
.violet{--color:var(--violet);--wash:var(--violet-wash)}
.colored{fill:var(--color)}.tile{fill:var(--wash);stroke:var(--color);stroke-width:1.25}
.trace{fill:none;stroke:var(--color);stroke-width:1.7;stroke-linecap:round}
.solid{fill:var(--color);stroke:var(--color);stroke-width:1}
.knockout{fill:#fff}.dash{stroke-dasharray:3 4}
@media(prefers-color-scheme:dark){svg{--ink:#e8edf2;--muted:#aab4bf;--line:#8d9aa8;--rule:#37424d;--paper:#0d1117;--neutral:#1b232c;--blue:#8ab7e6;--blue-wash:#172c43;--teal:#80c4b4;--teal-wash:#17322e;--copper:#dfb17e;--copper-wash:#392c20;--violet:#b6a7da;--violet-wash:#292338}.knockout{fill:#111820}}
"""


class SVG:
    def __init__(self, height: int, title: str, description: str):
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
            f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f"<style>{STYLE}</style>",
            '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" '
            'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            '<path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>',
        ]

    def text(self, x, y, value, cls="", anchor="start"):
        self.items.append(
            f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def rect(self, x, y, width, height, cls="outline", radius=0):
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" class="{cls}"/>'
        )

    def path(self, d, cls="edge", arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        if "C" in d and "trace" in cls:
            # Distinguish crossing routes from joined paths, including in dark mode.
            self.items.append(f'<path d="{d}" class="route-cutout"/>')
        self.items.append(f'<path d="{d}" class="{cls}"{marker}/>')

    def circle(self, x, y, radius, cls="outline"):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{radius}" class="{cls}"/>')

    def cell(self, x, y, value, color=None, width=54, height=34, solid=False):
        cls = f"{color} {'solid' if solid else 'tile'}" if color else "node"
        self.rect(x, y, width, height, cls, 2)
        cls = "data"
        if color:
            cls += f" {color} {'knockout' if solid else 'colored'}"
        elif str(value) in ("−1", "—", "0"):
            cls += " muted"
        self.text(x + width / 2, y + height / 2 + 5, value, cls, "middle")

    def bracket(self, x, y, width, value, color=None):
        self.path(f"M{x} {y}v5h{width}v-5", f"{color} trace" if color else "edge")
        self.text(x + width / 2, y + 24, value, "label muted", "middle")

    def grid(self, x, y, rows, columns, size=12, color=None, selected=()):
        for r in range(rows):
            for c in range(columns):
                cls = f"{color} solid" if r * columns + c in selected else "grid-cell"
                self.rect(x + c * size, y + r * size, size, size, cls)
        self.rect(x, y, columns * size, rows * size, "grid-frame")

    def icon(self, x, y, kind, color="blue"):
        if kind == "tensor":
            for offset in (6, 3, 0):
                self.rect(x + offset, y - offset, 30, 22, "outline", 2)
            self.path(f"M{x + 10} {y}v22M{x + 20} {y}v22M{x} {y + 11}h30", "rule")
        elif kind == "kernel":
            self.rect(x, y, 25, 25, f"{color} tile", 3)
            for d in (6, 12, 18):
                self.path(
                    f"M{x - 4} {y + d}h4M{x + 25} {y + d}h4M{x + d} {y - 4}v4M{x + d} {y + 25}v4"
                )
            self.path(f"M{x + 7} {y + 8}l5 5-5 5M{x + 15} {y + 18}h5", f"{color} trace")
        elif kind == "pages":
            self.rect(x + 4, y - 4, 27, 27, "outline", 2)
            self.rect(x, y, 27, 27, f"{color} tile", 2)
            self.path(f"M{x} {y + 9}h27M{x} {y + 18}h27M{x + 9} {y}v27", f"{color} trace")
        elif kind == "cache":
            self.grid(x, y, 3, 4, size=8, color=color, selected=(1, 6, 8))
        elif kind == "reduce":
            self.path(f"M{x} {y}l14 12M{x + 28} {y}l-14 12M{x + 14} {y + 12}v14")
            self.circle(x + 14, y + 12, 9, f"{color} tile")
            self.text(x + 14, y + 17, "Σ", "math", "middle")

    def save(self, name):
        (ROOT / "assets" / name).write_text("\n".join(self.items + ["</svg>", ""]))


def render_overview():
    s = SVG(
        652,
        "Sparse selections become ordered rank-local KV addresses",
        "Top: selection tensors pass through SILKern localization, a masked KV gather and selected attention. "
        "Panel a traces rank-zero input [8,5,130,7,-1,262] with two ranks, interleave one and page size 64. "
        "Blue, teal and copper identify the survivors 8, 130, 262. Their local positions 4, 65, 131 map through "
        "logical pages 0, 1, 2 to physical pages 11, 2, 7, producing 708, 129, 451. Validity bits 1, 0, 1, 0, 0, 1 "
        "supply stable destinations 0, 1, 2. Output is [708,129,451,-1,-1,-1], count 3. "
        "Panel b draws all 64 slots of physical pages 2, 7, 11, highlighting offsets 1, 3, 4. "
        "The page table is nonmonotonic; output retains selection order. Only valid mappings read the table. "
        "The example illustrates front compaction for multiple ranks, not the single-rank layout. "
        "Python, MLX/Metal and CUDA/Triton share the result with distinct memory ownership.",
    )
    # Context glyphs show data types, rather than adding decorative symbols.
    stages = [
        (22, "tensor", "Global selections"),
        (300, "kernel", "SILKern"),
        (563, "cache", "Masked KV gather"),
        (881, "reduce", "Attention"),
    ]
    for x, kind, label in stages:
        s.icon(x, 24, kind)
        s.text(x + 46, 36, label, "subhead")
    s.text(68, 57, "token positions · selector order", "small muted")
    s.text(346, 57, "rank-local slots + exact count", "small muted")
    s.text(609, 57, "only the selected cache entries", "small muted")
    s.text(927, 57, "shared normalization", "small muted")
    for x, end in ((239, 282), (512, 548), (816, 866)):
        s.path(f"M{x} 35H{end}", arrow=True)
    s.path("M22 81H1098", "rule")
    s.text(22, 112, "a  Filter, translate, preserve order", "heading")
    s.text(22, 136, "D = 2   ·   rank = 0   ·   interleave = 1   ·   page size = 64", "label muted")
    s.path("M627 102V586", "rule")
    s.text(653, 112, "b  Resolve the paged address", "heading")
    s.text(653, 136, "request 0   ·   page table P = [11, 2, 7, 5]", "label muted")
    xs = [204 + i * 65 for i in range(6)]
    active = {0: "blue", 2: "teal", 5: "copper"}
    s.text(22, 169, "Input column", "small muted")
    for j, x in enumerate(xs):
        s.text(x + 27, 169, j, "small mono muted", "middle")
    rows = [
        (182, "Global position  t", [8, 5, 130, 7, "−1", 262]),
        (275, "Local position  ℓ", [4, "—", 65, "—", "—", 131]),
        (372, "Physical slot  a", [708, "—", 129, "—", "—", 451]),
    ]
    for y, label, values in rows:
        s.text(22, y + 23, label)
        for j, (x, value) in enumerate(zip(xs, values, strict=True)):
            s.cell(x, y, value, active.get(j), solid=y == 372 and j in active)
    for j, x in enumerate(xs):
        s.path(f"M{x + 27} 216V231", f"{active[j]} trace" if j in active else "edge", True)
        val = "rank 0" if j in active else "invalid" if j == 4 else "rank 1"
        s.text(
            x + 27,
            249,
            val,
            "small " + (f"{active[j]} colored" if j in active else "muted"),
            "middle",
        )
        if j in active:
            s.path(f"M{x + 27} 256V271", f"{active[j]} trace", True)
            s.path(f"M{x + 27} 309V321", f"{active[j]} trace", True)
        else:
            s.path(f"M{x + 23} 261l8 8m-8 0l8-8", "edge")
    s.text(22, 337, "(page, offset)", "label muted")
    s.text(22, 356, "divmod(ℓ, 64)", "small mono muted")
    for j, value in ((0, "0 · 4"), (2, "1 · 1"), (5, "2 · 3")):
        x = xs[j] + 27
        s.text(x, 343, value, f"mono {active[j]} colored", "middle")
        s.path(f"M{x} 351V368", f"{active[j]} trace", True)
    s.path("M22 421H606", "rule")
    s.text(22, 449, "Validity  v", "label")
    s.text(22, 480, "Stable destination", "label")
    s.text(22, 500, "p = cumsum(v) − 1", "small mono muted")
    for j, x in enumerate(xs):
        if j in active:
            s.circle(x + 27, 444, 10, f"{active[j]} tile")
        s.text(
            x + 27,
            449,
            int(j in active),
            "mono " + (f"{active[j]} colored" if j in active else "muted"),
            "middle",
        )
        value = {0: 0, 2: 1, 5: 2}.get(j, "—")
        s.text(
            x + 27,
            480,
            value,
            "mono " + (f"{active[j]} colored" if j in active else "muted"),
            "middle",
        )
    for source, dest, color in ((0, 0, "blue"), (2, 1, "teal"), (5, 2, "copper")):
        a, b = xs[source] + 27, xs[dest] + 27
        s.path(f"M{a} 489C{a} 509 {b} 506 {b} 527", f"{color} trace", True)
    s.text(22, 553, "Output  s", "subhead")
    for j, (x, value) in enumerate(zip(xs, [708, 129, 451, "−1", "−1", "−1"], strict=True)):
        s.cell(x, 530, value, COLORS[j] if j < 3 else None, solid=j < 3)
    s.bracket(204, 571, 184, "valid prefix · count = 3", "blue")
    s.bracket(399, 571, 184, "padding")
    # Logical pages on the left fan into physically reordered cache blocks.
    s.icon(656, 160, "pages", "teal")
    s.text(696, 170, "Logical", "small muted")
    s.text(758, 170, "Physical", "small muted")
    for j, page in enumerate((11, 2, 7, 5)):
        y = 192 + j * 54
        color = COLORS[j] if j < 3 else None
        s.cell(656, y, j, color, width=42)
        s.path(f"M704 {y + 17}H718", arrow=True)
        s.cell(727, y, page, color, width=45)
    s.text(656, 429, "P[3] = 5 is unused", "small muted")
    s.text(656, 450, "in this selection.", "small muted")
    s.text(656, 480, "Survivor identity", "small muted")
    for y, token, slot, color in (
        (498, 8, 708, "blue"),
        (521, 130, 129, "teal"),
        (544, 262, 451, "copper"),
    ):
        s.circle(663, y - 5, 4, f"{color} solid")
        s.text(678, y, f"{token} → {slot}", f"small mono {color} colored")
    cache_rows = [(2, 1, "teal", 174), (7, 3, "copper", 306), (11, 4, "blue", 438)]
    for page, offset, color, y in cache_rows:
        s.text(869, y - 8, f"Page {page}", "subhead")
        s.text(1094, y - 8, f"{page * 64}–{page * 64 + 63}", "small mono muted", "end")
        s.grid(870, y, 8, 8, size=12, color=color, selected=(offset,))
        s.text(988, y + 33, f"+ {offset}", f"mono {color} colored")
        s.text(988, y + 58, page * 64 + offset, f"subhead {color} colored")
        s.text(988, y + 78, "KV slot", "small muted")
        s.path(f"M{870 + (offset + 0.5) * 12} {y + 6}H974", f"{color} trace")
    for y0, y1, color in ((209, 480, "blue"), (263, 216, "teal"), (317, 348, "copper")):
        s.path(f"M774 {y0}C816 {y0} 822 {y1} 863 {y1}", f"{color} trace", True)
    s.text(653, 568, "Each cell is one token’s K/V entry.", "label muted")
    s.text(653, 590, "Physical order differs from selection order.", "label muted")
    s.path("M22 613H1098", "rule")
    s.text(22, 639, "Shared result", "subhead")
    s.text(194, 639, "Python · lists", "label muted")
    s.text(424, 639, "MLX / Metal · returned arrays", "label muted")
    s.text(770, 639, "CUDA / Triton · caller-owned buffers", "label muted")
    s.save("fig-platforms.svg")


def render_contract():
    s = SVG(
        514,
        "Coordinate translation and the two output layouts",
        "The worked row uses two ranks, rank zero, interleave one, page size 64 and pages 11, 2, 7,5. "
        "Columns 0, 2, 5 survive: tokens 8, 130, 262 become local positions 4, 65, 131, logical page and offset "
        "pairs (0, 4), (1, 1), (2, 3), then physical slots 708, 129, 451. Validity is 1, 0, 1, 0, 0, 1 and "
        "the surviving scan destinations are0,1,2. Front compaction returns 708,129,451,-1,-1,-1. "
        "Disabling compaction returns 708,-1,129,-1,-1,451. Both counts are 3; only the compacted "
        "row has a valid count-length prefix. Single-rank calls also preserve columns, with their "
        "own coordinate translation. General ownership, deinterleaving and page formulas appear at right.",
    )
    s.text(22, 28, "a  A complete row, in selector order", "heading")
    s.text(22, 53, "D = 2 · rank = 0 · I = 1 · S = 64 · P = [11, 2, 7, 5]", "label muted")
    s.path("M22 68H686M716 10V496", "rule")
    xs = [230 + 75 * j for j in range(6)]
    active = {0: "blue", 2: "teal", 5: "copper"}
    s.text(22, 94, "Input column", "small muted")
    for j, x in enumerate(xs):
        s.text(x + 30, 94, j, "small mono muted", "middle")
    rows = [
        (107, "Global position  t", [8, 5, 130, 7, "−1", 262]),
        (157, "Owner", [0, 1, 0, 1, "—", 0]),
        (207, "Local position  ℓ", [4, "—", 65, "—", "—", 131]),
        (257, "(logical page, offset)", ["0, 4", "—", "1, 1", "—", "—", "2, 3"]),
        (307, "Physical slot  a", [708, "—", 129, "—", "—", 451]),
    ]
    for y, label, values in rows:
        s.text(22, y + 23, label, "label")
        for j, (x, value) in enumerate(zip(xs, values, strict=True)):
            s.cell(x, y, value, active.get(j), width=60, solid=y == 307 and j in active)
            if j in active and y != 307:
                s.path(f"M{x + 30} {y + 35}v11", f"{active[j]} trace", True)
    s.path("M22 356H686", "rule")
    s.text(22, 384, "Validity  v", "label")
    s.text(22, 415, "cumsum(v) − 1", "mono")
    for j, x in enumerate(xs):
        cls = f"mono {active[j]} colored" if j in active else "mono muted"
        s.text(x + 30, 384, int(j in active), cls, "middle")
        s.text(x + 30, 415, {0: 0, 2: 1, 5: 2}.get(j, "—"), cls, "middle")
    for source, dest, color in ((0, 0, "blue"), (2, 1, "teal"), (5, 2, "copper")):
        a, b = xs[source] + 30, xs[dest] + 30
        s.path(f"M{a} 424C{a} 438 {b} 437 {b} 450", f"{color} trace", True)
    s.text(22, 476, "Compacted output", "subhead")
    for j, (x, value) in enumerate(zip(xs, [708, 129, 451, "−1", "−1", "−1"], strict=True)):
        s.cell(x, 453, value, COLORS[j] if j < 3 else None, width=60, solid=j < 3)
    s.text(447, 507, "count = 3 · stable filter, not a sort", "small muted", "middle")
    s.text(742, 28, "b  Coordinates and layout", "heading")
    s.text(742, 66, "For nonnegative t", "small muted")
    for y, value in (
        (94, "owner = ⌊t / I⌋ mod D"),
        (127, "ℓ = ⌊t / (D I)⌋ I + t mod I"),
        (160, "(b, δ) = divmod(ℓ, S)"),
        (193, "a = P[request, b] S + δ"),
    ):
        s.text(742, y, value, "math")
    s.text(742, 221, "Read P only for an owned, in-range mapping.", "small muted")
    s.path("M742 239H1098", "rule")
    for y, title, values, mask, color_map in (
        (
            260,
            "Front compaction · count = 3",
            [708, 129, 451, "−1", "−1", "−1"],
            [1, 1, 1, 0, 0, 0],
            {0: "blue", 1: "teal", 2: "copper"},
        ),
        (
            363,
            "Compaction off · count = 3",
            [708, "−1", 129, "−1", "−1", 451],
            [1, 0, 1, 0, 0, 1],
            active,
        ),
    ):
        s.text(742, y, title, "subhead")
        for j, value in enumerate(values):
            s.cell(743 + j * 59, y + 17, value, color_map.get(j), width=52, height=32)
            s.text(769 + j * 59, y + 70, mask[j], "mono muted", "middle")
        s.text(742, y + 88, "validity mask", "small muted")
    s.text(742, 488, "Single-rank calls also preserve input columns.", "small muted")
    s.save("fig-contract.svg")


def render_order():
    s = SVG(
        368,
        "Completion-order reservation versus deterministic tile prefixes",
        "Tile groups T0=[704,705], T1=[132,133], T2=[448,896], T3=[897,260] are tracked by four colors. "
        "Atomic reservation may emit T1,T0,T3,T2; T0,T2,T1,T3; or T2,T3,T0,T1 on separate replays. "
        "Stable input-prefix offsets 0, 2, 4, 6 always emit T0,T1,T2,T3. Both paths have the same eight "
        "values and exact count. Curves illustrate only the first replay; these arrays are illustrative, "
        "not measured GPU results.",
    )
    vals = [(704, 705), (132, 133), (448, 896), (897, 260)]
    s.path("M560 15V350", "rule")
    for panel in (0, 1):
        x0 = 22 + panel * 560
        s.text(x0, 28, ("a  Atomic reservation", "b  Stable prefix destinations")[panel], "heading")
        s.text(
            x0,
            53,
            (
                "Tile arrival determines the reserved segment",
                "Input order determines every destination",
            )[panel],
            "label muted",
        )
        xs = [x0 + 100 + j * 99 for j in range(4)]
        orders = [[1, 0, 3, 2], [0, 2, 1, 3], [2, 3, 0, 1]] if panel == 0 else [[0, 1, 2, 3]] * 3
        for y, label, order in [(88, "Input", [0, 1, 2, 3])] + [
            (185 + i * 59, f"Replay {i + 1}", order) for i, order in enumerate(orders)
        ]:
            s.text(x0, y + 24, label, "label")
            for pos, identity in enumerate(order):
                x = xs[pos]
                color = COLORS[identity]
                s.text(x + 44, y - 7, f"T{identity}", f"small mono {color} colored", "middle")
                s.rect(x, y, 88, 34, f"{color} tile", 3)
                for j, value in enumerate(vals[identity]):
                    s.text(x + 22 + 44 * j, y + 23, value, f"mono {color} colored", "middle")
                s.path(f"M{x + 44} {y + 5}v24", "rule")
        for identity in range(4):
            a = xs[identity] + 44
            b = xs[orders[0].index(identity)] + 44
            s.path(f"M{a} 124C{a} 145 {b} 144 {b} 164", f"{COLORS[identity]} trace", True)
        s.text(
            x0,
            356,
            (
                "Same values and count; tile order can change.",
                "Offsets [0, 2, 4, 6] are independent of scheduling.",
            )[panel],
            "label muted",
        )
    s.save("fig-problem.svg")


def render_consumer():
    s = SVG(
        464,
        "Layout-aware masking, paged gathering and a shared softmax",
        "Panel a compares a compact row [14,12,-1,-1], mask [1,1,0,0], with a column-preserving row "
        "[14,-1,12,-1], mask [1,0,1,0]. Both counts are 2. Panel b substitutes address 0 for invalid "
        "positions, gathers K and V, then selects zero for invalid gathered values; unused cache "
        "slot 0 may contain NaN and must be masked after gathering. A 16-slot schematic highlights "
        "cache entries 12 and 14. Panel c computes one maximum across all logical shards and selected "
        "positions, zeroes invalid weights, and combines the sums of weights and weighted values "
        "with a single division. An empty selection returns zero. Valid arithmetic must be finite "
        "and representable; the diagram is a one-device consumer, not a distributed runtime.",
    )
    s.path("M351 12V450M728 12V450", "rule")
    s.text(22, 28, "a  Layout defines validity", "heading")
    s.text(374, 28, "b  Mask addresses and values", "heading")
    s.text(752, 28, "c  One shared normalization", "heading")
    for y, title, values, mask, active in (
        (70, "Front compaction", [14, 12, "−1", "−1"], [1, 1, 0, 0], {0: "blue", 1: "teal"}),
        (239, "Column preserving", [14, "−1", 12, "−1"], [1, 0, 1, 0], {0: "blue", 2: "teal"}),
    ):
        s.text(22, y, title, "subhead")
        s.text(326, y, "n = 2", "small mono muted", "end")
        s.text(24, y + 45, "s", "math")
        s.text(24, y + 89, "v", "math")
        for j, value in enumerate(values):
            s.cell(69 + j * 64, y + 23, value, active.get(j), width=57)
            s.text(
                97 + j * 64,
                y + 89,
                mask[j],
                "mono " + (f"{active[j]} colored" if j in active else "muted"),
                "middle",
            )
        s.text(
            193,
            y + 126,
            "vⱼ = [j < n]" if y == 70 else "vⱼ = input-column validity",
            "math",
            "middle",
        )
    s.path("M22 216H326", "rule")
    s.icon(28, 402, "tensor")
    s.text(78, 411, "Equal counts can have", "label muted")
    s.text(78, 432, "different valid positions.", "label muted")
    # The same concrete hole-preserving row flows through both masks.
    s.text(378, 79, "s", "math")
    s.text(418, 79, "[14, −1, 12, −1]", "mono")
    s.text(378, 109, "v", "math")
    s.text(418, 109, "[ 1,  0,  1,  0]", "mono blue colored")
    s.path("M402 72H408V155H426M544 119V138", arrow=True)
    s.rect(426, 139, 238, 34, "blue tile", 3)
    s.text(545, 161, "where(v, s, 0)", "mono", "middle")
    s.path("M545 173V190", arrow=True)
    for j, value in enumerate((14, 0, 12, 0)):
        s.cell(418 + j * 64, 194, value, {0: "blue", 2: "teal"}.get(j), width=57, height=30)
    s.path("M545 226V259", arrow=True)
    s.rect(451, 260, 177, 33, "node", 3)
    s.text(539, 282, "gather K / V", "subhead", "middle")
    s.grid(659, 253, 4, 4, size=11, color="teal", selected=(12, 14))
    s.text(703, 240, "KV cache", "small muted", "end")
    s.path("M655 277H632", arrow=True)
    s.path("M539 293V310", arrow=True)
    s.text(546, 332, "[x₁₄, x₀, x₁₂, x₀]", "math", "middle")
    s.path("M545 341V356M393 106V374H426", arrow=True)
    s.rect(426, 357, 238, 34, "teal tile", 3)
    s.text(545, 379, "where(v, x, 0)", "mono", "middle")
    s.path("M545 391V407", arrow=True)
    s.text(545, 430, "[x₁₄, 0, x₁₂, 0]", "math", "middle")
    s.text(545, 455, "x ∈ {K, V}; unused x₀ may be NaN.", "small muted", "middle")
    # Dataflow of the shared maximum and two reductions stays explicit.
    s.icon(762, 61, "tensor")
    s.text(945, 75, "zᵣⱼ = q · Kᵣⱼ / √d", "math", "middle")
    s.text(945, 99, "invalid zᵣⱼ ← −∞", "small muted", "middle")
    s.path("M944 107V130M784 113V228H802", arrow=True)
    s.path("M784 113H944")
    s.rect(850, 131, 188, 35, "blue tile", 3)
    s.text(944, 155, "m = maxᵣ,ⱼ zᵣⱼ", "math", "middle")
    s.path("M944 166V209", arrow=True)
    s.text(1037, 192, "empty: m ← 0", "small muted", "end")
    s.rect(803, 210, 292, 38, "node", 3)
    s.text(949, 235, "wᵣⱼ = vᵣⱼ ? exp(zᵣⱼ − m) : 0", "math", "middle")
    s.path("M949 248V265H845V279M949 265H1045V322M789 295H829", arrow=True)
    s.text(764, 298, "Vᵣⱼ", "math")
    s.circle(845, 295, 15, "teal tile")
    s.text(845, 301, "×", "math", "middle")
    s.path("M845 310V322", arrow=True)
    for x in (845, 1045):
        s.circle(x, 340, 18, "teal tile")
        s.text(x, 347, "Σ", "math", "middle")
    s.text(945, 336, "all shards", "small muted", "middle")
    s.text(945, 355, "all positions", "small muted", "middle")
    s.path("M845 358V392H927M1045 358V392H963", arrow=True)
    s.text(857, 383, "N", "math")
    s.text(1018, 383, "Z", "math")
    s.circle(945, 392, 17, "blue tile")
    s.text(945, 398, "÷", "math", "middle")
    s.path("M945 409V427", arrow=True)
    s.text(945, 454, "y = N / Z if Z > 0; otherwise 0", "math", "middle")
    s.save("fig-consumer.svg")


def main():
    assert localize_reference(
        [0],
        [[11, 2, 7, 5]],
        [[8, 5, 130, 7, -1, 262]],
        block_size=64,
        dcp_size=2,
        dcp_rank=0,
    ) == ([[708, 129, 451, -1, -1, -1]], [3])
    assert localize_reference(
        [0],
        [[11, 2, 7, 5]],
        [[8, 5, 130, 7, -1, 262]],
        block_size=64,
        dcp_size=2,
        dcp_rank=0,
        compact_valid_to_front=False,
    ) == ([[708, -1, 129, -1, -1, 451]], [3])
    render_overview()
    render_contract()
    render_order()
    render_consumer()


if __name__ == "__main__":
    main()
