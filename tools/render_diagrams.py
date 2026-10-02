"""Render the explanatory SVGs without raster assets or hardware execution.

The worked examples are checked against the pure-Python contract before export.
Semantic colors follow survivor identity; neutral canvas and adaptive ink keep
the figures legible when embedded in light or dark Markdown pages.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from figure_style import TEXT_CSS, THEME_CSS, WIDTH

from silkern import localize_reference

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("blue", "teal", "copper", "violet")
STYLE = (
    THEME_CSS
    + TEXT_CSS
    + """
.rule{fill:none;stroke:var(--rule);stroke-width:1}
.edge{fill:none;stroke:var(--line);stroke-width:1.25;stroke-linejoin:round;stroke-linecap:round}
.outline{fill:var(--paper);stroke:var(--line);stroke-width:1.2}
.grid-frame,.array-frame{fill:none;stroke:var(--line);stroke-width:1}
.wash{fill:var(--wash);stroke:none}.plain{fill:var(--paper);stroke:none}
.array-divider{fill:none;stroke:var(--rule);stroke-width:.8}
.port{fill:var(--paper);stroke:var(--color);stroke-width:1.25}
.route-cutout{fill:none;stroke:var(--paper);stroke-width:5;stroke-linecap:round}
.blue{--color:var(--blue);--wash:var(--blue-wash)}
.teal{--color:var(--teal);--wash:var(--teal-wash)}
.copper{--color:var(--copper);--wash:var(--copper-wash)}
.violet{--color:var(--violet);--wash:var(--violet-wash)}
.colored{fill:var(--color)}.tile{fill:var(--wash);stroke:var(--color);stroke-width:1.25}
.trace{fill:none;stroke:var(--color);stroke-width:1.7;stroke-linecap:round}
.solid{fill:var(--color);stroke:var(--color);stroke-width:1}
.knockout{fill:var(--knockout)}
"""
)


class SVG:
    def __init__(self, height: int, title: str, description: str):
        self.clip_count = 0
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
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
        # Separate branches so each receives an arrowhead, not only the last one.
        routes = ["M" + part for part in d.split("M")[1:]] if arrow else [d]
        for route in routes:
            if ("C" in route or "Q" in route) and "trace" in cls:
                # Distinguish crossing routes from joined paths in either theme.
                self.items.append(f'<path d="{route}" class="route-cutout"/>')
            self.items.append(f'<path d="{route}" class="{cls}"{marker}/>')

    def circle(self, x, y, radius, cls="outline"):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{radius}" class="{cls}"/>')

    def array(self, x, y, values, colors=None, *, step=70, height=40, solid=False):
        """One tensor strip, with internal divisions instead of disconnected boxes."""
        colors = colors or {}
        width = step * len(values)
        self.clip_count += 1
        clip_id = f"array-{self.clip_count}"
        self.items.append(
            f'<defs><clipPath id="{clip_id}"><rect x="{x}" y="{y}" '
            f'width="{width}" height="{height}" rx="6"/></clipPath></defs>'
        )
        self.items.append(f'<g clip-path="url(#{clip_id})">')
        self.rect(x, y, width, height, "plain")
        for j, value in enumerate(values):
            color = colors.get(j)
            if color:
                self.rect(x + j * step, y, step, height, f"{color} {'solid' if solid else 'wash'}")
            if j:
                self.path(f"M{x + j * step} {y}v{height}", "array-divider")
            cls = f"data {color} {'knockout' if solid else 'colored'}" if color else "data muted"
            self.text(x + (j + 0.5) * step, y + height / 2 + 6, value, cls, "middle")
        self.items.append("</g>")
        self.rect(x, y, width, height, "array-frame", 6)

    def operation(self, x, y, width, height, value, color="blue", gate=False):
        """Distinct gate geometry for selection; rounded blocks for arithmetic."""
        if gate:
            self.path(
                f"M{x + 12} {y}H{x + width - 12}L{x + width} {y + height / 2}"
                f"L{x + width - 12} {y + height}H{x + 12}L{x} {y + height / 2}Z",
                f"{color} tile",
            )
        else:
            self.rect(x, y, width, height, f"{color} tile", 7)
        self.text(x + width / 2, y + height / 2 + 6, value, "", "middle")

    def bracket(self, x, y, width, value, color=None):
        self.path(f"M{x} {y}v5h{width}v-5", f"{color} trace" if color else "edge")
        self.text(x + width / 2, y + 24, value, "muted", "middle")

    def grid(self, x, y, rows, columns, size=12, color=None, selected=(), paired=False):
        width, height = columns * size, rows * size
        if paired:
            # Two aligned planes represent the K and V pages, not decorative shadows.
            self.rect(x + 5, y - 5, width, height, "outline", 4)
        self.rect(x, y, width, height, "plain", 4)
        for r in range(1, rows):
            self.path(f"M{x} {y + r * size}h{width}", "array-divider")
        for c in range(1, columns):
            self.path(f"M{x + c * size} {y}v{height}", "array-divider")
        for index in selected:
            r, c = divmod(index, columns)
            self.rect(x + c * size + 1, y + r * size + 1, size - 2, size - 2, f"{color} solid", 1)
        self.rect(x, y, width, height, "grid-frame", 4)

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
            self.path(f"M{x} {y}l6 5M{x + 28} {y}l-6 5M{x + 14} {y + 23}v4")
            self.circle(x + 14, y + 12, 9, f"{color} tile")
            self.text(x + 14, y + 17, "Σ", "", "middle")

    def save(self, name):
        (ROOT / "assets" / name).write_text("\n".join(self.items + ["</svg>", ""]))


def render_overview():
    s = SVG(
        588,
        "Sparse selections become ordered rank-local KV addresses",
        "Selections pass through SILKern, a masked KV gather and selected attention. "
        "Panel a uses rank zero of two ranks, interleave one, page size 64, request zero, "
        "and page table [11,2,7,5]. Input [8,5,130,7,-1,262] has validity [1,0,1,0,0,1]. "
        "The surviving slots 708,129,451 receive prefix-derived destinations 0,1,2; "
        "the compacted result is [708,129,451,-1,-1,-1], count 3. "
        "Panel b maps logical pages 0,1,2 to physical pages 11,2,7. Each cache grid contains "
        "64 K/V entries; selected offsets 4,1,3 give slots 708,129,451. Physical pages are "
        "drawn in ascending order to distinguish physical placement from selector order. "
        "Page-table entry 3 maps to page 5 but is unused. This is the multi-rank compacting layout.",
    )
    for x, kind, label in (
        (24, "tensor", "Global selections"),
        (316, "kernel", "SILKern"),
        (567, "cache", "Masked KV gather"),
        (911, "reduce", "Attention"),
    ):
        s.icon(x, 26, kind)
        s.text(x + 46, 44, label, "subhead")
    for start, end in ((246, 297), (458, 547), (796, 891)):
        s.path(f"M{start} 38H{end}", arrow=True)
    s.path("M24 80H1096", "rule")
    s.text(24, 120, "a  Stable rank-local localization", "heading")
    s.text(700, 120, "b  Paged address translation", "heading")
    s.path("M672 102V556", "rule")

    xs = [230 + 70 * j for j in range(6)]
    active = {0: "blue", 2: "teal", 5: "copper"}
    for y, label, values in (
        (170, "Global position", [8, 5, 130, 7, "−1", 262]),
        (296, "Physical slot", [708, "—", 129, "—", "—", 451]),
    ):
        s.text(24, y + 26, label)
        s.array(230, y, values, active)
    s.text(24, 260, "Rank 0 validity")
    s.text(24, 389, "Prefix destination")
    for j, x in enumerate(xs):
        color = active.get(j)
        cls = f"{color} colored" if color else "muted"
        s.path(f"M{x + 35} 212V233", f"{color} trace" if color else "edge", True)
        s.text(x + 35, 260, int(j in active), cls, "middle")
        if color:
            s.path(f"M{x + 35} 271V291", f"{color} trace", True)
            s.path(f"M{x + 35} 340V364", f"{color} trace", True)
        s.text(x + 35, 389, {0: 0, 2: 1, 5: 2}.get(j, "—"), cls, "middle")
    for source, dest, color in ((0, 0, "blue"), (2, 1, "teal"), (5, 2, "copper")):
        a, b = xs[source] + 35, xs[dest] + 35
        s.path(f"M{a} 402C{a} 433 {b} 424 {b} 456", f"{color} trace", True)
        s.circle(a, 402, 2.5, f"{color} port")
    s.text(24, 487, "Output", "subhead")
    s.array(
        230, 461, [708, 129, 451, "−1", "−1", "−1"], {0: "blue", 1: "teal", 2: "copper"}, solid=True
    )
    s.bracket(230, 513, 210, "count = 3", "blue")
    s.bracket(440, 513, 210, "padding")

    s.text(700, 163, "Logical → physical", "muted")
    s.items.append(
        '<defs><clipPath id="page-table"><rect x="704" y="190" width="132" height="292" rx="6"/></clipPath></defs>'
    )
    s.items.append('<g clip-path="url(#page-table)">')
    s.rect(704, 190, 132, 292, "plain")
    for j, page in enumerate((11, 2, 7, 5)):
        y = 190 + j * 73
        color = COLORS[j] if j < 3 else None
        if color:
            s.rect(704, y, 132, 73, f"{color} wash")
            s.rect(704, y, 3, 73, f"{color} solid")
        if j:
            s.path(f"M704 {y}h132", "array-divider")
        cls = f"{color} colored" if color else "muted"
        s.text(730, y + 43, j, cls, "middle")
        s.path(f"M755 {y + 36.5}H774", arrow=True)
        s.text(806, y + 43, page, cls, "middle")
    s.items.append("</g>")
    s.rect(704, 190, 132, 292, "array-frame", 6)
    for page, offset, color, y in (
        (2, 1, "teal", 190),
        (7, 3, "copper", 330),
        (11, 4, "blue", 470),
    ):
        s.text(895, y - 18, f"Page {page} · slot {page * 64 + offset}", "subhead")
        s.grid(951, y, 8, 8, size=12, color=color, selected=(offset,), paired=True)
    for y0, color, route in (
        (226.5, "blue", "M836 226.5H854Q866 226.5 866 238.5V506Q866 518 878 518H946"),
        (299.5, "teal", "M836 299.5H868Q880 299.5 880 287.5V250Q880 238 892 238H946"),
        (372.5, "copper", "M836 372.5C882 372.5 900 378 946 378"),
    ):
        s.path(route, f"{color} trace", True)
        s.circle(836, y0, 2.5, f"{color} port")
    s.save("fig-platforms.svg")


def render_contract():
    s = SVG(
        432,
        "Coordinate translation and the two output layouts",
        "For D=2, rank=0, I=1, S=64, request zero and P=[11,2,7,5], tokens 8,130,262 "
        "survive at columns 0,2,5. Their local positions 4,65,131 give page-offset pairs "
        "(0,4),(1,1),(2,3) and physical slots 708,129,451. The other tokens belong to "
        "rank one or are negative. General formulas for nonnegative t appear at right. "
        "P is read only for an owned, in-range mapping. Front compaction returns "
        "[708,129,451,-1,-1,-1]; with compaction off the result is [708,-1,129,-1,-1,451]. "
        "Both counts are 3, but only the compacted row has a valid count-length prefix. "
        "Single-rank calls also preserve columns, with their own coordinate translation.",
    )
    s.text(24, 36, "a  Coordinate translation", "heading")
    s.text(740, 36, "b  Equations and layouts", "heading")
    s.path("M708 16V412", "rule")
    xs = [230 + 75 * j for j in range(6)]
    active = {0: "blue", 2: "teal", 5: "copper"}
    for y, label, values in (
        (80, "Global position  t", [8, 5, 130, 7, "−1", 262]),
        (148, "Owner", [0, 1, 0, 1, "—", 0]),
        (216, "Local position  ℓ", [4, "—", 65, "—", "—", 131]),
        (284, "Page, offset", ["0, 4", "—", "1, 1", "—", "—", "2, 3"]),
        (352, "Physical slot  a", [708, "—", 129, "—", "—", 451]),
    ):
        s.text(24, y + 26, label)
        s.array(230, y, values, active, step=75, solid=y == 352)
        for j, x in enumerate(xs):
            if j in active and y != 352:
                s.path(f"M{x + 37.5} {y + 42}v21", f"{active[j]} trace", True)
    for y, value in (
        (90, "owner = ⌊t / I⌋ mod D"),
        (128, "ℓ = ⌊t / (D I)⌋ I + t mod I"),
        (166, "(b, δ) = divmod(ℓ, S)"),
        (204, "a = P[request, b] S + δ"),
    ):
        s.text(740, y, value)
    s.path("M740 230H1096", "rule")
    for y, title, values, color_map in (
        (
            264,
            "Front compaction · count = 3",
            [708, 129, 451, "−1", "−1", "−1"],
            {0: "blue", 1: "teal", 2: "copper"},
        ),
        (360, "Preserve columns · count = 3", [708, "−1", 129, "−1", "−1", 451], active),
    ):
        s.text(740, y, title, "subhead")
        s.array(740, y + 16, values, color_map, step=356 / 6)
    s.save("fig-contract.svg")


def render_order():
    s = SVG(
        376,
        "Completion-order reservation versus deterministic tile prefixes",
        "Tile groups T0=[704,705], T1=[132,133], T2=[448,896], T3=[897,260] are tracked "
        "by four colors. Atomic reservation can emit T1,T0,T3,T2; T0,T2,T1,T3; "
        "or T2,T3,T0,T1 on separate replays. Stable input-prefix offsets 0,2,4,6 always "
        "emit T0,T1,T2,T3. Both paths have the same eight values and count. Curves "
        "illustrate the first replay only; these are schematic arrays, not measured GPU results.",
    )
    vals = [(704, 705), (132, 133), (448, 896), (897, 260)]
    s.path("M560 16V358", "rule")
    for panel in (0, 1):
        x0 = 24 + panel * 560
        s.text(x0, 36, ("a  Atomic reservation", "b  Stable prefix destinations")[panel], "heading")
        xs = [x0 + 104 + j * 96 for j in range(4)]
        orders = [[1, 0, 3, 2], [0, 2, 1, 3], [2, 3, 0, 1]] if panel == 0 else [[0, 1, 2, 3]] * 3
        for y, label, order in [(96, "Input", [0, 1, 2, 3])] + [
            (198 + i * 60, f"Replay {i + 1}", order) for i, order in enumerate(orders)
        ]:
            s.text(x0, y + 26, label)
            for pos, identity in enumerate(order):
                x = xs[pos]
                color = COLORS[identity]
                if label == "Input":
                    s.text(x + 48, y - 14, f"T{identity}", f"{color} colored", "middle")
            s.array(
                xs[0],
                y,
                [f"{vals[i][0]}  {vals[i][1]}" for i in order],
                {j: COLORS[i] for j, i in enumerate(order)},
                step=96,
            )
        for identity in range(4):
            a = xs[identity] + 48
            b = xs[orders[0].index(identity)] + 48
            s.path(f"M{a} 139C{a} 165 {b} 163 {b} 193", f"{COLORS[identity]} trace", True)
            s.circle(a, 139, 2.5, f"{COLORS[identity]} port")
    s.save("fig-problem.svg")


def render_consumer():
    s = SVG(
        516,
        "Layout-aware masking, paged gathering and a shared softmax",
        "Panel a compares compact slots [14,12,-1,-1] with mask [1,1,0,0] to preserved "
        "columns [14,-1,12,-1] with mask [1,0,1,0]. Both counts are 2. Panel b substitutes "
        "address zero for invalid positions, gathers K/V, then selects zero for invalid "
        "gathered values. Unused cache slot zero may contain NaN, so both masks are needed. "
        "A 16-slot grid highlights entries 12 and 14. Panel c masks invalid logits to "
        "negative infinity, computes one maximum over all logical shards and positions, "
        "and selects zero for invalid weights. An empty selection sets the maximum to zero. "
        "Sums N of weighted values and Z of weights are combined in one division; if Z=0, "
        "the output is zero. Valid arithmetic must be finite and representable. This "
        "illustrates a one-device consumer, not a distributed runtime.",
    )
    s.path("M351 16V496M728 16V496", "rule")
    s.text(24, 36, "a  Respect the layout", "heading")
    s.text(375, 36, "b  Mask the gather", "heading")
    s.text(752, 36, "c  Normalize together", "heading")
    for y, title, values, mask, active in (
        (
            90,
            "Front compaction · n = 2",
            [14, 12, "−1", "−1"],
            [1, 1, 0, 0],
            {0: "blue", 1: "teal"},
        ),
        (
            300,
            "Preserve columns · n = 2",
            [14, "−1", 12, "−1"],
            [1, 0, 1, 0],
            {0: "blue", 2: "teal"},
        ),
    ):
        s.text(24, y, title, "subhead")
        s.text(24, y + 56, "Slots")
        s.text(24, y + 106, "Mask")
        s.array(85, y + 30, values, active, step=60.5)
        for j in range(len(values)):
            s.text(
                115.25 + j * 60.5,
                y + 106,
                mask[j],
                f"{active[j]} colored" if j in active else "muted",
                "middle",
            )
        s.text(
            201, y + 151, "vⱼ = [j < n]" if y == 90 else "vⱼ = input-column validity", "", "middle"
        )
    s.path("M24 267H327", "rule")

    s.text(378, 90, "s")
    s.text(418, 90, "[14, −1, 12, −1]")
    s.text(378, 124, "v")
    s.text(418, 124, "[1, 0, 1, 0]", "blue colored")
    s.path("M403 84H408V176H426M545 133V150", arrow=True)
    s.operation(426, 154, 238, 42, "where(v, s, 0)", gate=True)
    s.path("M545 198V216", arrow=True)
    s.array(417, 221, [14, 0, 12, 0], {0: "blue", 2: "teal"}, step=64, height=36)
    s.path("M545 261V289", arrow=True)
    s.rect(451, 293, 177, 40, "outline", 7)
    s.text(539, 319, "Gather K / V", "subhead", "middle")
    s.grid(659, 291, 4, 4, size=11, color="teal", selected=(12, 14), paired=True)
    s.path("M655 313H632", arrow=True)
    s.path("M539 337V355", arrow=True)
    s.text(545, 383, "[x₁₄, x₀, x₁₂, x₀]", "", "middle")
    s.path("M545 395V413M398 118V439H422", arrow=True)
    s.operation(426, 417, 238, 42, "where(v, x, 0)", color="teal", gate=True)
    s.path("M545 463V477", arrow=True)
    s.text(545, 502, "[x₁₄, 0, x₁₂, 0]", "", "middle")

    s.text(924, 90, "zᵣⱼ = q · Kᵣⱼ / √d", "", "middle")
    s.text(924, 124, "Invalid zᵣⱼ ← −∞", "muted", "middle")
    s.path("M924 137V159", arrow=True)
    s.operation(799, 164, 250, 42, "m = maxᵣ,ⱼ zᵣⱼ")
    s.path("M924 210V248", arrow=True)
    s.operation(752, 253, 344, 42, "wᵣⱼ = vᵣⱼ ? exp(zᵣⱼ − m) : 0", gate=True)
    s.path("M924 299V324H829V351M924 324H1019V351", arrow=True)
    for x, value in ((757, "N = Σᵣ,ⱼ wᵣⱼ Vᵣⱼ"), (947, "Z = Σᵣ,ⱼ wᵣⱼ")):
        s.operation(x, 356, 144, 42, value, color="teal")
    s.path("M829 402V426H924V449M1019 402V426H924")
    s.path("M924 439V449", arrow=True)
    s.rect(764, 454, 320, 46, "blue tile", 23)
    s.text(924, 483, "y = N / Z; 0 if Z = 0", "", "middle")
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
