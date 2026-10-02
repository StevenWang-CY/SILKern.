"""Render the explanatory SVGs without raster assets or hardware execution.

The worked examples are checked against the pure-Python contract before export.
Semantic colors follow survivor identity; neutral canvas and adaptive ink keep
the figures legible when embedded in light or dark Markdown pages.
"""

from __future__ import annotations

import re
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
.edge{fill:none;stroke:var(--line);stroke-width:1.1;stroke-linejoin:round;stroke-linecap:round}
.outline{fill:var(--paper);stroke:var(--ink);stroke-width:1.05}
.tensor-bracket{fill:none;stroke:var(--line);stroke-width:1.1}
.control{fill:none;stroke:var(--line);stroke-width:1.1;stroke-dasharray:3 3}
.junction{fill:var(--line);stroke:none}
.memory{fill:var(--paper);stroke:var(--line);stroke-width:1}
.memory-active{fill:var(--wash);stroke:var(--color);stroke-width:1}
.mask-bit{fill:var(--color);stroke:none}
.mask-zero{fill:none;stroke:var(--rule);stroke-width:1}
.math-index{font-size:75%}
.array-divider{fill:none;stroke:var(--rule);stroke-width:.8}
.route-cutout{fill:none;stroke:var(--paper);stroke-width:5;stroke-linecap:round}
.blue{--color:var(--blue);--wash:var(--blue-wash)}
.teal{--color:var(--teal);--wash:var(--teal-wash)}
.copper{--color:var(--copper);--wash:var(--copper-wash)}
.violet{--color:var(--violet);--wash:var(--violet-wash)}
.colored{fill:var(--color)}
.trace{fill:none;stroke:var(--color);stroke-width:1.3;stroke-linecap:round}
"""
)


class SVG:
    def __init__(self, height: int, title: str, description: str):
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
        value = str(value)
        if cls == "heading" and len(value) > 3 and value[1:3] == "  ":
            content = (
                f'<tspan class="panel">{value[0]}</tspan><tspan dx="12">{escape(value[3:])}</tspan>'
            )
        else:
            content = escape(value)
        self.items.append(
            f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{content}</text>'
        )

    def math(self, x, y, value, cls="", anchor="start"):
        """Keep mathematical indices editable, attached to the parent glyphs."""
        parts = re.split(r"(_\{[^}]+\})", value)
        content = "".join(
            f'<tspan class="math-index" baseline-shift="sub">{escape(part[2:-1])}</tspan>'
            if part.startswith("_{")
            else escape(part)
            for part in parts
        )
        self.items.append(
            f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{content}</text>'
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

    def array(self, x, y, values, colors=None, *, step=70, height=40, emphasis=False):
        """An aligned vector in mathematical brackets; emphasis never adds a box."""
        colors = colors or {}
        width = step * len(values)
        self.path(
            f"M{x + 6} {y}H{x}V{y + height}H{x + 6}"
            f"M{x + width - 6} {y}H{x + width}V{y + height}H{x + width - 6}",
            "tensor-bracket",
        )
        for j, value in enumerate(values):
            color = colors.get(j)
            cls = f"data {color} colored" if color else "data muted"
            if emphasis and color:
                cls += " emphasis"
                self.path(f"M{x + j * step + 10} {y + height - 2}h{step - 20}", f"{color} trace")
            self.math(x + (j + 0.5) * step, y + height / 2 + 6, str(value), cls, "middle")

    def mask(self, x, y, values, colors, *, step=60.5):
        """Numeric validity with a small, redundant filled/empty bit encoding."""
        for j, value in enumerate(values):
            center = x + (j + 0.5) * step
            color = colors.get(j)
            self.text(center, y, value, f"{color} colored" if color else "muted", "middle")
            self.rect(center - 12, y + 9, 24, 4, f"{color} mask-bit" if value else "mask-zero")

    def cache_rows(self, x, y):
        """Two indexed cache matrices; each row glyph represents a feature vector."""
        for offset, label in ((0, "K"), (52, "V")):
            self.text(x + offset + 19, y - 12, label, "math", "middle")
            for dy, color in ((0, None), (36, "teal"), (58, "blue")):
                self.rect(
                    x + offset,
                    y + dy,
                    38,
                    16,
                    f"{color} memory-active" if color else "memory",
                )
                for col in range(1, 4):
                    self.path(f"M{x + offset + col * 9.5} {y + dy}v16", "array-divider")
            for dy in (22, 26, 30):
                self.circle(x + offset + 19, y + dy, 0.85, "junction")
        for dy, value in ((14, "0"), (50, "12"), (72, "14")):
            self.text(x + 120, y + dy, value, "muted", "end")

    def operation(self, x, y, width, height, value):
        self.rect(x, y, width, height, "outline", 2)
        self.text(x + width / 2, y + height / 2 + 6, value, "", "middle")

    def bracket(self, x, y, width, value, color=None):
        self.path(f"M{x} {y}v5h{width}v-5", f"{color} trace" if color else "edge")
        self.text(x + width / 2, y + 24, value, "muted", "middle")

    def save(self, name):
        (ROOT / "assets" / name).write_text("\n".join(self.items + ["</svg>", ""]))


def render_overview():
    s = SVG(
        544,
        "Ownership, paged translation and deterministic compaction",
        "Three panels expand one call on rank zero with two ranks, interleave one, page size 64 "
        "and request zero. Panel a classifies global positions [8,5,130,7,-1,262] by owner, "
        "keeps rank zero, and deinterleaves to local positions [4,-,65,-,-,131]. The valid mask "
        "is [1,0,1,0,0,1]. Panel b shows request page table P=[11,2,7,5], the surviving "
        "page-offset pairs (0,4),(1,1),(2,3), and their physical addresses 708,129,451. "
        "Panel c computes inclusive validity prefixes minus one; only valid destinations "
        "0,1,2 are used. Colored routes carry each surviving address to its stable output "
        "position, yielding [708,129,451,-1,-1,-1] and count 3. Filled and empty bit marks "
        "reinforce numeric validity. This illustrates multi-rank front compaction; the "
        "single-rank and non-compacting paths preserve columns. SILKern returns addresses "
        "and counts; K/V gathering and attention remain consumer operations.",
    )
    active = {0: "blue", 2: "teal", 5: "copper"}
    compact = {0: "blue", 1: "teal", 2: "copper"}
    s.path("M351 16V524M736 16V430M736 482V524", "rule")
    s.text(24, 36, "a  Ownership and coordinates", "heading")
    s.text(375, 36, "b  Paged addresses", "heading")
    s.text(760, 36, "c  Stable compaction", "heading")

    s.text(24, 84, "Global positions")
    s.text(24, 130, "t", "math")
    s.array(85, 104, [8, 5, 130, 7, "−1", 262], active, step=40.5)
    s.path("M334 124H341V407H333", arrow=True)
    s.text(24, 218, "Owner")
    s.array(85, 192, [0, 1, 0, 1, "—", 0], active, step=40.5)
    for j in (0, 1, 2, 3, 5):
        x = 85 + (j + 0.5) * 40.5
        s.path(f"M{x} 150V186", f"{active[j]} trace" if j in active else "edge", True)
    s.path("M193 238V258", arrow=True)
    s.operation(58, 264, 270, 40, "Keep rank 0")
    s.path("M193 310V320", arrow=True)
    s.text(24, 346, "v", "math")
    s.mask(85, 346, [1, 0, 1, 0, 0, 1], active, step=40.5)
    s.path("M193 365V383", arrow=True)
    s.operation(58, 389, 270, 36, "Deinterleave owned positions")
    s.path("M193 429V442", arrow=True)
    s.text(24, 474, "ℓ", "math")
    s.array(85, 448, [4, "—", 65, "—", "—", 131], active, step=40.5)

    s.text(375, 84, "Request page table")
    s.text(375, 136, "b", "math")
    s.array(424, 110, [0, 1, 2, 3], compact, step=65)
    s.text(375, 200, "P[b]")
    s.array(424, 174, [11, 2, 7, 5], compact, step=65)
    for j, color in compact.items():
        x = 424 + (j + 0.5) * 65
        s.path(f"M{x} 155V168", f"{color} trace", True)
    s.text(375, 272, "(b, δ) = divmod(ℓ, S)")
    s.array(375, 294, ["(0, 4)", "(1, 1)", "(2, 3)"], compact, step=112)
    for j, color in compact.items():
        x = 375 + (j + 0.5) * 112
        s.path(f"M{x} 340V365", f"{color} trace", True)
    s.path("M693 194H718V393H690", arrow=True)
    s.operation(412, 372, 272, 42, "P[b] · S + δ")
    for j, color in active.items():
        x = 424 + (j + 0.5) * 43.5
        s.path(f"M{x} 420V442", f"{color} trace", True)
    s.text(375, 474, "a", "math")
    s.array(424, 448, [708, "—", 129, "—", "—", 451], active, step=43.5)

    s.text(760, 84, "Validity and prefix destinations")
    s.text(760, 130, "v", "math")
    s.mask(800, 130, [1, 0, 1, 0, 0, 1], active, step=48)
    s.path("M944 151V184", arrow=True)
    s.operation(813, 190, 254, 40, "p = cumsum(v) − 1")
    s.path("M944 236V254", arrow=True)
    s.text(760, 286, "p", "math")
    s.array(800, 260, [0, "—", 1, "—", "—", 2], active, step=48)
    for source, dest, color in ((0, 0, "blue"), (2, 1, "teal"), (5, 2, "copper")):
        a, b = 800 + (source + 0.5) * 48, 800 + (dest + 0.5) * 48
        s.path(f"M{a} 306C{a} 367 {b} 373 {b} 442", f"{color} trace", True)
    s.math(1030, 400, "s[p_{j}] ← a_{j}", "", "middle")
    s.path("M691 468H794", arrow=True)
    s.array(800, 448, [708, 129, 451, "−1", "−1", "−1"], compact, step=48, emphasis=True)
    s.bracket(800, 498, 144, "count = 3")
    s.bracket(944, 498, 144, "padding")
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
        (284, "Page, offset", ["(0, 4)", "—", "(1, 1)", "—", "—", "(2, 3)"]),
        (352, "Physical slot  a", [708, "—", 129, "—", "—", 451]),
    ):
        s.text(24, y + 26, label)
        s.array(230, y, values, active, step=75, emphasis=y == 352)
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
                [f"{vals[i][0]}, {vals[i][1]}" for i in order],
                {j: COLORS[i] for j, i in enumerate(order)},
                step=96,
            )
        for identity in range(4):
            a = xs[identity] + 48
            b = xs[orders[0].index(identity)] + 48
            s.path(f"M{a} 139C{a} 165 {b} 163 {b} 193", f"{COLORS[identity]} trace", True)
    s.save("fig-problem.svg")


def render_consumer():
    s = SVG(
        596,
        "Layout-aware masking, paged gathering and a shared softmax",
        "Panel a compares compact slots [14,12,-1,-1] with mask [1,1,0,0] to preserved "
        "columns [14,-1,12,-1] with mask [1,0,1,0]. Both counts are 2. Panel b substitutes "
        "address zero for invalid positions, gathers K/V, then selects zero for invalid "
        "gathered values. Unused cache slot zero may contain NaN, so both masks are needed. "
        "Indexed K and V matrices show placeholder row zero and selected rows 12 and 14. "
        "Each four-cell row is a schematic feature vector, not its actual dimension. "
        "Solid routes carry values and dashed routes carry validity. Panel c masks invalid logits to "
        "negative infinity, computes one maximum over all logical shards and positions, "
        "and selects zero for invalid weights. An empty selection sets the maximum to zero. "
        "Sums N of weighted values and Z of weights are combined in one division; if Z=0, "
        "the output is zero. Valid arithmetic must be finite and representable. This "
        "illustrates a one-device consumer, not a distributed runtime.",
    )
    s.path("M351 16V576M736 16V576", "rule")
    s.text(24, 36, "a  Layout and validity", "heading")
    s.text(375, 36, "b  Masked gather", "heading")
    s.text(760, 36, "c  Shared normalization", "heading")
    for y, title, values, mask, active in (
        (
            94,
            "Front compaction · n = 2",
            [14, 12, "−1", "−1"],
            [1, 1, 0, 0],
            {0: "blue", 1: "teal"},
        ),
        (
            358,
            "Preserve columns · n = 2",
            [14, "−1", 12, "−1"],
            [1, 0, 1, 0],
            {0: "blue", 2: "teal"},
        ),
    ):
        s.text(24, y, title, "subhead")
        s.text(24, y + 56, "Slots")
        s.text(24, y + 116, "Mask")
        s.array(85, y + 30, values, active, step=60.5)
        s.mask(85, y + 116, mask, active)
        for j in active:
            x = 115.25 + j * 60.5
            s.path(f"M{x} {y + 78}v12", f"{active[j]} trace")
        s.math(
            201,
            y + 174,
            "v_{j} = [j < n]" if y == 94 else "v_{j} = input-column validity",
            "",
            "middle",
        )
    s.path("M24 310H327", "rule")

    s.text(378, 90, "s")
    active = {0: "blue", 2: "teal"}
    s.array(426, 64, [14, "−1", 12, "−1"], active, step=61, height=36)
    s.text(378, 139, "v")
    s.mask(426, 139, [1, 0, 1, 0], active, step=61)
    # The two buses are offset: s supplies addresses, v controls both selections.
    s.path("M412 82H407V201H422", arrow=True)
    s.path("M393 132H397V477H422M548 158V174", "control", True)
    s.operation(426, 180, 244, 42, "where(v, s, 0)")
    s.path("M548 226V243", arrow=True)
    s.array(426, 249, [14, 0, 12, 0], active, step=61, height=36)
    s.path("M501 291V318", arrow=True)
    s.operation(426, 324, 150, 40, "Gather K / V")
    s.cache_rows(606, 322)
    s.path("M602 344H581", arrow=True)
    s.path("M501 370V396", arrow=True)
    s.array(426, 402, ["x_{14}", "x_{0}", "x_{12}", "x_{0}"], active, step=61, height=36)
    s.path("M548 442V451", arrow=True)
    s.operation(426, 456, 244, 42, "where(v, x, 0)")
    s.path("M548 502V530", arrow=True)
    s.array(426, 536, ["x_{14}", 0, "x_{12}", 0], active, step=61, height=44, emphasis=True)

    s.math(927, 90, "z_{rj} = q · K_{rj} / √d", "", "middle")
    s.math(927, 125, "Invalid z_{rj} ← −∞", "muted", "middle")
    # A reduction bus denotes a maximum over every shard and selected position.
    s.path("M801 154V171H1053V154")
    s.path("M927 171V199", arrow=True)
    s.circle(927, 171, 2.5, "junction")
    s.math(927, 228, "m = max_{r,j} z_{rj}", "", "middle")
    s.path("M927 244V287", arrow=True)
    s.math(927, 319, "w_{rj} = v_{rj} ? exp(z_{rj} − m) : 0", "", "middle")
    s.path("M927 335V358H830V387M927 358H1022V387", arrow=True)
    s.circle(927, 358, 2.5, "junction")
    s.math(830, 416, "N = Σ_{r,j} w_{rj} V_{rj}", "", "middle")
    s.math(1022, 416, "Z = Σ_{r,j} w_{rj}", "", "middle")
    s.path("M830 435V470H927M1022 435V470H927")
    s.path("M927 470V492", arrow=True)
    s.circle(927, 470, 2.5, "junction")
    s.text(909, 530, "y =", "blue colored", "end")
    s.text(942, 515, "N", "blue colored", "middle")
    s.path("M926 524H958", "blue trace")
    s.text(942, 547, "Z", "blue colored", "middle")
    s.text(927, 579, "Z = 0 ⇒ y = 0", "muted", "middle")
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
    for compact, expected in (
        (True, [[14, 12, -1, -1]]),
        (False, [[14, -1, 12, -1]]),
    ):
        assert localize_reference(
            [0],
            [[3]],
            [[4, -1, 0, 99]],
            block_size=4,
            dcp_size=2,
            dcp_rank=0,
            compact_valid_to_front=compact,
        ) == (expected, [2])
    render_overview()
    render_contract()
    render_order()
    render_consumer()


if __name__ == "__main__":
    main()
