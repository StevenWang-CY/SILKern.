"""Render the explanatory SVGs without raster assets or hardware execution.

The worked examples are checked against the pure-Python contract before export.
Survivor hues follow identity through every stage; neutral ink and adaptive
tokens keep the figures legible when embedded in light or dark Markdown pages.
"""

from __future__ import annotations

from figure_style import NARROW, NOTE, VALUE, WIDTH, Figure

from silkern import localize_reference

HAIR_RULE = 0.9
ROW = [8, 5, 130, 7, -1, 262]
TABLE = [11, 2, 7, 5]
SURVIVORS = {0: "blue", 2: "orange", 5: "green"}
COMPACT = {0: "blue", 1: "orange", 2: "green"}
MINUS = "−1"
COMPACTED = [708, 129, 451, MINUS, MINUS, MINUS]
PRESERVED = [708, MINUS, 129, MINUS, MINUS, 451]
TRANSLATION = [
    ("global position $t$", ROW[:4] + [MINUS, ROW[5]]),
    ("owner $o$", [0, 1, 0, 1, "—", 0]),
    ("valid $v$", [1, 0, 1, 0, 0, 1]),
    ("local position $\\ell$", [4, "—", 65, "—", "—", 131]),
    ("page, offset $(b, \\delta)$", ["(0, 4)", "—", "(1, 1)", "—", "—", "(2, 3)"]),
    ("physical page $P[q, b]$", [11, "—", 2, "—", "—", 7]),
    ("physical slot $a$", [708, "—", 129, "—", "—", 451]),
]
EQUATIONS = [
    ("o", "\\lfloor t/I \\rfloor \\bmod D"),
    ("v", "[t \\ge 0]\\,[o = r]"),
    ("\\ell", "\\lfloor t/(DI) \\rfloor\\, I + t \\bmod I"),
    ("(b, \\delta)", "\\mathrm{divmod}(\\ell, S)"),
    ("a", "P[q, b] \\cdot S + \\delta"),
]
DOT = "·"
TILES = [(704, 705), (132, 133), (448, 896), (897, 260)]
TILE_HUES = ("blue", "orange", "green", "violet")
ATOMIC = [[1, 0, 3, 2], [0, 2, 1, 3], [2, 3, 0, 1]]


def _worked_row(rank=0, ranks=2, interleave=1, page=64):
    """Recompute the figure's table from the contract's equations, column by column."""
    rows = {key: [] for key in ("o", "v", "l", "bd", "p", "a")}
    for t in ROW:
        owner = (t // interleave) % ranks if t >= 0 else None
        valid = int(owner == rank)
        local = (t // (ranks * interleave)) * interleave + t % interleave if valid else None
        b, d = divmod(local, page) if valid else (None, None)
        rows["o"].append("—" if owner is None else owner)
        rows["v"].append(valid)
        rows["l"].append(local if valid else "—")
        rows["bd"].append(f"({b}, {d})" if valid else "—")
        rows["p"].append(TABLE[b] if valid else "—")
        rows["a"].append(TABLE[b] * page + d if valid else "—")
    return rows


def render_overview():
    """Figure 1: ownership, logical-to-physical page translation, and compaction."""
    s = Figure(WIDTH, 344, "From global positions to stable cache addresses", "")
    s.description = (
        "Three panels follow one call on rank 0 of two ranks, interleave 1, page size 64 and "
        "request 0. Panel a: selector row t = [8, 5, 130, 7, -1, 262]; owners are "
        "[0, 1, 0, 1, none, 0], so columns 0, 2 and 5 survive and deinterleave to local "
        "positions 4, 65 and 131. Panel b: local positions fall in logical pages 0, 1 and 2 at "
        "offsets 4, 1 and 3; the request page table [11, 2, 7, 5] sends them to physical blocks "
        "11, 2 and 7, giving slots 708, 129 and 451. Panel c: inclusive validity prefixes minus "
        "one give destinations 0, 1 and 2, so the output is [708, 129, 451, -1, -1, -1] with "
        "count 3, in selector order. Color follows each surviving token."
    )

    r1, r2, r3 = 70, 144, 232  # strip rows shared by panels a and c
    # (a) Ownership ----------------------------------------------------------
    x0, cell = 70, 40
    cols = [x0 + (j + 0.5) * cell for j in range(6)]
    for j, x in enumerate(cols):
        hue = SURVIVORS.get(j)
        if j != 4:
            s.line([(x, r1 + 32), (x, r2)], cls=f"{hue} trace" if hue else "wire", collide=False)
        if hue:
            s.line([(x, r2 + 26), (x, r3 - 1)], cls=f"{hue} trace", arrow=True, collide=False)
    s.text(
        x0 + 3 * cell,
        56,
        "rank $r = 0$ of $D = 2$,  $I = 1$",
        size=NOTE,
        anchor="middle",
    )
    s.band(x0, 110, 6 * cell, "$\\lfloor t/I \\rfloor \\bmod D$")
    s.band(x0, 188, 6 * cell, "$\\lfloor t/(DI) \\rfloor\\, I + t \\bmod I$")
    s.text(x0 - 12, r1 + 22, "$t$", anchor="end")
    s.text(x0 - 12, r2 + 19, "$o$", anchor="end")
    s.text(x0 - 12, r3 + 22, "$\\ell$", anchor="end")
    s.strip(x0, r1, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SURVIVORS, quiet=(4,))
    s.strip(x0, r2, [0, 1, 0, 1, DOT, 0], cell=cell, height=26, hues=SURVIVORS, quiet=(4,))
    s.strip(x0, r3, [4, DOT, 65, DOT, DOT, 131], cell=cell, hues=SURVIVORS, quiet=(1, 3, 4))

    # (b) Paged translation ----------------------------------------------------
    pitch, bx, slot = 48, 356, 13
    rows = [r1 + pitch * k for k in range(4)]
    offsets = {0: (4, "blue"), 1: (1, "orange"), 2: (3, "green")}
    physical = sorted(TABLE)
    tx, px = bx + 134, bx + 284  # page-table column; physical blocks
    s.rect(px - 68, 40, 234, 238, "zone")
    s.text(bx + 52, 56, "logical page $b$", size=NOTE, anchor="middle")
    s.text(tx + 21, 56, "$P[b]$", size=NOTE, anchor="middle")
    s.text(px - 56, 56, "physical KV blocks", size=NOTE)
    s.text(px + 118, 56, "slot $a$", size=NOTE)
    for b, y in enumerate(rows):
        offset, hue = offsets.get(b, (None, None))
        s.text(bx - 8, y + 30, str(b), size=NOTE, anchor="end", cls="mute")
        s.strip(
            bx,
            y + 12,
            [""] * 7,
            cell=slot,
            height=24,
            open_end=True,
            hues={offset: hue} if hue else {},
        )
        s.line(
            [(bx + 112, y + 24), (tx - 2, y + 24)],
            cls=f"{hue} trace" if hue else "wire",
            arrow=True,
        )
    s.column(tx, r1, TABLE, width=42, cell=pitch, hues={b: hue for b, (_, hue) in offsets.items()})
    # Cased curves, drawn gray first and blue last, read as passing over one another.
    for b in reversed(range(len(TABLE))):
        hue = offsets.get(b, (None, None))[1]
        k = physical.index(TABLE[b])
        s.curve(
            (tx + 46, rows[b] + 24),
            (px - 34, rows[k] + 24),
            horizontal=True,
            arrow=True,
            casing=True,
            cls=f"{hue} trace" if hue else "wire",
        )
    for k, (block, y) in enumerate(zip(physical, rows, strict=True)):
        offset, hue = offsets.get(TABLE.index(block), (None, None))
        s.text(px - 6, y + 30, str(block), size=NOTE, anchor="end", cls="mute")
        s.strip(
            px,
            y + 12,
            [""] * 7,
            cell=slot,
            height=24,
            open_end=True,
            hues={offset: hue} if hue else {},
        )
        if hue:
            s.text(
                px + 118,
                y + 31,
                str(block * 64 + offset),
                size=VALUE,
                cls=f"{hue} tint",
                weight="semibold",
            )
        if k < 3:
            for dy in (-4.5, 0, 4.5):
                s.circle(px - 15, y + 48 + dy, 1.3, "dot")
    bc = (bx - 20 + px + 166) / 2  # visual center of panel b
    s.text(bc, 300, "$a = P[b] \\cdot S + \\delta$,  $S = 64$", size=NOTE, anchor="middle")

    # (c) Stable compaction ------------------------------------------------------
    x1, cell = 848, 41
    ccols = [x1 + (j + 0.5) * cell for j in range(6)]
    for j, hue in SURVIVORS.items():
        s.line([(ccols[j], r1 + 32), (ccols[j], r2)], cls=f"{hue} trace", collide=False)
    for (j, hue), dest in zip(SURVIVORS.items(), range(3), strict=True):
        s.curve((ccols[j], r2 + 26), (ccols[dest], r3 - 1), cls=f"{hue} trace", collide=False)
    s.text(x1 + 3 * cell, 56, "valid $v = [o = r]$", size=NOTE, anchor="middle")
    s.band(x1, 110, 6 * cell, "$\\mathrm{cumsum}(v) - 1$")
    s.text(x1 - 12, r1 + 22, "$a$", anchor="end")
    s.text(x1 - 12, r2 + 19, "$p$", anchor="end")
    s.text(x1 - 12, r3 + 22, "$s$", anchor="end")
    s.strip(x1, r1, [708, DOT, 129, DOT, DOT, 451], cell=cell, hues=SURVIVORS, quiet=(1, 3, 4))
    s.strip(x1, r2, [0, DOT, 1, DOT, DOT, 2], cell=cell, height=26, hues=SURVIVORS, quiet=(1, 3, 4))
    s.strip(x1, r3, [708, 129, 451, MINUS, MINUS, MINUS], cell=cell, hues=COMPACT, bold=True)
    s.brace(x1 + 2, x1 + 3 * cell - 2, r3 + 38)
    s.brace(x1 + 3 * cell + 2, x1 + 6 * cell - 2, r3 + 38)
    s.text(x1 + 1.5 * cell, r3 + 64, "count $= 3$", size=NOTE, anchor="middle")
    s.text(x1 + 4.5 * cell, r3 + 64, "padding", size=NOTE, anchor="middle")

    for cx, letter, title in (
        (x0 + 3 * 40, "a", "Ownership"),
        (bc, "b", "Paged translation"),
        (x1 + 3 * cell, "c", "Stable compaction"),
    ):
        s.subcaption(cx, 326, letter, title)
    s.save(
        "fig-platforms.svg",
        panels=(
            (22, 36, 306, 236, 6),
            (22, 306, 306, 30),
            (334, 36, 472, 304),
            (808, 36, 300, 304),
        ),
    )


def render_contract():
    """Figure 2: the worked row as a ruled table, with equations and both layouts."""
    s = Figure(WIDTH, 400, "Coordinate translation and the two output layouts", "")
    s.description = (
        "Panel a is a table over the six input columns for rank 0 of two ranks, interleave 1, "
        "page size 64 and request page table [11, 2, 7, 5]. Global positions "
        "[8, 5, 130, 7, -1, 262] have owners [0, 1, 0, 1, none, 0] and validity "
        "[1, 0, 1, 0, 0, 1]. Columns 0, 2 and 5 survive with local positions 4, 65 and 131, "
        "page-offset pairs (0, 4), (1, 1) and (2, 3), physical pages 11, 2 and 7, and physical "
        "slots 708, 129 and 451. Panel b states the general equations for nonnegative t and the "
        "two output layouts: front compaction [708, 129, 451, -1, -1, -1] and preserved columns "
        "[708, -1, 129, -1, -1, 451]. Both counts are 3; only the compacted row has a valid "
        "three-element prefix."
    )
    left, x0, width = 24, 300, 66
    centers = [x0 + (j + 0.5) * width for j in range(6)]
    top, pitch = 52, 38
    rows = TRANSLATION
    bottom = top + 34 + pitch * len(rows) + 4
    for j, hue in SURVIVORS.items():
        s.rect(x0 + j * width + 3, top + 2, width - 6, bottom - top - 4, f"{hue} area")
    s.rule(left, x0 + 6 * width, top, weight=1.4)
    s.rule(left, x0 + 6 * width, top + 34, weight=HAIR_RULE)
    s.rule(left, x0 + 6 * width, bottom, weight=1.4)
    s.text(left, top + 23, "column $j$")
    for j, cx in enumerate(centers):
        s.text(cx, top + 23, str(j), anchor="middle")
    for k, (label, values) in enumerate(rows):
        y = top + 34 + pitch * (k + 0.5) + 6.5
        s.text(left, y, label)
        for j, (cx, value) in enumerate(zip(centers, values, strict=True)):
            hue = SURVIVORS.get(j)
            final = k == len(rows) - 1
            s.text(
                cx,
                y,
                value,
                size=VALUE,
                anchor="middle",
                cls="mute" if value in ("—", MINUS) else "",
                weight="semibold" if (final and hue) else "regular",
            )

    xe = 820
    s.equations(xe, 80, 34, EQUATIONS)
    s.rule(752, 1096, 238, weight=HAIR_RULE, cls="rule")
    sx, cell = 752, 49
    for y, label, values, hues, bold in (
        (262, "front compaction", COMPACTED, COMPACT, True),
        (326, "column-preserving", PRESERVED, SURVIVORS, False),
    ):
        s.text(sx, y, label, size=NOTE)
        s.strip(sx, y + 8, values, cell=cell, hues=hues, bold=bold)
        s.text(1096, y + 29, "$n = 3$", size=NOTE, anchor="end")
    s.subcaption((left + x0 + 6 * width) / 2, 390, "a", "Coordinate translation")
    s.subcaption(924, 390, "b", "Equations and layouts")
    s.save("fig-contract.svg")


def render_contract_narrow():
    """Phone layout of Figure 2: the table transposed so each input column is a row."""
    s = Figure(NARROW, 700, "Coordinate translation and output layouts, narrow layout", "")
    s.description = (
        "The same worked row as the wide figure, transposed: each input column j is a table row. "
        "Columns 0, 2 and 5 survive: t = 8, 130, 262 map to local positions 4, 65, 131, "
        "page-offset pairs (0, 4), (1, 1), (2, 3), physical pages 11, 2, 7 and physical slots "
        "708, 129, 451. The equations and both output layouts follow."
    )
    heads = ["$j$", "$t$", "$o$", "$v$", "$\\ell$", "$(b, \\delta)$", "$P$", "$a$"]
    xs = [34, 80, 124, 160, 202, 258, 318, 370]
    top, pitch = 24, 34
    columns = [[j] + [values[j] for _, values in TRANSLATION] for j in range(6)]
    bottom = top + 34 + pitch * 6 + 4
    for j, hue in SURVIVORS.items():
        y = top + 34 + pitch * j + 2
        s.rect(16, y, NARROW - 32, pitch, f"{hue} area")
    s.rule(16, NARROW - 16, top, weight=1.4)
    s.rule(16, NARROW - 16, top + 34, weight=HAIR_RULE)
    s.rule(16, NARROW - 16, bottom, weight=1.4)
    for x, head in zip(xs, heads, strict=True):
        s.text(x, top + 23, head, size=VALUE, anchor="middle")
    for j, values in enumerate(columns):
        hue = SURVIVORS.get(j)
        y = top + 34 + pitch * (j + 0.5) + 6.5
        for k, (x, value) in enumerate(zip(xs, values, strict=True)):
            cls = "mute" if value in ("—", MINUS) else ""
            s.text(
                x,
                y,
                value,
                size=VALUE,
                anchor="middle",
                cls=cls,
                weight="semibold" if (hue and k == len(values) - 1) else "regular",
            )
    s.subcaption(NARROW / 2, bottom + 34, "a", "Coordinate translation")
    xe = 150
    s.equations(xe, bottom + 92, 34, EQUATIONS)
    y = bottom + 92 + 34 * 4 + 30
    s.rule(16, NARROW - 16, y - 2, weight=HAIR_RULE, cls="rule")
    for dy, label, values, hues, bold in (
        (24, "front compaction  $n = 3$", COMPACTED, COMPACT, True),
        (88, "column-preserving  $n = 3$", PRESERVED, SURVIVORS, False),
    ):
        s.text(30, y + dy, label, size=NOTE)
        s.strip(30, y + dy + 8, values, cell=60, hues=hues, bold=bold)
    s.subcaption(NARROW / 2, y + 160, "b", "Equations and layouts")
    s.height = int(y + 172)
    s.save("fig-contract-narrow.svg")


def render_consumer():
    """Figure 5: layout masks, masked paged gathering, and shared normalization."""
    s = Figure(WIDTH, 452, "Layout-aware masking, paged gathering and a shared softmax", "")
    s.description = (
        "Panel a compares compact slots [14, 12, -1, -1] with mask [1, 1, 0, 0] to preserved "
        "columns [14, -1, 12, -1] with mask [1, 0, 1, 0]; both counts are 2. Panel b replaces "
        "invalid addresses with 0, gathers K and V rows 14, 0, 12 and 0 from the paged cache, then "
        "selects zero for invalid gathered values, because placeholder row 0 may contain NaN. "
        "Solid lines carry values; the dashed line carries the mask to both selections. Panel c "
        "masks invalid logits to negative infinity, takes one maximum m over every logical shard "
        "r and position j, forms weights w = v exp(z - m), sums N = sum w V and Z = sum w over the "
        "same set, and returns y = N / Z, or zero when Z = 0. This illustrates a one-device "
        "consumer, not a distributed runtime."
    )
    pair = {0: "blue", 1: "orange"}
    spread = {0: "blue", 2: "orange"}

    # (a) Layout and validity ------------------------------------------------
    x0, cell = 64, 56
    for y, title, slots, mask, hues, rule in (
        (
            58,
            "front compaction",
            [14, 12, MINUS, MINUS],
            [1, 1, 0, 0],
            pair,
            "$v_j = [\\, j < n \\,]$",
        ),
        (
            232,
            "column-preserving",
            [14, MINUS, 12, MINUS],
            [1, 0, 1, 0],
            spread,
            "$v_j$ = input-column validity",
        ),
    ):
        s.text(x0, y, title, size=NOTE)
        s.text(x0 + 4 * cell, y, "$n = 2$", size=NOTE, anchor="end")
        s.text(x0 - 12, y + 32, "$s$", anchor="end")
        s.text(x0 - 12, y + 70, "$v$", anchor="end")
        s.strip(
            x0,
            y + 10,
            slots,
            cell=cell,
            hues=hues,
            quiet=tuple(j for j in range(4) if j not in hues),
        )
        s.strip(x0, y + 52, mask, cell=cell, height=26, hues=hues)
        s.text(x0 + 2 * cell, y + 118, rule, anchor="middle", cls="soft")
    s.rule(28, x0 + 4 * cell + 8, 202, weight=HAIR_RULE, cls="rule")

    # (b) Masked gather --------------------------------------------------------
    bx, bc = 400, 50
    cols = [bx + (j + 0.5) * bc for j in range(4)]
    v_y, s_y, w1, s0, g, x_y, w2, out = 58, 96, 146, 188, 238, 280, 330, 372
    for j, x in enumerate(cols):
        hue = spread.get(j)
        cls = f"{hue} trace" if hue else "wire"
        s.line([(x, s_y + 32), (x, s0)], cls=cls, collide=False)
        s.line([(x, s0 + 32), (x, x_y)], cls=cls, collide=False)
        s.line([(x, x_y + 32), (x, out - 1)], cls=cls, arrow=True, collide=False)
    # The mask reaches both selections through one dashed bus.
    bus = bx - 16
    s.line(
        [(bx, v_y + 13), (bus, v_y + 13), (bus, w2 + 12), (bx - 1, w2 + 12)],
        cls="mask",
        arrow=True,
        radius=4,
        collide=False,
    )
    s.line([(bus, w1 + 12), (bx - 1, w1 + 12)], cls="mask", arrow=True, collide=False)
    s.circle(bus, w1 + 12, 2.2, "dot")
    s.band(bx, w1 - 1, 4 * bc, "$\\mathrm{where}(v, s, 0)$")
    s.band(bx, g - 1, 4 * bc, "gather $K$, $V$")
    s.band(bx, w2 - 1, 4 * bc, "$\\mathrm{where}(v, x, 0)$")
    s.text(bus - 10, v_y + 19, "$v$", anchor="end")
    s.text(bus - 10, s_y + 22, "$s$", anchor="end")
    s.strip(bx, v_y, [1, 0, 1, 0], cell=bc, height=26, hues=spread)
    s.strip(bx, s_y, [14, MINUS, 12, MINUS], cell=bc, hues=spread, quiet=(1, 3))
    s.strip(bx, s0, [14, 0, 12, 0], cell=bc, hues=spread)
    s.strip(bx, x_y, ["$x_{14}$", "$x_0$", "$x_{12}$", "$x_0$"], cell=bc, hues=spread)
    s.strip(bx, out, ["$x_{14}$", 0, "$x_{12}$", 0], cell=bc, hues=spread, bold=True)
    # Paged cache: indexed K and V rows; row 0 is the placeholder address.
    kx = 664
    s.rect(kx - 30, 166, 150, 144, "zone")
    s.text(kx + 45, 188, "paged KV cache", size=NOTE, anchor="middle")
    for dx, name in ((0, "K"), (60, "V")):
        s.text(kx + dx + 24, 212, f"${name}$", size=NOTE, anchor="middle")
        for row, hue in enumerate((None, "orange", "blue")):
            y = 222 + row * 22 + (12 if row else 0)
            s.strip(
                kx + dx,
                y,
                [""] * 4,
                cell=12,
                height=18,
                hues={k: hue for k in range(4)} if hue else {},
            )
    for row, slot in enumerate((0, 12, 14)):
        s.text(
            kx - 8,
            237 + row * 22 + (12 if row else 0),
            str(slot),
            size=NOTE,
            anchor="end",
            cls="mute",
        )
    for dy in (-4.5, 0, 4.5):
        s.circle(kx + 24, 248 + dy, 1.3, "dot")
        s.circle(kx + 84, 248 + dy, 1.3, "dot")
    s.line([(kx - 32, g + 12), (bx + 4 * bc + 1, g + 12)], arrow=True, collide=False)

    # (c) Shared normalization -----------------------------------------------
    cx = 950
    s.text(cx, 76, "$z_{rj} = q \\cdot K_{rj} / \\sqrt{d}$", anchor="middle")
    s.text(
        cx,
        104,
        "$z_{rj} \\gets -\\infty$ where $v_{rj} = 0$",
        size=NOTE,
        anchor="middle",
        cls="soft",
    )
    s.line([(cx, 114), (cx, 140)], cls="flow", arrow=True)
    s.text(cx, 164, "$m = \\max_{r, j} z_{rj}$", anchor="middle")
    s.line([(cx, 176), (cx, 202)], cls="flow", arrow=True)
    s.text(cx, 226, "$w_{rj} = v_{rj} \\exp(z_{rj} - m)$", anchor="middle")
    s.line([(cx, 238), (cx, 252), (cx - 74, 252), (cx - 74, 272)], cls="flow", arrow=True,
           radius=5)
    s.line([(cx, 252), (cx + 74, 252), (cx + 74, 272)], cls="flow", arrow=True, radius=5)
    s.circle(cx, 252, 2.2, "flow-head")
    s.text(cx - 74, 296, "$N = \\sum_{r, j} w_{rj} V_{rj}$", anchor="middle")
    s.text(cx + 74, 296, "$Z = \\sum_{r, j} w_{rj}$", anchor="middle")
    s.line([(cx - 74, 308), (cx - 74, 322), (cx + 74, 322), (cx + 74, 308)], cls="flow",
           radius=5)
    s.circle(cx, 322, 2.2, "flow-head")
    s.line([(cx, 322), (cx, 340)], cls="flow", arrow=True)
    s.text(cx - 6, 372, "$y =$", anchor="end", cls="accent tint")
    s.fraction(cx, 372, "$N$", "$Z$", cls="accent tint")
    s.text(cx, 412, "$y = 0$ if $Z = 0$", size=NOTE, anchor="middle", cls="soft")

    for x, letter, title in (
        (x0 + 2 * cell, "a", "Layout and validity"),
        (bx + 2 * bc + 72, "b", "Masked gather"),
        (cx, "c", "Shared normalization"),
    ):
        s.subcaption(x, 442, letter, title)
    s.save(
        "fig-consumer.svg",
        panels=(
            (0, 40, 334, 324, 6),
            (0, 420, 334, 32),
            (338, 40, 460, 412),
            (806, 40, 300, 412),
        ),
    )


def _tile_row(s, x, y, order, cell):
    values = [v for t in order for v in TILES[t]]
    hues = {2 * k + i: TILE_HUES[t] for k, t in enumerate(order) for i in (0, 1)}
    s.strip(x, y, values, cell=cell, height=30, hues=hues, group=2)


def render_order():
    """Figure 6: atomic reservation against stable prefix destinations, over replays."""
    s = Figure(WIDTH, 346, "Completion-order reservation versus deterministic tile prefixes", "")
    s.description = (
        "Tile groups T0 = [704, 705], T1 = [132, 133], T2 = [448, 896] and T3 = [897, 260] "
        "are tracked by four colors. Panel a: atomic reservation claims output offsets in "
        "completion order, so three replays of the same input emit T1, T0, T3, T2; then T0, T2, "
        "T1, T3; then T2, T3, T0, T1. Panel b: offsets from an exclusive scan of tile counts, "
        "0, 2, 4 and 6, emit T0, T1, T2, T3 on every replay. Both paths write the same eight "
        "values and count; curves trace the first replay. These are schematic arrays, not "
        "measured GPU results."
    )
    cell = 46
    for panel, (x, orders, title, note) in enumerate(
        (
            (124, ATOMIC, "Atomic reservation", "offsets claimed in completion order"),
            (
                690,
                [[0, 1, 2, 3]] * 3,
                "Stable prefix destinations",
                "offsets from an exclusive scan: 0, 2, 4, 6",
            ),
        )
    ):
        centers = [x + (2 * k + 1) * cell for k in range(4)]
        s.text(x + 4 * cell, 50, note, size=NOTE, anchor="middle")
        for k, hue in enumerate(TILE_HUES):
            s.text(centers[k], 80, f"$T_{k}$", size=NOTE, anchor="middle", cls=f"{hue} tint")
        s.text(x - 14, 110, "input", anchor="end")
        _tile_row(s, x, 88, [0, 1, 2, 3], cell)
        for r, order in enumerate(orders):
            y = 174 + 46 * r
            s.text(x - 14, y + 22, f"replay {r + 1}", anchor="end")
            _tile_row(s, x, y, order, cell)
        for t in range(4):
            dest = orders[0].index(t)
            s.curve(
                (centers[t], 118), (centers[dest], 173), cls=f"{TILE_HUES[t]} trace", collide=False
            )
        s.subcaption(x + 4 * cell, 334, "ab"[panel], title)
    s.save("fig-problem.svg", panels=((32, 36, 472, 310), (598, 36, 472, 310)))


def main():
    # Every drawn mapping must agree with the contract's equations and the oracle.
    worked = _worked_row()
    assert [values for _, values in TRANSLATION[1:]] == [
        worked[key] for key in ("o", "v", "l", "bd", "p", "a")
    ]
    for compact, expected in ((True, COMPACTED), (False, PRESERVED)):
        assert localize_reference(
            [0],
            [TABLE],
            [ROW],
            block_size=64,
            dcp_size=2,
            dcp_rank=0,
            compact_valid_to_front=compact,
        ) == ([[-1 if v == MINUS else v for v in expected]], [3])
    for compact, expected in ((True, [[14, 12, -1, -1]]), (False, [[14, -1, 12, -1]])):
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
    render_contract_narrow()
    render_consumer()
    render_order()


if __name__ == "__main__":
    main()
