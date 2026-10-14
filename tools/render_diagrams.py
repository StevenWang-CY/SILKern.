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


SELECTION = {0: "blue", 1: "gray", 2: "orange", 3: "gray", 5: "green"}  # by owning rank
# Physical blocks 2, 7 and 11 of each rank's own cache, drawn as rows of offsets 0-4;
# both ranks use the same page table, so rank 1's slots 706 and 707 share block 11.
KV_BLOCKS = (2, 7, 11)
KV_SLOTS = ({708: "blue", 129: "orange", 451: "green"}, {706: "gray", 707: "gray"})
RANK_OUTPUTS = (
    ([708, 129, 451, MINUS, MINUS, MINUS], COMPACT, 3),
    ([706, 707, MINUS, MINUS, MINUS, MINUS], {0: "gray", 1: "gray"}, 2),
)


def _rank_panel(s, x, top, rank, *, width, a, cell=38, kv_cell=13, bw=232, p_input=True):
    """One rank of the system figure, as a panel: SILKern (pale blue), the slots it
    returns, and the attention consumer (white) reading its own paged KV cache.

    ``x`` and ``width`` place the panel, ``a`` is the rank's flow axis and ``top``
    the panel's top edge. Returns the baseline of the partial result.
    """
    values, hues, count = RANK_OUTPUTS[rank]
    s.panel(x, top, width, 248, f"rank {rank}", ground="b")
    y = top + 46
    s.box(a - bw / 2, y, bw, 34, f"SILKern,  $r = {rank}$", kind="ours")
    if p_input:
        s.text(a - bw / 2 - 52, y + 24, "$P$", anchor="end")
        s.flow([(a - bw / 2 - 44, y + 17), (a - bw / 2 - 2, y + 17)])
    s.flow([(a, y + 34), (a, y + 56)])
    y += 56
    s.text(a - 3 * cell - 12, y + 22, f"$s_{rank}$", anchor="end")
    s.base(a - 3 * cell, y, 6 * cell, 30)
    s.strip(a - 3 * cell, y, values, cell=cell, hues=hues,
            quiet=tuple(j for j in range(6) if j not in hues), name=f"s{rank}")
    s.text(a + 3 * cell + 10, y + 21, f"$n_{rank} = {count}$", size=NOTE)
    s.flow([(a, y + 30), (a, y + 54)])
    y += 54
    s.box(a - bw / 2, y, bw, 34, "sparse attention")
    # The rank's own paged KV cache: rows are physical blocks, cells their slots.
    zw = 52 + 5 * kv_cell
    zx, zy = x + width - zw - 18, y - 20
    s.memory(zx, zy, zw, 100)
    s.text(zx + zw / 2, zy + 22, "KV cache", size=NOTE, anchor="middle")
    for k, block in enumerate(KV_BLOCKS):
        cells = {slot % 64: hue for slot, hue in KV_SLOTS[rank].items() if slot // 64 == block}
        row = zy + 34 + 23 * k
        s.text(zx + 32, row + 13, str(block), size=NOTE - 2, anchor="end", cls="mute")
        s.base(zx + 38, row, 5 * kv_cell, 15)
        s.strip(zx + 38, row, [""] * 5, cell=kv_cell, height=15, hues=cells,
                name=f"kv{rank}{k}")
    s.flow([(zx - 2, y + 17), (a + bw / 2 + 2, y + 17)])
    s.flow([(a, y + 34), (a, y + 56)])
    s.text(a, y + 78, f"$(m_{rank}, N_{rank}, Z_{rank})$", anchor="middle")
    return y + 78


def render_system():
    """Hero: one selection row, localized independently on each rank, then consumed."""
    s = Figure(WIDTH, 500, "Where SILKern sits in a context-parallel decode step", "")
    s.description = (
        "A sparse selector picks global positions for one query, t = [8, 5, 130, 7, -1, 262], "
        "and every rank of a two-rank decode receives the same row. Each rank runs SILKern with "
        "its own rank number and the page table P: rank 0 keeps tokens 8, 130 and 262 and "
        "returns physical slots [708, 129, 451, -1, -1, -1] with count 3; rank 1 keeps tokens 5 "
        "and 7 and returns [706, 707, -1, -1, -1, -1] with count 2. Each rank's sparse attention "
        "gathers those slots from its own paged KV cache, drawn as physical blocks 2, 7 and 11, "
        "and produces partial softmax terms (m, N, Z); one step across both ranks combines them "
        "into y = N / Z. SILKern is the pale blue component; each rank's work sits in its own "
        "panel, and arrows carry the data from one component to the next."
    )
    cx = WIDTH / 2
    s.text(cx, 24, "query $q$", anchor="middle")
    s.flow([(cx, 32), (cx, 52)])
    s.funnel(cx, 52, 270, 34, 26, "sparse selector")
    s.flow([(cx, 86), (cx, 108)])
    cell, ty = 46, 108
    s.text(cx - 3 * cell - 12, ty + 22, "$t$", anchor="end")
    s.base(cx - 3 * cell, ty, 6 * cell, 30)
    s.strip(cx - 3 * cell, ty, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SELECTION,
            quiet=(4,), name="t")
    s.text(cx + 3 * cell + 14, ty + 21, "the same row on every rank", size=NOTE, cls="soft")
    panels = ((40, 230), (580, 770))  # (panel left, flow axis) per rank
    width, top, fork = 500, 182, ty + 50
    # Panels first, so the arrows that enter them are drawn over their grounds.
    bottoms = [_rank_panel(s, x, top, rank, width=width, a=a)
               for rank, (x, a) in enumerate(panels)]
    s.flow([(cx, ty + 30), (cx, fork)], arrow=False, collide=False)
    for _, a in panels:
        s.flow([(cx, fork), (a, fork), (a, top + 46)], radius=10, collide=False)
    s.junction(cx, fork)
    combine = top + 248 + 24
    for (_, a), y in zip(panels, bottoms, strict=True):
        s.flow([(a, y + 14), (a, combine)])
    s.box(40, combine, 1040, 36,
          "combine across ranks:  $m = \\max_r m_r$,  "
          "$N = \\sum_r \\, \\exp(m_r - m)\\, N_r$,  $Z = \\sum_r \\, \\exp(m_r - m)\\, Z_r$,  "
          "$y = N / Z$")
    s.height = combine + 36 + 16
    s.save("fig-system.svg")


def render_system_narrow():
    """Phone layout of the system figure: ranks stacked, with buses at both margins."""
    s = Figure(NARROW, 900, "Where SILKern sits in a context-parallel decode step, narrow", "")
    s.description = (
        "The same decode step as the wide figure, stacked for narrow pages. The selector's row "
        "t = [8, 5, 130, 7, -1, 262] reaches rank 0 directly and rank 1 along the right margin. "
        "Rank 0's SILKern returns [708, 129, 451, -1, -1, -1] with count 3 and rank 1's returns "
        "[706, 707, -1, -1, -1, -1] with count 2; each rank's attention gathers from its own KV "
        "cache, and the partial terms meet along the left margin to give y = N / Z."
    )
    cx = NARROW / 2
    s.text(cx - 122, 41, "$q$", anchor="end")
    s.flow([(cx - 114, 34), (cx - 96, 34)])
    s.funnel(cx, 17, 192, 34, 20, "sparse selector")
    s.flow([(cx, 51), (cx, 70)])
    cell = 50
    s.text(cx - 3 * cell - 10, 92, "$t$", anchor="end")
    s.base(cx - 3 * cell, 70, 6 * cell, 30)
    s.strip(cx - 3 * cell, 70, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SELECTION, quiet=(4,),
            name="t")
    s.text(cx + 12, 128, "same row on every rank", size=NOTE, cls="soft")
    x, width, a, bus_r, bus_l = 26, 368, 156, 410, 10
    tops = (162, 450)
    bottoms = [
        _rank_panel(s, x, top, rank, width=width, a=a, cell=33, kv_cell=10, bw=188,
                    p_input=False)
        for rank, top in enumerate(tops)
    ]
    s.flow([(cx, 100), (cx, 140)], arrow=False, collide=False)
    s.flow([(cx, 140), (a, 140), (a, tops[0] + 46)], radius=8, collide=False)
    s.flow([(cx, 140), (bus_r, 140), (bus_r, tops[1] + 63), (a + 94, tops[1] + 63)],
           radius=8, collide=False)
    s.junction(cx, 140)
    join = tops[1] + 248 + 26
    s.flow([(a, bottoms[0] + 14), (a, tops[1] - 12), (bus_l, tops[1] - 12), (bus_l, join + 18),
            (cx - 112, join + 18)], radius=8, collide=False)
    s.flow([(a, bottoms[1] + 14), (a, join)], collide=False)
    s.box(cx - 110, join, 196, 36, "combine partials")
    s.flow([(cx + 86, join + 18), (cx + 116, join + 18)])
    s.text(cx + 122, join + 25, "$y = N / Z$")
    s.height = join + 36 + 16
    s.save("fig-system-narrow.svg")


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
    s = Figure(WIDTH, 360, "From global positions to stable cache addresses", "")
    s.description = (
        "Three panels follow one call on rank 0 of two ranks, interleave 1, page size 64 and "
        "request 0. Panel a: selector row t = [8, 5, 130, 7, -1, 262]; owners are "
        "[0, 1, 0, 1, none, 0], so columns 0, 2 and 5 survive and deinterleave to local "
        "positions 4, 65 and 131. Panel b: local positions fall in logical pages 0, 1 and 2 at "
        "offsets 4, 1 and 3; the request page table [11, 2, 7, 5] sends them to physical blocks "
        "11, 2 and 7, giving slots 708, 129 and 451. Panel c: inclusive validity prefixes minus "
        "one give destinations 0, 1 and 2, so the output is [708, 129, 451, -1, -1, -1] with "
        "count 3, in selector order. Color follows each surviving token; operations are white "
        "boxes, and physical KV memory sits inside a dashed outline."
    )
    dy = 30  # every row sits below its panel's title
    r1, r2, r3 = 70 + dy, 144 + dy, 232 + dy  # strip rows shared by panels a and c
    s.panel(14, 12, 316, 334, "Ownership", "a", ground="a")
    s.panel(340, 12, 472, 334, "Paged translation", "b", ground="b")
    s.panel(822, 12, 286, 334, "Stable compaction", "c", ground="a")

    # (a) Ownership ----------------------------------------------------------
    x0, cell = 70, 40
    cols = [x0 + (j + 0.5) * cell for j in range(6)]
    for j, x in enumerate(cols):
        hue = SURVIVORS.get(j)
        if j != 4:
            s.line([(x, r1 + 32), (x, r2)], cls=f"{hue} trace" if hue else "wire", collide=False)
        if hue:
            s.line([(x, r2 + 26), (x, r3 - 1)], cls=f"{hue} trace", arrow=True, collide=False)
    s.text(x0 + 3 * cell, 56 + dy, "rank $r = 0$ of $D = 2$,  $I = 1$", size=NOTE,
           anchor="middle", cls="soft")
    s.box(x0, 108 + dy, 6 * cell, 30, "$\\lfloor t/I \\rfloor \\bmod D$")
    s.box(x0, 186 + dy, 6 * cell, 30, "$\\lfloor t/(DI) \\rfloor\\, I + t \\bmod I$")
    s.text(x0 - 12, r1 + 22, "$t$", anchor="end")
    s.text(x0 - 12, r2 + 19, "$o$", anchor="end")
    s.text(x0 - 12, r3 + 22, "$\\ell$", anchor="end")
    for y, h in ((r1, 32), (r2, 26), (r3, 32)):
        s.base(x0, y, 6 * cell, h)
    s.strip(x0, r1, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SURVIVORS, quiet=(4,))
    s.strip(x0, r2, [0, 1, 0, 1, DOT, 0], cell=cell, height=26, hues=SURVIVORS, quiet=(4,))
    s.strip(x0, r3, [4, DOT, 65, DOT, DOT, 131], cell=cell, hues=SURVIVORS, quiet=(1, 3, 4))

    # (b) Paged translation ----------------------------------------------------
    pitch, bx, slot = 48, 368, 13
    rows = [r1 + pitch * k for k in range(4)]
    offsets = {0: (4, "blue"), 1: (1, "orange"), 2: (3, "green")}
    physical = sorted(TABLE)
    tx, px = bx + 128, bx + 270  # page-table column; physical blocks
    s.memory(px - 62, 42 + dy, 172, 236)  # physical KV memory; the slot column sits outside
    s.text(bx + 52, 58 + dy, "logical page $b$", size=NOTE, anchor="middle", cls="soft")
    s.text(tx + 21, 58 + dy, "$P[b]$", size=NOTE, anchor="middle", cls="soft")
    s.text(px - 52, 58 + dy, "physical KV blocks", size=NOTE, cls="soft")
    s.text(px + 118, 58 + dy, "slot $a$", size=NOTE, cls="soft")
    for b, y in enumerate(rows):
        offset, hue = offsets.get(b, (None, None))
        s.text(bx - 8, y + 30, str(b), size=NOTE, anchor="end", cls="mute")
        s.base(bx, y + 12, 7 * slot, 24)
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
    s.base(tx, r1, 42, pitch * len(TABLE))
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
            casing="p-casing-b",
            cls=f"{hue} trace" if hue else "wire",
        )
    for k, (block, y) in enumerate(zip(physical, rows, strict=True)):
        offset, hue = offsets.get(TABLE.index(block), (None, None))
        s.text(px - 6, y + 30, str(block), size=NOTE, anchor="end", cls="mute")
        s.base(px, y + 12, 7 * slot, 24)
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
            s.text(px + 118, y + 31, str(block * 64 + offset), size=VALUE, cls=f"{hue} tint")
        if k < 3:
            for ddy in (-4.5, 0, 4.5):
                s.circle(px - 15, y + 48 + ddy, 1.3, "dot")
    bc = (340 + 812) / 2  # centre of panel b
    s.text(bc, 304 + dy, "$a = P[b] \\cdot S + \\delta$,  $S = 64$", size=NOTE,
           anchor="middle")

    # (c) Stable compaction ------------------------------------------------------
    x1, cell = 852, 41
    ccols = [x1 + (j + 0.5) * cell for j in range(6)]
    for j, hue in SURVIVORS.items():
        s.line([(ccols[j], r1 + 32), (ccols[j], r2)], cls=f"{hue} trace", collide=False)
    for (j, hue), dest in zip(SURVIVORS.items(), range(3), strict=True):
        s.curve((ccols[j], r2 + 26), (ccols[dest], r3 - 1), cls=f"{hue} trace", collide=False)
    s.text(x1 + 3 * cell, 56 + dy, "valid $v = [o = r]$", size=NOTE, anchor="middle",
           cls="soft")
    s.box(x1, 108 + dy, 6 * cell, 30, "$\\mathrm{cumsum}(v) - 1$")
    s.text(x1 - 12, r1 + 22, "$a$", anchor="end")
    s.text(x1 - 12, r2 + 19, "$p$", anchor="end")
    s.text(x1 - 12, r3 + 22, "$s$", anchor="end")
    for y, h in ((r1, 32), (r2, 26), (r3, 32)):
        s.base(x1, y, 6 * cell, h)
    s.strip(x1, r1, [708, DOT, 129, DOT, DOT, 451], cell=cell, hues=SURVIVORS, quiet=(1, 3, 4))
    s.strip(x1, r2, [0, DOT, 1, DOT, DOT, 2], cell=cell, height=26, hues=SURVIVORS, quiet=(1, 3, 4))
    s.strip(x1, r3, [708, 129, 451, MINUS, MINUS, MINUS], cell=cell, hues=COMPACT)
    s.brace(x1 + 2, x1 + 3 * cell - 2, r3 + 38)
    s.brace(x1 + 3 * cell + 2, x1 + 6 * cell - 2, r3 + 38)
    s.text(x1 + 1.5 * cell, r3 + 64, "count $= 3$", size=NOTE, anchor="middle")
    s.text(x1 + 4.5 * cell, r3 + 64, "padding", size=NOTE, anchor="middle")
    s.save(
        "fig-localization.svg",
        panels=((10, 8, 324, 342), (336, 8, 480, 342), (818, 8, 294, 342)),
    )


def render_contract():
    """Figure 2: the worked row as a ruled table, with equations and both layouts."""
    s = Figure(WIDTH, 420, "Coordinate translation and the two output layouts", "")
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
    s.panel(14, 12, 704, 396, "Coordinate translation", "a", ground="b")
    s.panel(728, 12, 378, 396, "Equations and layouts", "b", ground="a")
    left, x0, width = 34, 302, 66
    centers = [x0 + (j + 0.5) * width for j in range(6)]
    top, pitch = 62, 40
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
        for cx, value in zip(centers, values, strict=True):
            s.text(cx, y, value, size=VALUE, anchor="middle",
                   cls="mute" if value in ("—", MINUS) else "")

    xe = 840
    s.equations(xe, 96, 34, EQUATIONS)
    s.rule(748, 1086, 254, weight=HAIR_RULE, cls="rule")
    sx, cell = 748, 49
    for y, label, values, hues in (
        (282, "front compaction", COMPACTED, COMPACT),
        (346, "column-preserving", PRESERVED, SURVIVORS),
    ):
        s.text(sx, y, label, size=NOTE)
        s.text(1086, y, "$n = 3$", size=NOTE, anchor="end")
        s.base(sx, y + 8, 6 * cell, 30)
        s.strip(sx, y + 8, values, cell=cell, hues=hues)
    s.save("fig-contract.svg")


def render_contract_narrow():
    """Phone layout of Figure 2: the table transposed so each input column is a row."""
    s = Figure(NARROW, 760, "Coordinate translation and output layouts, narrow layout", "")
    s.description = (
        "The same worked row as the wide figure, transposed: each input column j is a table row. "
        "Columns 0, 2 and 5 survive: t = 8, 130, 262 map to local positions 4, 65, 131, "
        "page-offset pairs (0, 4), (1, 1), (2, 3), physical pages 11, 2, 7 and physical slots "
        "708, 129, 451. The equations and both output layouts follow."
    )
    heads = ["$j$", "$t$", "$o$", "$v$", "$\\ell$", "$(b, \\delta)$", "$P$", "$a$"]
    xs = [36, 80, 122, 158, 198, 254, 314, 366]
    top, pitch = 58, 34
    columns = [[j] + [values[j] for _, values in TRANSLATION] for j in range(6)]
    bottom = top + 34 + pitch * 6 + 4
    s.panel(8, 12, 404, bottom + 18 - 12, "Coordinate translation", "a", ground="b")
    for j, hue in SURVIVORS.items():
        y = top + 34 + pitch * j + 2
        s.rect(18, y, NARROW - 36, pitch, f"{hue} area")
    s.rule(18, NARROW - 18, top, weight=1.4)
    s.rule(18, NARROW - 18, top + 34, weight=HAIR_RULE)
    s.rule(18, NARROW - 18, bottom, weight=1.4)
    for x, head in zip(xs, heads, strict=True):
        s.text(x, top + 23, head, size=VALUE, anchor="middle")
    for j, values in enumerate(columns):
        y = top + 34 + pitch * (j + 0.5) + 6.5
        for x, value in zip(xs, values, strict=True):
            s.text(x, y, value, size=VALUE, anchor="middle",
                   cls="mute" if value in ("—", MINUS) else "")
    y0 = bottom + 30
    s.panel(8, y0, 404, 396, "Equations and layouts", "b", ground="a")
    s.equations(150, y0 + 78, 34, EQUATIONS)
    y = y0 + 78 + 34 * 4 + 26
    s.rule(28, NARROW - 28, y - 2, weight=HAIR_RULE, cls="rule")
    for ddy, label, values, hues in (
        (24, "front compaction  $n = 3$", COMPACTED, COMPACT),
        (88, "column-preserving  $n = 3$", PRESERVED, SURVIVORS),
    ):
        s.text(30, y + ddy, label, size=NOTE)
        s.base(30, y + ddy + 8, 6 * 60, 30)
        s.strip(30, y + ddy + 8, values, cell=60, hues=hues)
    s.height = int(y0 + 396 + 10)
    s.save("fig-contract-narrow.svg")


def render_consumer():
    """Figure 5: layout masks, masked paged gathering, and shared normalization."""
    s = Figure(WIDTH, 470, "Layout-aware masking, paged gathering and a shared softmax", "")
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
    dy = 26
    s.panel(14, 12, 294, 446, "Layout and validity", "a", ground="a")
    s.panel(318, 12, 458, 446, "Masked gather", "b", ground="b")
    s.panel(786, 12, 320, 446, "Shared normalization", "c", ground="a")

    # (a) Layout and validity ------------------------------------------------
    x0, cell = 64, 56
    for y, title, slots, mask, hues, rule in (
        (
            70 + dy,
            "front compaction",
            [14, 12, MINUS, MINUS],
            [1, 1, 0, 0],
            pair,
            "$v_j = [\\, j < n \\,]$",
        ),
        (
            262 + dy,
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
        s.base(x0, y + 10, 4 * cell, 30)
        s.strip(
            x0,
            y + 10,
            slots,
            cell=cell,
            hues=hues,
            quiet=tuple(j for j in range(4) if j not in hues),
        )
        s.base(x0, y + 52, 4 * cell, 26)
        s.strip(x0, y + 52, mask, cell=cell, height=26, hues=hues)
        s.text(x0 + 2 * cell, y + 118, rule, anchor="middle", cls="soft")
    s.rule(30, x0 + 4 * cell + 14, 226 + dy, weight=HAIR_RULE, cls="rule")

    # (b) Masked gather --------------------------------------------------------
    bx, bc = 380, 50
    cols = [bx + (j + 0.5) * bc for j in range(4)]
    v_y, s_y, w1, s0, g, x_y, w2, out = (v + dy for v in (58, 96, 146, 188, 238, 280, 330, 372))
    for j, x in enumerate(cols):
        hue = spread.get(j)
        cls = f"{hue} trace" if hue else "wire"
        s.line([(x, s_y + 32), (x, s0)], cls=cls, collide=False)
        s.line([(x, s0 + 32), (x, x_y)], cls=cls, collide=False)
        s.line([(x, x_y + 32), (x, out - 1)], cls=cls, arrow=True, collide=False)
    # The mask reaches both selections through one dashed bus.
    bus = bx - 16
    s.line(
        [(bx, v_y + 13), (bus, v_y + 13), (bus, w2 + 13), (bx - 1, w2 + 13)],
        cls="mask",
        arrow=True,
        radius=4,
        collide=False,
    )
    s.line([(bus, w1 + 13), (bx - 1, w1 + 13)], cls="mask", arrow=True, collide=False)
    s.circle(bus, w1 + 13, 2.2, "dot")
    s.box(bx, w1 - 2, 4 * bc, 30, "$\\mathrm{where}(v, s, 0)$")
    s.box(bx, g - 2, 4 * bc, 30, "gather $K$, $V$")
    s.box(bx, w2 - 2, 4 * bc, 30, "$\\mathrm{where}(v, x, 0)$")
    s.text(bus - 10, v_y + 19, "$v$", anchor="end")
    s.text(bus - 10, s_y + 22, "$s$", anchor="end")
    for y, values, h, quiet in (
        (v_y, [1, 0, 1, 0], 26, ()),
        (s_y, [14, MINUS, 12, MINUS], 30, (1, 3)),
        (s0, [14, 0, 12, 0], 30, ()),
        (x_y, ["$x_{14}$", "$x_0$", "$x_{12}$", "$x_0$"], 30, ()),
        (out, ["$x_{14}$", 0, "$x_{12}$", 0], 30, ()),
    ):
        s.base(bx, y, 4 * bc, h)
        s.strip(bx, y, values, cell=bc, height=h, hues=spread, quiet=quiet)
    # Paged cache: indexed K and V rows; row 0 is the placeholder address.
    kx = 640
    s.memory(kx - 30, 166 + dy, 150, 144)
    s.text(kx + 45, 188 + dy, "paged KV cache", size=NOTE, anchor="middle")
    for ddx, name in ((0, "K"), (60, "V")):
        s.text(kx + ddx + 24, 212 + dy, f"${name}$", size=NOTE, anchor="middle")
        for row, hue in enumerate((None, "orange", "blue")):
            y = 222 + dy + row * 22 + (12 if row else 0)
            s.base(kx + ddx, y, 48, 18)
            s.strip(
                kx + ddx,
                y,
                [""] * 4,
                cell=12,
                height=18,
                hues={k: hue for k in range(4)} if hue else {},
            )
    for row, slot in enumerate((0, 12, 14)):
        s.text(
            kx - 8,
            237 + dy + row * 22 + (12 if row else 0),
            str(slot),
            size=NOTE,
            anchor="end",
            cls="mute",
        )
    for ddy in (-4.5, 0, 4.5):
        s.circle(kx + 24, 248 + dy + ddy, 1.3, "dot")
        s.circle(kx + 84, 248 + dy + ddy, 1.3, "dot")
    s.flow([(kx - 32, g + 13), (bx + 4 * bc + 2, g + 13)], collide=False)

    # (c) Shared normalization -----------------------------------------------
    cx, fork = 946, 80  # N and Z sit one fork apart on either side
    s.text(cx, 76 + dy, "$z_{rj} = q \\cdot K_{rj} / \\sqrt{d}$", anchor="middle")
    s.text(
        cx,
        104 + dy,
        "$z_{rj} \\gets -\\infty$ where $v_{rj} = 0$",
        size=NOTE,
        anchor="middle",
        cls="soft",
    )
    s.flow([(cx, 114 + dy), (cx, 140 + dy)])
    s.text(cx, 164 + dy, "$m = \\max_{r, j} z_{rj}$", anchor="middle")
    s.flow([(cx, 176 + dy), (cx, 202 + dy)])
    s.text(cx, 226 + dy, "$w_{rj} = v_{rj} \\exp(z_{rj} - m)$", anchor="middle")
    s.flow([(cx, 238 + dy), (cx, 252 + dy), (cx - fork, 252 + dy), (cx - fork, 272 + dy)],
           radius=6)
    s.flow([(cx, 252 + dy), (cx + fork, 252 + dy), (cx + fork, 272 + dy)], radius=6)
    s.junction(cx, 252 + dy)
    s.text(cx - fork, 296 + dy, "$N = \\sum_{r, j} w_{rj} V_{rj}$", anchor="middle")
    s.text(cx + fork, 296 + dy, "$Z = \\sum_{r, j} w_{rj}$", anchor="middle")
    s.flow([(cx - fork, 310 + dy), (cx - fork, 322 + dy), (cx + fork, 322 + dy), (cx + fork, 310 + dy)],
           arrow=False, radius=6)
    s.junction(cx, 322 + dy)
    s.flow([(cx, 322 + dy), (cx, 342 + dy)])
    s.text(cx - 6, 372 + dy, "$y =$", anchor="end", cls="accent tint")
    s.fraction(cx, 372 + dy, "$N$", "$Z$", cls="accent tint")
    s.text(cx, 412 + dy, "$y = 0$ if $Z = 0$", size=NOTE, anchor="middle", cls="soft")
    s.save(
        "fig-consumer.svg",
        panels=((10, 8, 302, 454), (314, 8, 466, 454), (782, 8, 328, 454)),
    )


SCAN_VALID = [1, 0, 1, 1, 0, 1, 0, 1]
SCAN_DEST = [0, DOT, 1, 2, DOT, 3, DOT, 4]


def _scan_schedules():
    """Each schedule's intermediates for SCAN_VALID, derived the way its kernel does."""
    valid = SCAN_VALID
    row = [sum(valid[:j]) for j in range(len(valid))]  # cumsum(v) - v
    tiles = [valid[k : k + 4] for k in (0, 4)]
    local = [sum(tile[:i]) for tile in tiles for i in range(4)]
    tile_counts = [sum(tile) for tile in tiles]
    tile_offsets = [0, tile_counts[0]]
    chunks = [valid[k : k + 2] for k in range(0, 8, 2)]  # 4 threads x 2 items
    chunk_counts = [sum(chunk) for chunk in chunks]
    groups = [chunk_counts[0:2], chunk_counts[2:4]]  # SIMD groups of 2 threads
    lane_prefix = [sum(group[:i]) for group in groups for i in range(2)]
    group_totals = [sum(group) for group in groups]
    preceding = [0, group_totals[0]]
    thread_offsets = [preceding[t // 2] + lane_prefix[t] for t in range(4)]
    dest_row = [row[j] if valid[j] else DOT for j in range(8)]
    dest_tile = [tile_offsets[j // 4] + local[j] if valid[j] else DOT for j in range(8)]
    dest_threads = []
    for t, chunk in enumerate(chunks):
        position = thread_offsets[t]
        for keep in chunk:
            dest_threads.append(position if keep else DOT)
            position += keep
    assert dest_row == dest_tile == dest_threads == SCAN_DEST
    return {
        "local": [local[j] if valid[j] else DOT for j in range(8)],
        "tile_counts": tile_counts,
        "tile_offsets": tile_offsets,
        "chunk_counts": chunk_counts,
        "thread_offsets": thread_offsets,
        "group_totals": group_totals,
    }


def render_scan():
    """Docs figure: three scan schedules that assign the same stable destinations."""
    k = _scan_schedules()
    s = Figure(WIDTH, 424, "Three scan schedules, one set of destinations", "")
    s.description = (
        "One row of eight columns with validity [1, 0, 1, 1, 0, 1, 0, 1] and three ways to "
        "number its survivors. Panel a, the rowwise kernel and the MLX composition: "
        "cumsum(v) - v over the whole row gives destinations 0 to 4. Panel b, the hierarchical "
        "kernel with tiles of four columns: local positions within each tile, tile counts "
        f"{tuple(k['tile_counts'])}, exclusive tile offsets {tuple(k['tile_offsets'])}, then "
        "offset plus local position. Panel c, the Metal kernel with four threads of two columns "
        "in two SIMD groups: chunk counts "
        f"{k['chunk_counts']}, thread offsets {k['thread_offsets']} from a SIMD prefix and "
        "group totals, then each thread writes its chunk in order. All three give "
        "destinations [0, -, 1, 2, -, 3, -, 4]. Sizes are schematic."
    )
    cell, v_y, d_y = 34, 64, 344
    hues = {j: "blue" for j, keep in enumerate(SCAN_VALID) if keep}
    quiet = tuple(j for j, keep in enumerate(SCAN_VALID) if not keep)
    xs = (44, 424, 824)
    w = 8 * cell

    def col(x, j):
        return x + (j + 0.5) * cell

    def traces(x, y0, y1, columns=hues):
        for j in columns:
            s.line([(col(x, j), y0), (col(x, j), y1 - 1)], cls="blue trace", arrow=True,
                   collide=False)

    # Connectors first: every stage drawn afterwards is a paper band they pass beneath.
    traces(xs[0], v_y + 31, d_y)
    traces(xs[1], v_y + 31, 168)
    traces(xs[1], 168 + 31, d_y)
    traces(xs[2], v_y + 31, 150)
    chunk = 2 * cell
    for t in range(4):
        cx = xs[2] + (t + 0.5) * chunk
        s.line([(cx, 150 + 31), (cx, 238 - 1)], cls="wire", arrow=True, collide=False)
    traces(xs[2], 238 + 31, d_y)

    for p, x in enumerate(xs):
        s.text(x - 10, v_y + 21, "$v$", anchor="end")
        s.strip(x, v_y, SCAN_VALID, cell=cell, hues=hues, quiet=quiet, name=f"v{p}")
        s.text(x - 10, d_y + 21, "$d$", anchor="end")
        s.strip(x, d_y, SCAN_DEST, cell=cell, hues=hues, quiet=quiet, bold=True, name=f"d{p}")
    # (a) One program numbers the whole row.
    s.band(xs[0], 196, w, "$\\mathrm{cumsum}(v) - v$")
    # (b) Number within tiles, then add each tile's offset.
    x = xs[1]
    s.band(x, 108, w, "scan within each tile")
    s.text(x - 10, 168 + 21, "$\\ell$", anchor="end")
    s.strip(x, 168, k["local"], cell=cell, hues=hues, quiet=quiet, group=4, name="local")
    counts = ", ".join(map(str, k["tile_counts"]))
    s.text(x + w + 10, 168 + 21, f"$c = ({counts})$", size=NOTE, cls="soft")
    offsets = ", ".join(map(str, k["tile_offsets"]))
    s.band(x, 254, w, f"add tile offsets $({offsets})$")
    # (c) Count each thread's chunk, scan the counts, write each chunk in order.
    x = xs[2]
    s.band(x, 104, w, "count each chunk")
    s.text(x - 10, 150 + 21, "$n$", anchor="end")
    s.strip(x, 150, k["chunk_counts"], cell=chunk, group=2, name="chunk counts")
    s.band(x, 196, w, "SIMD prefix + group totals")
    s.text(x - 10, 238 + 21, "$o$", anchor="end")
    s.strip(x, 238, k["thread_offsets"], cell=chunk, group=2, name="thread offsets")
    s.band(x, 290, w, "write chunks in order")
    for x, letter, title in zip(xs, "abc", ("Rowwise and MLX", "Hierarchical", "Metal"),
                                strict=True):
        s.subcaption(x + w / 2, 414, letter, title)
    s.save(
        "fig-scan.svg",
        panels=((14, 48, 314, 376), (394, 48, 396, 376), (794, 48, 314, 376)),
    )


def _tile_row(s, x, y, order, cell):
    values = [v for t in order for v in TILES[t]]
    hues = {2 * k + i: TILE_HUES[t] for k, t in enumerate(order) for i in (0, 1)}
    s.strip(x, y, values, cell=cell, height=30, hues=hues, group=2)


def render_order():
    """Figure 6: atomic reservation against stable prefix destinations, over replays."""
    s = Figure(WIDTH, 362, "Completion-order reservation versus deterministic tile prefixes", "")
    s.description = (
        "Tile groups T0 = [704, 705], T1 = [132, 133], T2 = [448, 896] and T3 = [897, 260] "
        "are tracked by four colors. Panel a: atomic reservation claims output offsets in "
        "completion order, so three replays of the same input emit T1, T0, T3, T2; then T0, T2, "
        "T1, T3; then T2, T3, T0, T1. Panel b: offsets from an exclusive scan of tile counts, "
        "0, 2, 4 and 6, emit T0, T1, T2, T3 on every replay. Both paths write the same eight "
        "values and count; curves trace the first replay. These are schematic arrays, not "
        "measured GPU results."
    )
    cell, dy = 46, 34
    for panel, (left, x, orders, title, note, ground) in enumerate(
        (
            (40, 140, ATOMIC, "Atomic reservation", "offsets claimed in completion order", "a"),
            (
                600,
                700,
                [[0, 1, 2, 3]] * 3,
                "Stable prefix destinations",
                "offsets from an exclusive scan: 0, 2, 4, 6",
                "b",
            ),
        )
    ):
        s.panel(left, 12, 480, 336, title, "ab"[panel], ground=ground)
        centers = [x + (2 * k + 1) * cell for k in range(4)]
        s.text(x + 4 * cell, 50 + dy, note, size=NOTE, anchor="middle", cls="soft")
        for k, hue in enumerate(TILE_HUES):
            s.text(centers[k], 80 + dy, f"$T_{k}$", size=NOTE, anchor="middle", cls=f"{hue} tint")
        s.text(x - 14, 110 + dy, "input", anchor="end")
        s.base(x, 88 + dy, 8 * cell, 30)
        _tile_row(s, x, 88 + dy, [0, 1, 2, 3], cell)
        for r, order in enumerate(orders):
            y = 174 + dy + 46 * r
            s.text(x - 14, y + 22, f"replay {r + 1}", anchor="end")
            s.base(x, y, 8 * cell, 30)
            _tile_row(s, x, y, order, cell)
        for t in range(4):
            dest = orders[0].index(t)
            s.curve(
                (centers[t], 118 + dy),
                (centers[dest], 173 + dy),
                cls=f"{TILE_HUES[t]} trace",
                collide=False,
            )
    s.save("fig-problem.svg", panels=((36, 8, 488, 344), (596, 8, 488, 344)))


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
    # The system figure: each rank's slots, their hues, and the cache cells they fill.
    assert set(SELECTION) == {j for j, t in enumerate(ROW) if t >= 0}
    for rank, (values, hues, count) in enumerate(RANK_OUTPUTS):
        slots = [-1 if v == MINUS else v for v in values]
        assert localize_reference(
            [0], [TABLE], [ROW], block_size=64, dcp_size=2, dcp_rank=rank
        ) == ([slots], [count])
        owned = [j for j, owner in enumerate(worked["o"]) if owner == rank]
        assert all(SELECTION[j] == (SURVIVORS[j] if rank == 0 else "gray") for j in owned)
        assert hues == {k: SELECTION[j] for k, j in enumerate(owned)}
        assert KV_SLOTS[rank] == {slots[k]: hues[k] for k in range(count)}
        assert all(slot // 64 in KV_BLOCKS and slot % 64 < 5 for slot in KV_SLOTS[rank])
    render_system()
    render_system_narrow()
    render_overview()
    render_contract()
    render_contract_narrow()
    render_consumer()
    render_order()
    render_scan()


if __name__ == "__main__":
    main()
