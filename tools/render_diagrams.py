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


def _selector(s, cx, top, width=260, inset=28, height=30):
    """The external selector as a funnel: many candidate positions in, a few out."""
    left, right = cx - width / 2, cx + width / 2
    s.path(
        f"M{left} {top}H{right}L{right - inset} {top + height}H{left + inset}Z", "selector"
    )
    s.text(cx, top + height / 2 + 7, "sparse selector", anchor="middle")


def _rank_lane(s, a, top, rank, *, frame, cell=38, kv_dx=135, p_input=True, bw=230, kv_cell=13):
    """One rank of the system figure: SILKern, its slots, and the attention consumer.

    ``a`` is the lane's flow axis, ``top`` the top edge of the rank's frame, and
    ``frame`` its (left, right) edges. Returns the baseline of the partial result.
    """
    values, hues, count = RANK_OUTPUTS[rank]
    left, right = frame
    s.rect(left, top, right - left, 222, "rule-frame")
    s.text(left + 14, top + 26, f"rank {rank}", size=NOTE)
    y = top + 40
    s.band(a - bw / 2, y, bw, f"**SILKern**,  $r = {rank}$", accent=True)
    if p_input:
        s.text(a - bw / 2 - 45, y + 20, "$P$", anchor="end")
        s.line([(a - bw / 2 - 39, y + 13), (a - bw / 2 - 4, y + 13)], cls="flow", arrow=True)
    s.line([(a, y + 30), (a, y + 48)], cls="flow", arrow=True)
    y += 50
    s.text(a - 3 * cell - 12, y + 22, f"$s_{rank}$", anchor="end")
    s.strip(a - 3 * cell, y, values, cell=cell, hues=hues, bold=rank == 0,
            quiet=tuple(j for j in range(6) if j not in hues), name=f"s{rank}")
    s.text(a + 3 * cell + 10, y + 21, f"$n_{rank} = {count}$", size=NOTE)
    s.line([(a, y + 30), (a, y + 52)], cls="flow", arrow=True)
    y += 54
    s.band(a - bw / 2, y, bw, "sparse attention")
    # The rank's own paged KV cache: rows are physical blocks, cells their slots.
    zx, zy = a + kv_dx, y - 26
    zw = 47 + 5 * kv_cell
    s.rect(zx, zy, zw, 100, "zone")
    s.text(zx + zw / 2, zy + 20, "KV cache", size=NOTE, anchor="middle")
    for k, block in enumerate(KV_BLOCKS):
        cells = {slot % 64: hue for slot, hue in KV_SLOTS[rank].items() if slot // 64 == block}
        row = zy + 30 + 21 * k
        s.text(zx + 30, row + 13, str(block), size=NOTE - 2, anchor="end", cls="mute")
        s.strip(zx + 36, row, [""] * 5, cell=kv_cell, height=14, hues=cells,
                name=f"kv{rank}{k}")
    s.line([(zx - 2, y + 13), (a + bw / 2 + 2, y + 13)], cls="flow", arrow=True)
    s.line([(a, y + 27), (a, y + 46)], cls="flow", arrow=True)
    s.text(a, y + 67, f"$(m_{rank}, N_{rank}, Z_{rank})$", anchor="middle")
    return y + 67


def render_system():
    """Hero: one selection row, localized independently on each rank, then consumed."""
    s = Figure(WIDTH, 438, "Where SILKern sits in a context-parallel decode step", "")
    s.description = (
        "A sparse selector picks global positions for one query, t = [8, 5, 130, 7, -1, 262], "
        "and every rank of a two-rank decode receives the same row. Each rank runs SILKern with "
        "its own rank number and the page table P: rank 0 keeps tokens 8, 130 and 262 and "
        "returns physical slots [708, 129, 451, -1, -1, -1] with count 3; rank 1 keeps tokens 5 "
        "and 7 and returns [706, 707, -1, -1, -1, -1] with count 2. Each rank's sparse attention "
        "gathers those slots from its own paged KV cache, drawn as physical blocks 2, 7 and 11, "
        "and produces partial softmax terms (m, N, Z); one step across both ranks combines them "
        "into y = N / Z. SILKern's stages are highlighted."
    )
    cx = WIDTH / 2
    s.text(cx, 18, "query $q$", anchor="middle")
    s.line([(cx, 27), (cx, 38)], cls="flow", arrow=True)
    _selector(s, cx, 40)
    s.line([(cx, 71), (cx, 88)], cls="flow", arrow=True)
    cell, ty = 46, 90
    s.text(cx - 3 * cell - 12, ty + 22, "$t$", anchor="end")
    s.strip(cx - 3 * cell, ty, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SELECTION,
            quiet=(4,), name="t")
    s.text(cx + 3 * cell + 12, ty + 21, "the same row on every rank", size=NOTE, cls="soft")
    lanes = ((260, (70, 530)), (780, (590, 1050)))
    fork = ty + 42
    s.line([(cx, ty + 30), (cx, fork)], cls="flow", collide=False)
    s.circle(cx, fork, 2.2, "flow-head")
    top = fork + 10
    for a, _ in lanes:
        s.line([(cx, fork), (a, fork), (a, top + 38)], cls="flow", arrow=True, radius=6,
               collide=False)
    bottoms = [_rank_lane(s, a, top, rank, frame=frame) for rank, (a, frame) in enumerate(lanes)]
    band = top + 240
    for (a, _), y in zip(lanes, bottoms, strict=True):
        s.line([(a, y + 12), (a, band - 1)], cls="flow", arrow=True)
    s.band(70, band, 980, "combine across ranks:  $m = \\max_r m_r$,  "
           "$N = \\sum_r \\, \\exp(m_r - m)\\, N_r$,  $Z = \\sum_r \\, \\exp(m_r - m)\\, Z_r$,  "
           "$y = N / Z$")
    s.height = band + 40
    s.save("fig-system.svg")


def render_system_narrow():
    """Phone layout of the system figure: ranks stacked, with buses at both margins."""
    s = Figure(NARROW, 720, "Where SILKern sits in a context-parallel decode step, narrow", "")
    s.description = (
        "The same decode step as the wide figure, stacked for narrow pages. The selector's row "
        "t = [8, 5, 130, 7, -1, 262] reaches rank 0 directly and rank 1 along the right margin. "
        "Rank 0's SILKern returns [708, 129, 451, -1, -1, -1] with count 3 and rank 1's returns "
        "[706, 707, -1, -1, -1, -1] with count 2; each rank's attention gathers from its own KV "
        "cache, and the partial terms meet along the left margin to give y = N / Z."
    )
    cx = NARROW / 2
    s.text(cx - 118, 35, "$q$", anchor="end")
    s.line([(cx - 112, 29), (cx - 96, 29)], cls="flow", arrow=True)
    _selector(s, cx, 14, width=200, inset=22)
    s.line([(cx, 45), (cx, 62)], cls="flow", arrow=True)
    cell = 50
    s.text(cx - 3 * cell - 10, 86, "$t$", anchor="end")
    s.strip(cx - 3 * cell, 64, ROW[:4] + [MINUS, ROW[5]], cell=cell, hues=SELECTION, quiet=(4,),
            name="t")
    s.text(cx + 10, 122, "same row on every rank", size=NOTE, cls="soft")
    a, frame, bus_r, bus_l = 166, (30, 390), 405, 15
    tops = (152, 408)
    band_y = [top + 53 for top in tops]  # centre line of each SILKern stage
    s.line([(cx, 94), (cx, 132)], cls="flow")
    s.circle(cx, 132, 2.2, "flow-head")
    s.line([(cx, 132), (a, 132), (a, tops[0] + 38)], cls="flow", arrow=True, radius=6,
           collide=False)
    s.line([(cx, 132), (bus_r, 132), (bus_r, band_y[1]), (a + 100, band_y[1])], cls="flow",
           arrow=True, radius=6, collide=False)
    bottoms = [
        _rank_lane(s, a, top, rank, frame=frame, cell=36, kv_dx=126, p_input=False, bw=196,
                   kv_cell=11)
        for rank, top in enumerate(tops)
    ]
    join = 676
    s.line([(a, bottoms[0] + 8), (a, tops[1] - 14), (bus_l, tops[1] - 14), (bus_l, join + 13),
            (cx - 112, join + 13)], cls="flow", arrow=True, radius=6, collide=False)
    s.line([(a, bottoms[1] + 8), (a, join - 1)], cls="flow", arrow=True, collide=False)
    s.band(cx - 110, join, 200, "combine partials")
    s.line([(cx + 92, join + 13), (cx + 116, join + 13)], cls="flow", arrow=True)
    s.text(cx + 122, join + 20, "$y = N / Z$")
    s.height = join + 40
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
        "fig-localization.svg",
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
