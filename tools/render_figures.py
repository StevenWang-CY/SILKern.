"""Regenerate the evidence-backed SVG charts from checked-in measurement records.

    python -m pip install -e ".[docs]"
    python tools/render_figures.py

Every plotted value and every number in the accessible descriptions is read from
``evidence/``; nothing here runs a benchmark. Charts share the typography of the
explanatory diagrams and the layout of a model-release chart: a title, a gray
subtitle that says which way is better, flat bars with a hairline outline, the
measured SILKern arm in a saturated blue against grays, and values set just
above their bars. Every latency scale starts at zero, and bar labels repeat the
recorded medians to one decimal place.
"""

from __future__ import annotations

import json
from pathlib import Path

from figure_style import LABEL, NARROW, NOTE, ROOT, VALUE, WIDTH, Figure

APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
CUDA_RECORD = Path("evidence/05-full-decode-canary")
MLX, METAL = "mlx_compiled", "metal_compiled"
LATENCY_TOP = 400  # one zero-based latency scale for every Apple row, µs
LATENCY_TICKS = (0, 100, 200, 300, 400)
ARMS = (
    ("row_stable", "rowwise"),
    ("pinned_atomic", "atomic"),
    ("hierarchical_stable", "hierarchical"),
)
CONTRASTS = (
    ("row_stable", 32768, "32K", "rowwise"),
    ("hierarchical_stable", 32768, "32K", "hierarchical"),
    ("row_stable", 65536, "64K", "rowwise"),
    ("hierarchical_stable", 65536, "64K", "hierarchical"),
)
RATIO_AXIS = (0.995, 1.020)
RATIO_TICKS = (1.0, 1.01, 1.02)  # labeled; minor gridlines fall halfway between
RATIO_MINOR = (0.995, 1.005, 1.015)


def _converter_bars(s: Figure, segments, x0, base, scale, *, width, pitch) -> None:
    """The 32K converter segment: rowwise highlighted, the hierarchical kernel in pale blue
    as SILKern's other arm, atomic reservation in gray; each value set above its bar and
    each bar named twice, by kernel and by whose kernel it is."""
    owners = {"row_stable": "SILKern", "pinned_atomic": "baseline",
              "hierarchical_stable": "SILKern"}
    fills = {"row_stable": "g-hi", "pinned_atomic": "g-mid", "hierarchical_stable": "g-soft"}
    for k, (arm, label) in enumerate(ARMS):
        value = segments["contexts"]["32768"][f"converter.{arm}"]["pooled_median_us"]
        x = x0 + pitch * k
        _bar(s, x, base, width, value * scale, fills[arm])
        s.text(x + width / 2, base - value * scale - 9, f"{value:.1f}", size=NOTE,
               anchor="middle")
        s.text(x + width / 2, base + 26, label, size=VALUE, anchor="middle")
        s.text(x + width / 2, base + 50, owners[arm], size=NOTE, anchor="middle", cls="mute")


def _estimate(contrast) -> str:
    return f"{contrast['point']:.4f}  [{contrast['lo']:.4f}, {contrast['hi']:.4f}]"


def _contrast_style(arm: str, contrast, margin: float) -> tuple[str, bool]:
    """Orange marks an interval beyond the margin; blue the rowwise arm; pale blue the
    hierarchical arm."""
    adverse = contrast["hi"] > margin
    return ("orange" if adverse else "accent" if arm == "row_stable" else "soft"), adverse


def _interval(s: Figure, xr, contrast, y: float, hue: str) -> None:
    """A confidence interval with end caps and an outlined estimate."""
    s.readme = True
    stroke = "wire" if hue == "soft" else f"{hue} trace"
    s.line([(xr(contrast["lo"]), y), (xr(contrast["hi"]), y)], cls=stroke, collide=False)
    for end in ("lo", "hi"):
        s.line([(xr(contrast[end]), y - 6), (xr(contrast[end]), y + 6)], cls=stroke,
               collide=False)
    fill = {"accent": "g-hi", "soft": "g-soft", "orange": "orange tint"}[hue]
    s.circle(xr(contrast["point"]), y, 6, f"g-bar {fill}")


def _margin_rule(s: Figure, x: float, y0: float, y1: float) -> None:
    s.raw(f'<path d="M{x:.2f} {y0}V{y1}" class="axis" style="stroke-dasharray:4 3"/>')


def _apple_rows(cells) -> list[tuple[int, int, float, float, float]]:
    """(width, batch, compiled MLX, compiled Metal, speedup) rows, ordered by width then batch."""
    rows = []
    for cell in sorted(cells, key=lambda c: (c["geometry"]["width"], c["geometry"]["batch"])):
        mlx, metal = cell["median_us"][MLX], cell["median_us"][METAL]
        # The recorded ratio must be the ratio of the plotted medians.
        assert abs(cell["compiled_ratio"] - mlx / metal) < 1e-9, cell["geometry"]
        rows.append((cell["geometry"]["width"], cell["geometry"]["batch"], mlx, metal,
                     cell["compiled_ratio"]))
    return rows


TITLE = 23  # a chart's title; every other size comes from the shared type scale


def _titles(s: Figure, cx: float, y: float, title: str, subtitle: str, *, size=TITLE) -> None:
    """A centred title over a gray subtitle that names the measure and its direction."""
    s.readme = True
    s.text(cx, y, title, size=size, anchor="middle")
    s.text(cx, y + 28, subtitle, size=NOTE, anchor="middle", cls="mute")


def _bar(s: Figure, x: float, base: float, w: float, h: float, fill: str, r: float = 4) -> None:
    """A flat bar with a hairline outline and slightly rounded top corners."""
    s.readme = True
    r = min(r, w / 2, h)
    s.path(
        f"M{x:.2f} {base:.2f}V{base - h + r:.2f}Q{x:.2f} {base - h:.2f} {x + r:.2f} {base - h:.2f}"
        f"H{x + w - r:.2f}Q{x + w:.2f} {base - h:.2f} {x + w:.2f} {base - h + r:.2f}"
        f"V{base:.2f}Z",
        f"g-bar {fill}",
    )


def _hbar(s: Figure, x: float, y: float, length: float, h: float, fill: str, r: float = 3):
    """A horizontal flat bar from ``x``, its far corners slightly rounded."""
    s.readme = True
    r = min(r, h / 2, length)
    s.path(
        f"M{x:.2f} {y:.2f}H{x + length - r:.2f}Q{x + length:.2f} {y:.2f} {x + length:.2f} "
        f"{y + r:.2f}V{y + h - r:.2f}Q{x + length:.2f} {y + h:.2f} {x + length - r:.2f} "
        f"{y + h:.2f}H{x:.2f}Z",
        f"g-bar {fill}",
    )


def _legend(s: Figure, cx: float, y: float, items, *, gap: float = 30) -> None:
    """A centred row of swatches and names: (fill class, label) pairs."""
    widths = [16 + 8 + s_measure(label) for _, label in items]
    x = cx - (sum(widths) + gap * (len(items) - 1)) / 2
    for (fill, label), w in zip(items, widths, strict=True):
        s.rect(x, y - 13, 16, 16, f"g-bar {fill}", rx=3)
        s.text(x + 24, y, label, size=NOTE, cls="soft")
        x += w + gap


def s_measure(label: str) -> float:
    from figure_style import measure

    return measure(label, NOTE)


def _behind(s: Figure, mark: int, draw) -> None:
    """Run ``draw`` and move what it drew to position ``mark``, beneath later marks."""
    start = len(s.items)
    draw()
    drawn = s.items[start:]
    del s.items[start:]
    s.items[mark:mark] = drawn


def _yaxis(s: Figure, x0: float, x1: float, base: float, scale: float, ticks, *, spans=None):
    """Light horizontal gridlines at each tick, a dark baseline, and gray tick values at the
    left. ``spans`` limits the gridlines to (left, right) pieces, so they break between
    groups."""
    for tick in ticks:
        y = base - tick * scale
        for left, right in spans or ((x0, x1),):
            s.path(f"M{left:.2f} {y:.2f}H{right:.2f}", "grid" if tick else "axis")
        s.text(x0 - 10, y + 6, f"{tick:g}", size=NOTE, anchor="end", cls="mute")


def _session_ratios(record: Path, names, key) -> list[float]:
    """Compiled MLX over compiled Metal, for every session and geometry."""
    ratios = []
    for name in names:
        for cell in json.loads((record / name).read_text())["cells"]:
            results = cell["results"]
            ratios.append(results[MLX]["median_us"] / results[METAL]["median_us"])
    return ratios


def _session_note(record: Path, data, consumer) -> str:
    localization = _session_ratios(record, data["sessions"], MLX)
    attention = _session_ratios(record, consumer["sessions"], MLX)
    every = localization + attention
    assert min(every) > 1, "a session where compiled Metal was not faster needs saying"
    return (
        f"Compiled Metal was faster in all {len(every)} session pairs; per-session ratios "
        f"range from {min(localization):.2f} to {max(localization):.2f} for localization "
        "because one of three sessions ran slower for both arms."
    )


def _apple_bars(s: Figure, groups, *, left, base, scale, bar, pitch, cluster, between,
                labels, solo) -> float:
    """Grouped bars on one zero-based latency scale: compiled Metal (blue) beside compiled
    MLX (gray) for every geometry, each pair's speedup set above it. ``groups`` holds
    (heading, rows, by_width) triples; ``by_width`` clusters rows that share a selection
    width under one width label, and otherwise pairs sit ``solo`` apart, each under its
    own. Returns the right edge of the last group."""
    x = left
    for k, (heading, rows, by_width) in enumerate(groups):
        if k:
            x += between
        start, previous, centres = x, None, []
        for width, batch, mlx, metal, ratio in rows:
            if previous is not None:
                if not by_width:
                    x += solo
                else:
                    x += cluster if width != previous else pitch - 2 * bar - 3
            _bar(s, x, base, bar, metal * scale, "g-hi")
            _bar(s, x + bar + 3, base, bar, mlx * scale, "g-lo")
            centre = x + bar + 1.5
            top = base - max(mlx, metal) * scale
            s.text(centre, top - 9, f"{ratio:.2f}×", size=NOTE, anchor="middle")
            s.text(centre, base + 26, str(batch), size=VALUE, anchor="middle")
            centres.append((width, centre))
            previous = width
            x += 2 * bar + 3
        end = x
        s.text((start + end) / 2, base - 400 * scale - 26, heading, size=LABEL, anchor="middle")
        if by_width:
            for width in sorted({w for w, _ in centres}):
                xs = [c for w, c in centres if w == width]
                s.text((xs[0] + xs[-1]) / 2, base + 54, f"width {width:,}", size=NOTE,
                       anchor="middle", cls="soft")
        else:
            for width, centre in centres:
                s.text(centre, base + 54, f"width {width:,}", size=NOTE, anchor="middle",
                       cls="soft")
        labels.append((start, end))
    return x


def render_apple() -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    consumer = json.loads((record / "consumer-summary.json").read_text())
    session = json.loads((record / data["sessions"][0]).read_text())
    device = session["metadata"]["device"]["device_name"]
    mlx_version = session["metadata"]["versions"]["mlx"]
    localization, attention = _apple_rows(data["cells"]), _apple_rows(consumer["cells"])
    ratios = [row[4] for row in localization]
    groups = (("Localization", localization, True), ("Selected attention", attention, False))
    title = f"{device} compiled localization and selected-attention latency"
    description = (
        f"Grouped bar chart of median latency per synchronized call on {device} with MLX "
        f"{mlx_version}, on one zero-based 0 to {LATENCY_TOP} microsecond scale. For each "
        "geometry a blue bar gives the compiled custom Metal kernel and a gray bar compiled "
        "compositional MLX, and the ratio of the two is set above the pair. The left group is "
        f"localization at selection widths "
        f"{', '.join(f'{w:,}' for w in sorted({r[0] for r in localization}))} and batch sizes "
        f"{', '.join(map(str, sorted({r[1] for r in localization})))}; Metal is "
        f"{min(ratios):.2f} to {max(ratios):.2f} times faster. The right group is complete "
        "selected attention with two logical shards on one device, fixed caches and "
        "64-dimensional keys and values: "
        + "; ".join(f"batch {b} with width {w:,}, {r:.2f} times" for w, b, _, _, r in attention)
        + ". Each ratio is compiled MLX latency divided by compiled Metal latency. Values are "
        "medians of three process-session medians and include dispatch, allocation, execution "
        "and synchronization; warmup and first-use compilation are excluded. "
        f"{_session_note(record, data, consumer)} Source: "
        f"{APPLE_RECORD.as_posix()}/summary.json, consumer-summary.json and the session records."
    )

    s = Figure(WIDTH, 520, title, description)
    _titles(s, WIDTH / 2, 34, f"Median latency per call on {device}",
            f"Compiled Metal kernel against compiled MLX {mlx_version}, µs  •  Lower is better")
    _legend(s, WIDTH / 2, 104, (("g-hi", "compiled Metal kernel"), ("g-lo", "compiled MLX")))
    base, scale = 420, 0.6
    spans: list[tuple[float, float]] = []
    mark = len(s.items)
    right = _apple_bars(s, groups, left=118, base=base, scale=scale, bar=27, pitch=80,
                        cluster=40, between=72, labels=spans, solo=54)
    _behind(s, mark, lambda: _yaxis(
        s, 96, right, base, scale, LATENCY_TICKS,
        spans=[(spans[0][0] - 14, spans[0][1] + 14), (spans[1][0] - 14, spans[1][1] + 14)]))
    s.text(84, base + 26, "batch", size=NOTE, anchor="end", cls="mute")
    s.height = base + 72
    s.save("fig-apple-performance.svg")

    # Phone layout: one row per geometry, the two bars stacked and the speedup at the
    # right, so nothing crowds at 340 pixels.
    n = Figure(NARROW, 900, title, description)
    _titles(n, NARROW / 2, 30, "Median latency per call", f"{device}, µs  •  Lower is better",
            size=LABEL)
    _legend(n, NARROW / 2, 92, (("g-hi", "Metal kernel"), ("g-lo", "compiled MLX")), gap=20)
    x0, nscale, bar_h, pitch = 118, 0.6, 9, 31
    y = 138
    n.text(52, y, "width", size=NOTE, anchor="end", cls="mute")
    n.text(106, y, "batch", size=NOTE, anchor="end", cls="mute")
    n.text(408, y, "MLX / Metal", size=NOTE, anchor="end", cls="mute")
    for heading, rows, _ in groups:
        y += 40
        n.text(14, y, heading, size=NOTE, cls="soft")
        top, previous = y + 10, None
        rows_mark = len(n.items)
        for width, batch, mlx, metal, ratio in rows:
            y += pitch + (10 if previous is not None and width != previous else 0)
            if width != previous:
                n.text(52, y, f"{width:,}", size=NOTE, anchor="end")
            n.text(106, y, str(batch), size=NOTE, anchor="end")
            n.text(408, y, f"{ratio:.2f}×", size=NOTE, anchor="end")
            _hbar(n, x0, y - 19, metal * nscale, bar_h, "g-hi")
            _hbar(n, x0, y - 19 + bar_h + 2, mlx * nscale, bar_h, "g-lo")
            previous = width
        bottom = y + 6
        _behind(n, rows_mark, lambda top=top, bottom=bottom: [
            n.path(f"M{x0 + tick * nscale:.2f} {top:.2f}V{bottom:.2f}", "grid")
            for tick in (100, 200, 300, 400)
        ])
        n.path(f"M{x0} {top:.2f}V{y + 6:.2f}", "axis")
    axis_y = y + 14
    for tick in (0, 200, 400):
        n.text(x0 + tick * nscale, axis_y + 18, f"{tick:g}", size=NOTE, anchor="middle",
               cls="mute")
    n.height = int(axis_y + 34)
    n.save("fig-apple-performance-narrow.svg")


def render_cuda() -> None:
    segments = json.loads((ROOT / CUDA_RECORD / "segments.json").read_text())
    analysis = json.loads((ROOT / CUDA_RECORD / "analysis.json").read_text())
    margin = analysis["tier1_margin"]
    medians = {
        arm: segments["contexts"]["32768"][f"converter.{arm}"]["pooled_median_us"]
        for arm, _ in ARMS
    }
    contrasts = {
        (arm, context): analysis["contrasts"][f"{arm}_over_atomic.c{context}"]
        for arm, context, _, _ in CONTRASTS
    }
    first = next(iter(contrasts.values()))
    confidence = f"{first['confidence']:.2%}"
    adverse = [
        f"{arm_label.capitalize()} at {ctx_label} is {contrasts[arm, context]['point']:.6f} "
        "times atomic"
        for arm, context, ctx_label, arm_label in CONTRASTS
        if contrasts[arm, context]["hi"] > margin
    ]
    description = (
        "Left: at 32K context the 48-layer converter segment takes "
        f"{medians['row_stable']:.3f} microseconds rowwise, {medians['pinned_atomic']:.3f} atomic "
        f"and {medians['hierarchical_stable']:.3f} hierarchical; rowwise is "
        f"{1 - medians['row_stable'] / medians['pinned_atomic']:.1%} lower than atomic. Bars start "
        "at zero and labels round to one decimal place. Right: complete-step latency ratios "
        f"against atomic with {confidence} intervals over {first['sessions']} sessions, on a "
        f"{RATIO_AXIS[0]:.3f} to {RATIO_AXIS[1]:.3f} axis. The solid line marks equal latency and "
        f"the dashed line the prespecified {margin:.2f} margin, with the region beyond it shaded; "
        "the numeric column repeats each estimate and interval. "
        + "; ".join(adverse)
        + (", and its interval lies" if len(adverse) == 1 else ", and their intervals lie")
        + " beyond the margin, a measured regression; the other contrasts meet the margin. "
        "These are historical two-H100 reference-executor results, "
        "not serving-runtime performance; no NVIDIA experiments were run for this update. "
        f"Source: {CUDA_RECORD.as_posix()}/segments.json and analysis.json."
    )
    s = Figure(WIDTH, 380, "Archived CUDA converter latency and complete decode cost", description)

    # Converter segment at 32K --------------------------------------------------
    _titles(s, 222, 32, "Converter segment, 32K context", "Pooled median, µs  •  Lower is better")
    base, scale = 304, 0.84
    mark = len(s.items)
    _converter_bars(s, segments, 112, base, scale, width=58, pitch=96)
    _behind(s, mark, lambda: _yaxis(s, 90, 362, base, scale, (0, 100, 200)))

    # Complete-step ratios ----------------------------------------------------------
    px, plength = 626, 272

    def xr(value: float) -> float:
        return px + plength * (value - RATIO_AXIS[0]) / (RATIO_AXIS[1] - RATIO_AXIS[0])

    _titles(s, 776, 32, "Complete decode step relative to atomic",
            f"Latency ratio, {confidence} interval over {first['sessions']} sessions  •  "
            "Lower is better")
    yt, yb = 104, 286
    s.rect(xr(margin), yt, xr(RATIO_AXIS[1]) - xr(margin), yb - yt, "g-margin")
    for tick in RATIO_MINOR + RATIO_TICKS:
        s.path(f"M{xr(tick):.2f} {yt}V{yb}", "grid")
    for tick in RATIO_TICKS:
        s.text(xr(tick), yb + 24, f"{tick:.2f}", size=NOTE, anchor="middle", cls="mute")
    s.path(f"M{px} {yb}H{px + plength}", "axis")
    s.path(f"M{xr(1.0):.2f} {yt}V{yb}", "axis")
    _margin_rule(s, xr(margin), yt, yb)
    s.text(xr(margin) + 7, yt + 18, f"margin {margin:.2f}", size=NOTE, cls="orange tint")
    s.text(px + plength + 20, yt + 18, f"ratio [{confidence} interval]", size=NOTE, cls="mute")
    for k, (arm, context, ctx_label, arm_label) in enumerate(CONTRASTS):
        contrast = contrasts[arm, context]
        hue, beyond = _contrast_style(arm, contrast, margin)
        y = yt + 52 + 38 * k
        s.text(px - 14, y + 7, f"{ctx_label} {arm_label}", size=VALUE, anchor="end")
        _interval(s, xr, contrast, y, hue)
        s.text(px + plength + 20, y + 7, _estimate(contrast), size=NOTE,
               cls="orange tint" if beyond else "soft")
    s.height = 372
    s.save("fig-cost.svg")
    render_cuda_narrow(segments, contrasts, margin, confidence, first, description)


def render_cuda_narrow(segments, contrasts, margin, confidence, first, description: str) -> None:
    """Phone layout: the bars over the ratios; each interval gets its own band under its
    label and estimate."""
    s = Figure(NARROW, 720, "Archived CUDA converter latency and complete decode cost", description)
    _titles(s, NARROW / 2, 30, "Converter segment, 32K context",
            "Pooled median, µs  •  Lower is better", size=LABEL)
    base, scale = 276, 0.72
    mark = len(s.items)
    _converter_bars(s, segments, 96, base, scale, width=52, pitch=104)
    _behind(s, mark, lambda: _yaxis(s, 74, 400, base, scale, (0, 100, 200)))

    px, plength = 28, 364

    def xr(value: float) -> float:
        return px + plength * (value - RATIO_AXIS[0]) / (RATIO_AXIS[1] - RATIO_AXIS[0])

    _titles(s, NARROW / 2, 384, "Complete step relative to atomic",
            f"Latency ratio, {confidence} interval  •  Lower is better", size=LABEL)
    # Each row owns a text line and a plot band, so no rule crosses a label.
    top_row, pitch, band = 470, 58, 26
    yb = top_row + pitch * len(CONTRASTS) + 4
    s.text(px + plength, top_row - 28, f"ratio [{confidence} interval]", size=NOTE,
           anchor="end", cls="mute")
    s.text(xr(margin) + 6, top_row - 2, f"margin {margin:.2f}", size=NOTE, cls="orange tint")
    for k, (arm, context, ctx_label, arm_label) in enumerate(CONTRASTS):
        contrast = contrasts[arm, context]
        hue, beyond = _contrast_style(arm, contrast, margin)
        y = top_row + 24 + pitch * k
        s.text(px, y, f"{ctx_label} {arm_label}", size=NOTE)
        s.text(px + plength, y, _estimate(contrast), size=NOTE, anchor="end",
               cls="orange tint" if beyond else "soft")
        y0 = y + 8
        s.rect(xr(margin), y0, xr(RATIO_AXIS[1]) - xr(margin), band, "g-margin")
        for tick in (t for t in RATIO_MINOR + RATIO_TICKS if t not in (1.0, margin)):
            s.path(f"M{xr(tick):.2f} {y0}v{band}", "grid")
        s.path(f"M{xr(1.0):.2f} {y0}v{band}", "axis")
        _margin_rule(s, xr(margin), y0, y0 + band)
        _interval(s, xr, contrast, y0 + band / 2, hue)
    s.path(f"M{px} {yb}H{px + plength}", "axis")
    for tick in RATIO_MINOR + RATIO_TICKS:
        s.path(f"M{xr(tick):.2f} {yb}v{5 if tick in RATIO_TICKS else 3}", "axis")
    for tick in RATIO_TICKS:
        s.text(xr(tick), yb + 24, f"{tick:.2f}", size=NOTE, anchor="middle", cls="mute")
    s.height = int(yb + 40)
    s.save("fig-cost-narrow.svg")


def main() -> None:
    render_apple()
    render_cuda()


if __name__ == "__main__":
    main()
