"""Regenerate the evidence-backed SVG charts from checked-in measurement records.

    python -m pip install -e ".[docs]"
    python tools/render_figures.py

Every plotted value and every number in the accessible descriptions is read from
``evidence/``; nothing here runs a benchmark. Charts share the typography and
palette of the explanatory diagrams. Every latency scale starts at zero, and
bar labels repeat the recorded medians to one decimal place.
"""

from __future__ import annotations

import json
from pathlib import Path

from figure_style import NARROW, NOTE, ROOT, WIDTH, Figure

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


def _xaxis(s: Figure, x0, length, y_top, y_axis, ticks, top, *, title: str) -> None:
    """Vertical gridlines, a zero baseline, tick values and an axis title."""
    for tick in ticks:
        x = x0 + length * tick / top
        s.path(f"M{x:.2f} {y_top:.2f}V{y_axis:.2f}", "grid" if tick else "axis")
        s.text(x, y_axis + 20, f"{tick:g}", size=NOTE, anchor="middle", cls="mute")
    s.text(x0 + length / 2, y_axis + 42, title, size=NOTE, anchor="middle", cls="mute")


def _converter_bars(s: Figure, segments, x0, y0, length, top) -> None:
    """The 32K converter segment: rowwise highlighted against atomic and hierarchical."""
    for k, (arm, label) in enumerate(ARMS):
        value = segments["contexts"]["32768"][f"converter.{arm}"]["pooled_median_us"]
        y = y0 + 50 * k
        ours = arm == "row_stable"
        w = length * value / top
        s.text(x0 - 10, y + 15, label, anchor="end")
        s.rect(x0, y, w, 20, "accent tint" if ours else "compare")
        s.text(
            x0 + w + 6,
            y + 15,
            f"{value:.1f}",
            size=NOTE,
            cls="accent tint" if ours else "soft",
            weight="semibold" if ours else "regular",
        )


def _estimate(contrast) -> str:
    return f"{contrast['point']:.4f}  [{contrast['lo']:.4f}, {contrast['hi']:.4f}]"


def _contrast_style(arm: str, contrast, margin: float) -> tuple[str | None, bool]:
    """Orange marks an interval beyond the margin; blue the rowwise arm; gray the rest."""
    adverse = contrast["hi"] > margin
    return ("orange" if adverse else "accent" if arm == "row_stable" else None), adverse


def _interval(s: Figure, xr, contrast, y: float, hue: str | None) -> None:
    """A confidence interval with end caps; open markers recede, filled ones lead."""
    stroke = f"{hue} trace" if hue else "wire"
    s.line([(xr(contrast["lo"]), y), (xr(contrast["hi"]), y)], cls=stroke, collide=False)
    for end in ("lo", "hi"):
        s.line([(xr(contrast[end]), y - 5), (xr(contrast[end]), y + 5)], cls=stroke, collide=False)
    x = xr(contrast["point"])
    if hue:
        s.circle(x, y, 4.2, f"{hue} tint")
    else:
        s.circle(x, y, 3.8, "paper")
        s.raw(f'<circle cx="{x:.2f}" cy="{y}" r="3.8" class="wire" style="stroke-width:1.3"/>')


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


def _latency_dots(
    s: Figure, groups, *, columns, x0, length, y, pitch, ticks, grid, radius, label_first
) -> None:
    """Dumbbell rows on one zero-based scale, with a right-hand column of ratios.

    ``groups`` holds (heading, rows, split) triples; ``split`` adds space between
    selection widths. ``columns`` gives the left edge of the headings and the
    right edges of the width, batch and ratio columns. A hollow dot marks
    compiled MLX and a filled dot compiled Metal; a light span joins each pair
    and a dotted leader carries the row to its ratio. Gridlines at ``grid`` run
    only through each group's rows, so no rule crosses a heading.
    ``label_first`` names the two marks in place, on the first heading line.
    """
    heading_x, width_x, batch_x, ratio_x = columns

    def xs(value: float) -> float:
        return x0 + length * value / LATENCY_TOP

    s.text(width_x, y, "width", size=NOTE, anchor="end", cls="mute")
    s.text(batch_x, y, "batch", size=NOTE, anchor="end", cls="mute")
    ratio_head = s.text(ratio_x, y, "MLX / Metal", size=NOTE, anchor="end", cls="mute")
    ratio_left = ratio_head.x0
    marks = []
    for index, (name, rows, split) in enumerate(groups):
        y += pitch + 10
        s.text(heading_x, y, name, size=NOTE, cls="mute", weight="italic")
        if index == 0 and label_first:
            _, _, mlx, metal, _ = rows[0]
            s.text(xs(metal) - radius - 7, y, "compiled Metal", size=NOTE, anchor="end",
                   cls="accent tint")
            s.text(xs(mlx) + radius + 7, y, "compiled MLX", size=NOTE, cls="soft")
        top, previous = y + 8, None
        for width, batch, mlx, metal, ratio in rows:
            y += pitch + (12 if split and previous is not None and width != previous else 0)
            if width != previous:
                s.text(width_x, y, f"{width:,}", size=NOTE, anchor="end")
            s.text(batch_x, y, str(batch), size=NOTE, anchor="end")
            box = s.text(ratio_x, y, f"{ratio:.2f}×", size=NOTE, anchor="end")
            ratio_left = min(ratio_left, box.x0)
            marks.append((y - 6, mlx, metal))
            previous = width
        bottom = y + pitch / 2
        for tick in grid:
            s.path(f"M{xs(tick):.2f} {top:.2f}V{bottom:.2f}", "grid" if tick else "axis")
    axis_y = y + pitch / 2 + 6
    s.path(f"M{x0} {axis_y:.2f}H{x0 + length}", "axis")
    for tick in ticks:
        s.path(f"M{xs(tick):.2f} {axis_y:.2f}v5", "axis")
        s.text(xs(tick), axis_y + 24, f"{tick:g}", size=NOTE, anchor="middle", cls="mute")
    s.text(x0 + length / 2, axis_y + 48, "median latency per call (µs)", size=NOTE,
           anchor="middle", cls="mute")
    for cy, mlx, metal in marks:
        s.path(f"M{xs(mlx) + radius + 5:.2f} {cy:.2f}H{ratio_left - 8:.2f}", "leader")
        s.path(f"M{xs(metal):.2f} {cy:.2f}H{xs(mlx) - radius:.2f}", "span")
        s.circle(xs(mlx), cy, radius, "ring")
        s.circle(xs(metal), cy, radius, "accent tint")
    s.height = int(axis_y + 64)


def _dot_legend(s: Figure, x: float, y: float, radius: float) -> None:
    s.circle(x + radius, y - 6, radius, "accent tint")
    box = s.text(x + 2 * radius + 8, y, "compiled Metal", size=NOTE, cls="accent tint")
    x = box.x1 + 24
    s.circle(x + radius, y - 6, radius, "ring")
    s.text(x + 2 * radius + 8, y, "compiled MLX", size=NOTE, cls="soft")


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
        f"Dot plot of median latency per synchronized call on {device} with MLX {mlx_version}, "
        f"on one zero-based 0 to {LATENCY_TOP} microsecond scale. Each row joins compiled "
        "compositional MLX (hollow gray dot) and the compiled custom Metal kernel (filled blue "
        f"dot). The first {len(localization)} rows are localization at selection widths "
        f"{', '.join(f'{w:,}' for w in sorted({r[0] for r in localization}))} and batch sizes "
        f"{', '.join(map(str, sorted({r[1] for r in localization})))}; Metal is "
        f"{min(ratios):.2f} to {max(ratios):.2f} times faster. The last {len(attention)} rows are "
        "complete selected attention with two logical shards on one device, fixed caches and "
        "64-dimensional keys and values: "
        + "; ".join(f"batch {b} with width {w:,}, {r:.2f} times" for w, b, _, _, r in attention)
        + ". The right-hand column gives each ratio, compiled MLX latency divided by compiled "
        "Metal latency. Values are medians of three process-session medians and include "
        "dispatch, allocation, execution and synchronization; warmup and first-use compilation "
        f"are excluded. {_session_note(record, data, consumer)} Source: "
        f"{APPLE_RECORD.as_posix()}/summary.json, consumer-summary.json and the session records."
    )

    s = Figure(WIDTH, 600, title, description)
    _latency_dots(s, groups, columns=(40, 152, 212, 1062), x0=256, length=700, y=30,
                  pitch=26, ticks=LATENCY_TICKS, grid=LATENCY_TICKS, radius=5.5,
                  label_first=True)
    s.save("fig-apple-performance.svg")

    # Phone layout: the labels move into the empty 0-150 µs zone, so the scale
    # keeps its zero and the gaps stay visible at 340 pixels.
    n = Figure(NARROW, 600, title, description)
    _dot_legend(n, 14, 30, 4)
    _latency_dots(n, groups, columns=(14, 60, 112, 406), x0=14, length=340, y=70,
                  pitch=24, ticks=(0, 200, 400), grid=(200, 300, 400), radius=4,
                  label_first=False)
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
        "Panel a: at 32K context the 48-layer converter segment takes "
        f"{medians['row_stable']:.3f} microseconds rowwise, {medians['pinned_atomic']:.3f} atomic "
        f"and {medians['hierarchical_stable']:.3f} hierarchical; rowwise is "
        f"{1 - medians['row_stable'] / medians['pinned_atomic']:.1%} lower than atomic. Bars start "
        "at zero and labels round to one decimal place. Panel b: complete-step latency ratios "
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
    s = Figure(WIDTH, 318, "Archived CUDA converter latency and complete decode cost", description)

    # (a) Converter segment at 32K --------------------------------------------
    x0, length, top = 132, 196, 250
    _xaxis(s, x0, length, 56, 232, (0, 100, 200), top, title="converter latency (µs)")
    _converter_bars(s, segments, x0, 84, length, top)
    s.subcaption(x0 + length / 2 - 30, 308, "a", "Converter segment, 32K context")

    # (b) Complete-step ratios ----------------------------------------------------
    px, plength = 590, 300

    def xr(value: float) -> float:
        return px + plength * (value - RATIO_AXIS[0]) / (RATIO_AXIS[1] - RATIO_AXIS[0])

    yt, yb = 56, 232
    s.rect(xr(margin), yt, xr(RATIO_AXIS[1]) - xr(margin), yb - yt, "orange area")
    for tick in RATIO_MINOR + RATIO_TICKS:
        s.path(f"M{xr(tick):.2f} {yt}V{yb}", "grid")
    for tick in RATIO_TICKS:
        s.text(xr(tick), yb + 20, f"{tick:.2f}", size=NOTE, anchor="middle", cls="mute")
    s.path(f"M{px} {yb}H{px + plength}", "axis")
    s.path(f"M{xr(1.0):.2f} {yt}V{yb}", "axis")
    _margin_rule(s, xr(margin), yt, yb)
    s.text(xr(margin) + 6, yt + 14, f"margin {margin:.2f}", size=NOTE, cls="orange tint")
    s.text(
        px + plength / 2,
        yb + 42,
        "complete-step latency ratio to atomic",
        size=NOTE,
        anchor="middle",
        cls="mute",
    )
    s.text(px + plength + 24, yt + 14, f"ratio [{confidence} interval]", size=NOTE, cls="mute")
    for k, (arm, context, ctx_label, arm_label) in enumerate(CONTRASTS):
        contrast = contrasts[arm, context]
        hue, beyond = _contrast_style(arm, contrast, margin)
        y = yt + 46 + 38 * k
        s.text(px - 12, y + 6, f"{ctx_label} {arm_label}", anchor="end")
        _interval(s, xr, contrast, y, hue)
        s.text(
            px + plength + 24,
            y + 6,
            _estimate(contrast),
            size=NOTE,
            cls=f"{hue} tint" if beyond else "soft",
        )
    s.subcaption(px + plength / 2 + 80, 308, "b", "Complete decode step relative to atomic")
    s.save("fig-cost.svg")
    render_cuda_narrow(segments, contrasts, margin, confidence, description)


def render_cuda_narrow(segments, contrasts, margin, confidence, description: str) -> None:
    """Phone layout: each interval gets its own band under its label and estimate."""
    s = Figure(NARROW, 640, "Archived CUDA converter latency and complete decode cost", description)
    x0, length, top = 128, 196, 250
    _xaxis(s, x0, length, 24, 186, (0, 100, 200), top, title="converter latency (µs)")
    _converter_bars(s, segments, x0, 40, length, top)
    s.subcaption(NARROW / 2, 270, "a", "Converter segment, 32K context")

    px, plength = 28, 364

    def xr(value: float) -> float:
        return px + plength * (value - RATIO_AXIS[0]) / (RATIO_AXIS[1] - RATIO_AXIS[0])

    # Each row owns a text line and a plot band, so no rule crosses a label.
    top_row, pitch, band = 340, 58, 26
    yb = top_row + pitch * len(CONTRASTS) + 4
    s.text(px + plength, top_row - 28, f"ratio [{confidence} interval]", size=NOTE,
           anchor="end", cls="mute")
    s.text(xr(margin) + 6, top_row - 2, f"margin {margin:.2f}", size=NOTE, cls="orange tint")
    for k, (arm, context, ctx_label, arm_label) in enumerate(CONTRASTS):
        contrast = contrasts[arm, context]
        hue, beyond = _contrast_style(arm, contrast, margin)
        y = top_row + 24 + pitch * k
        s.text(px, y, f"{ctx_label} {arm_label}", size=NOTE)
        s.text(
            px + plength,
            y,
            _estimate(contrast),
            size=NOTE,
            anchor="end",
            cls=f"{hue} tint" if beyond else "soft",
        )
        y0 = y + 8
        s.rect(xr(margin), y0, xr(RATIO_AXIS[1]) - xr(margin), band, "orange area")
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
    s.text(
        px + plength / 2,
        yb + 46,
        "complete-step latency ratio to atomic",
        size=NOTE,
        anchor="middle",
        cls="mute",
    )
    s.subcaption(NARROW / 2, yb + 82, "b", "Complete step relative to atomic")
    s.height = int(yb + 94)
    s.save("fig-cost-narrow.svg")


def main() -> None:
    render_apple()
    render_cuda()


if __name__ == "__main__":
    main()
