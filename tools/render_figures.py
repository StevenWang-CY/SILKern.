"""Regenerate the evidence-backed SVG charts from checked-in measurement records.

    python -m pip install -e ".[docs]"
    python tools/render_figures.py

Every plotted value and every number in the accessible descriptions is read from
``evidence/``; nothing here runs a benchmark. Charts share the typography and
palette of the explanatory diagrams. Bars start at zero, and direct labels
repeat the recorded medians to one decimal place.
"""

from __future__ import annotations

import json
from pathlib import Path

from figure_style import NARROW, NOTE, ROOT, WIDTH, Figure

APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
CUDA_RECORD = Path("evidence/05-full-decode-canary")
MLX, METAL = "mlx_compiled", "metal_compiled"
LOCALIZATION_TOP, CONSUMER_TOP = 240, 400  # shared zero-based latency scales, µs
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
RATIO_TICKS = (0.995, 1.0, 1.005, 1.01, 1.015, 1.02)


def _xaxis(s: Figure, x0, length, y_top, y_axis, ticks, top, *, title: str) -> None:
    """Vertical gridlines, a zero baseline, tick values and an axis title."""
    for tick in ticks:
        x = x0 + length * tick / top
        s.path(f"M{x:.2f} {y_top:.2f}V{y_axis:.2f}", "grid" if tick else "axis")
        s.text(x, y_axis + 20, f"{tick:g}", size=NOTE, anchor="middle", cls="faint")
    s.text(x0 + length / 2, y_axis + 42, title, size=NOTE, anchor="middle", cls="mute")


def _paired_bars(s: Figure, x0, y0, length, top, rows) -> None:
    """Rows of (label, comparison, highlighted) values as paired horizontal bars."""
    for k, (label, compare, highlight) in enumerate(rows):
        y = y0 + 44 * k
        s.text(x0 - 10, y + 19, label, size=NOTE, anchor="end")
        for dy, value, fill, cls, weight in (
            (0, compare, "compare", "soft", "regular"),
            (15, highlight, "accent tint", "accent tint", "semibold"),
        ):
            w = length * value / top
            s.rect(x0, y + dy, w, 12, fill)
            s.text(x0 + w + 6, y + dy + 10.5, f"{value:.1f}", size=NOTE, cls=cls, weight=weight)


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


def _ratio_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").ljust(4, "0")


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


def _legend(s: Figure, x: float, y: float, entries) -> None:
    for label, fill in entries:
        s.rect(x, y - 10, 12, 12, fill)
        box = s.text(x + 18, y, label, size=NOTE, cls="soft")
        x = box.x1 + 22


def render_apple() -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    consumer = json.loads((record / "consumer-summary.json").read_text())
    session = json.loads((record / data["sessions"][0]).read_text())
    device = session["metadata"]["device"]["device_name"]
    mlx_version = session["metadata"]["versions"]["mlx"]
    ratios = [cell["compiled_ratio"] for cell in data["cells"]]
    widths = sorted({cell["geometry"]["width"] for cell in data["cells"]})
    batches = sorted({cell["geometry"]["batch"] for cell in data["cells"]})
    consumer_shapes = " and ".join(
        f"batch {c['geometry']['batch']} with width {c['geometry']['width']}"
        for c in consumer["cells"]
    )

    s = Figure(WIDTH, 286, f"{device} compiled localization and selected-attention latency", "")
    s.description = (
        "Paired horizontal bars compare compiled compositional MLX (gray) with compiled custom "
        f"Metal (blue) on {device} with MLX {mlx_version}. Panels a, b and c are localization at "
        f"selection widths {', '.join(map(str, widths))} for batch sizes "
        f"{', '.join(map(str, batches))}, on one zero-based 0 to {LOCALIZATION_TOP} microsecond "
        f"scale. Metal is {min(ratios):.2f} to {max(ratios):.2f} times faster across the "
        f"{len(ratios)} geometries. Panel d is complete selected attention at {consumer_shapes}, "
        f"on a separate zero-based 0 to {CONSUMER_TOP} microsecond scale, with two logical shards "
        "on one device, fixed caches and 64-dimensional keys and values. Direct labels give each "
        "median to one decimal place. Bars are medians of three process-session medians and "
        "include dispatch, allocation, execution and synchronization; warmup and first-use "
        "compilation are excluded. No confidence intervals are implied, and these are not "
        f"full-model results. Source: {APPLE_RECORD.as_posix()}/summary.json and "
        "consumer-summary.json."
    )
    _legend(s, 24, 30, (("compiled MLX", "compare"), ("compiled Metal", "accent tint")))
    s.text(WIDTH - 24, 30, "median latency, lower is better", size=NOTE, anchor="end", cls="mute")
    y0, length = 72, 148
    panels = []
    for index, width in enumerate(widths):
        x0 = 58 + 252 * index
        cells = sorted(
            (c for c in data["cells"] if c["geometry"]["width"] == width),
            key=lambda c: c["geometry"]["batch"],
        )
        rows = [
            (str(c["geometry"]["batch"]), c["median_us"][MLX], c["median_us"][METAL]) for c in cells
        ]
        s.text(x0 - 10, y0 - 10, "batch", size=NOTE, anchor="end", cls="faint")
        _xaxis(
            s, x0, length, y0 - 4, y0 + 128, (0, 100, 200), LOCALIZATION_TOP, title="latency (µs)"
        )
        _paired_bars(s, x0, y0, length, LOCALIZATION_TOP, rows)
        s.subcaption(x0 + 74, 274, "abc"[index], f"Width {width:,}")
        panels.append((x0 - 50, 50, 244, 236))
    x0 = 872
    rows = [
        (
            f"{c['geometry']['batch']} × {c['geometry']['width']:,}",
            c["median_us"][MLX],
            c["median_us"][METAL],
        )
        for c in consumer["cells"]
    ]
    s.text(x0 - 10, y0 - 10, "batch × width", size=NOTE, anchor="end", cls="faint")
    _xaxis(s, x0, 160, y0 - 4, y0 + 128, (0, 200, 400), CONSUMER_TOP, title="latency (µs)")
    _paired_bars(s, x0, y0 + 22, 160, CONSUMER_TOP, rows)
    s.subcaption(x0 + 46, 274, "d", "Selected attention")
    panels.append((x0 - 104, 50, 342, 236))
    s.save("fig-apple-performance.svg", panels=[(16, 12, 420, 28, 12), *panels])


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
    s.rect(xr(margin), yt, xr(RATIO_AXIS[1]) - xr(margin), yb - yt, "orange wash")
    for tick in RATIO_TICKS:
        s.path(f"M{xr(tick):.2f} {yt}V{yb}", "grid")
        s.text(xr(tick), yb + 20, _ratio_label(tick), size=NOTE, anchor="middle", cls="faint")
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
    s.text(px + plength + 24, yt + 14, f"ratio [{confidence} interval]", size=NOTE, cls="faint")
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
    top_row, pitch, band = 322, 58, 26
    yb = top_row + pitch * len(CONTRASTS) + 4
    s.text(px, top_row - 2, f"ratio [{confidence} interval]", size=NOTE, cls="faint")
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
        s.rect(xr(margin), y0, xr(RATIO_AXIS[1]) - xr(margin), band, "orange wash")
        for tick in (t for t in RATIO_TICKS if t not in (1.0, margin)):
            s.path(f"M{xr(tick):.2f} {y0}v{band}", "grid")
        s.path(f"M{xr(1.0):.2f} {y0}v{band}", "axis")
        _margin_rule(s, xr(margin), y0, y0 + band)
        _interval(s, xr, contrast, y0 + band / 2, hue)
    s.path(f"M{px} {yb}H{px + plength}", "axis")
    for tick in RATIO_TICKS:
        s.path(f"M{xr(tick):.2f} {yb}v5", "axis")
        s.text(xr(tick), yb + 24, _ratio_label(tick), size=NOTE, anchor="middle", cls="faint")
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
