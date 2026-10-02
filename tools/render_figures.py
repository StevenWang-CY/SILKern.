"""Regenerate evidence-backed SVG figures: pip install -e '.[docs]'; python tools/render_figures.py.

Source values are read from checked-in artifacts. Fonts remain SVG text for
accessibility. Fixed SVG IDs and omitted timestamps make regeneration stable
within the same Matplotlib version. Hardware benchmarks are never invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from figure_style import FONT_FAMILIES, HEADING, LABEL, LIGHT, THEME_CSS, WIDTH

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
INK = LIGHT["ink"]
MUTED = LIGHT["muted"]
RULE = LIGHT["rule"]
NEUTRAL = LIGHT["line"]
ACCENT = LIGHT["blue"]
ADVERSE = LIGHT["copper"]
COMPARISON = LIGHT["comparison"]
COLORS = {
    LIGHT[name]: name for name in ("ink", "muted", "rule", "line", "blue", "copper", "comparison")
}


def _save_svg(fig, filename: str, title: str, description: str) -> None:
    # Measure the actual glyphs before export; this catches clipped annotations
    # when labels or evidence values change without rasterizing the SVG's text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(renderer)
            if (
                bounds.x0 < 0
                or bounds.y0 < 0
                or bounds.x1 > fig.bbox.width
                or bounds.y1 > fig.bbox.height
            ):
                raise ValueError(f"Figure text falls outside the canvas: {artist.get_text()!r}")
    path = ROOT / "assets" / filename
    fig.savefig(path, format="svg", transparent=True, metadata={"Date": None})
    plt.close(fig)
    svg = path.read_text()
    start = svg.index("<svg")
    end = svg.index(">", start)
    svg = svg[:end] + ' role="img" aria-labelledby="figure-title figure-desc"' + svg[end:]
    end = svg.index(">", start) + 1
    accessibility = (
        f'<title id="figure-title">{escape(title)}</title>'
        f'<desc id="figure-desc">{escape(description)}</desc>'
        f"<style>{THEME_CSS}text{{font-variant-numeric:tabular-nums}}</style>"
    )
    svg = svg[:end] + "\n" + accessibility + svg[end:]
    for color, variable in COLORS.items():
        svg = svg.replace(f"fill: {color}", f"fill: var(--{variable})")
        svg = svg.replace(f"stroke: {color}", f"stroke: var(--{variable})")
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": list(FONT_FAMILIES),
            "font.size": LABEL,
            "svg.fonttype": "none",
            "svg.hashsalt": "silkern-evidence-figures",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": NEUTRAL,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.labelsize": LABEL,
            "ytick.labelsize": LABEL,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
        }
    )


def _figure(height: int, width: int = WIDTH):
    # One SVG point per layout unit; the README can scale the whole figure.
    return plt.figure(figsize=(width / 72, height / 72), dpi=72)


def _panel_title(fig, x, y, letter, title):
    height = fig.get_figheight() * 72
    width = fig.get_figwidth() * 72
    fig.text(x / width, 1 - y / height, letter, fontsize=HEADING, weight="bold")
    fig.text((x + 24) / width, 1 - y / height, title, fontsize=HEADING, weight="normal")


def _axis(fig, rectangle):
    ax = fig.add_axes(rectangle)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=4, pad=6)
    return ax


def _bar_panel(fig, rectangle, cells, *, limit, ticks, labels, narrow=False):
    ax = fig.add_axes(rectangle)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=0.7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=0, pad=10)
    # All four panels use the same physical bar width, including the two-group consumer.
    positions = list(range(3)) if len(cells) == 3 else [0.4, 1.6]
    spacing = 0.23 if narrow else 0.205
    for arm, offset, color in (
        ("mlx_compiled", -spacing, COMPARISON),
        ("metal_compiled", spacing, ACCENT),
    ):
        heights = [cell["median_us"][arm] for cell in cells]
        bars = ax.bar(
            [v + offset for v in positions],
            heights,
            width=0.38 if narrow else 0.34,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=3,
        )
        ax.bar_label(
            bars, labels=[f"{v:.1f}" for v in heights], padding=6, color=INK, fontsize=LABEL
        )
    ax.set_xticks(positions, labels)
    ax.set_yticks(ticks)
    ax.set_ylim(0, limit)
    ax.set_xlim(-0.6, 2.6)
    return ax


def render_apple(*, narrow=False) -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    consumer = json.loads((record / "consumer-summary.json").read_text())
    first_session = json.loads((record / data["sessions"][0]).read_text())
    metadata = first_session["metadata"]
    device = metadata["device"]["device_name"]
    mlx_version = metadata["versions"]["mlx"]
    ratios = [cell["compiled_ratio"] for cell in data["cells"]]
    height, canvas_width = (1380, 420) if narrow else (672, WIDTH)
    fig = _figure(height, canvas_width)
    fig.text(24 / canvas_width, 1 - 30 / height, "Latency (µs)", fontsize=LABEL)
    fig.legend(
        handles=[
            Patch(facecolor=COMPARISON, edgecolor=INK, linewidth=0.45, label="Compiled MLX"),
            Patch(facecolor=ACCENT, edgecolor=INK, linewidth=0.45, label="Compiled Metal"),
        ],
        loc="upper right",
        bbox_to_anchor=((canvas_width - 24) / canvas_width, 1 - (52 if narrow else 10) / height),
        ncols=2,
        frameon=False,
        fontsize=LABEL,
        handlelength=1.2,
        handleheight=1,
        handletextpad=0.55,
        columnspacing=1.5,
        borderpad=0,
        borderaxespad=0,
    )
    for index, width in enumerate((128, 2048, 4096, None)):
        x, y = (
            (64, 150 + index * 320)
            if narrow
            else (80 + (index % 2) * 570, 108 + (index // 2) * 300)
        )
        is_consumer = width is None
        cells = (
            consumer["cells"]
            if is_consumer
            else sorted(
                (c for c in data["cells"] if c["geometry"]["width"] == width),
                key=lambda c: c["geometry"]["batch"],
            )
        )
        title = "Selected attention" if is_consumer else f"Localization, width {width:,}"
        _panel_title(fig, 24 if narrow else x - 56, y - 32, "abcd"[index], title)
        ax = _bar_panel(
            fig,
            [
                x / canvas_width,
                1 - (y + 185) / height,
                (332 if narrow else 428) / canvas_width,
                185 / height,
            ],
            cells,
            limit=440 if is_consumer else 240,
            ticks=[0, 100, 200, 300, 400] if is_consumer else [0, 50, 100, 150, 200],
            labels=[
                f"{c['geometry']['batch']} × {c['geometry']['width']:,}"
                if is_consumer
                else str(c["geometry"]["batch"])
                for c in cells
            ],
            narrow=narrow,
        )
        ax.set_xlabel(
            "Batch × selection width" if is_consumer else "Batch size", labelpad=14, fontsize=LABEL
        )
    _save_svg(
        fig,
        "fig-apple-performance-narrow.svg" if narrow else "fig-apple-performance.svg",
        f"{device} compiled localization and selected-attention measurements",
        f"Grouped bars compare compiled compositional MLX in gray with compiled custom Metal in blue on {device}, MLX {mlx_version}. "
        f"Four panels form {'one vertical column' if narrow else 'a two-by-two grid'}. "
        "Panels a, b, c are localization at widths 128, 2048, 4096, "
        "with batch sizes 1, 8, 32 and identical zero-based 0–240 microsecond scales. "
        f"Metal is {min(ratios):.2f} to {max(ratios):.2f} times faster across the nine localization geometries. "
        "Panel d covers complete selected attention at batch 1 / width 128 and batch 8 / width 2048 "
        "on a separate zero-based 0–440 microsecond scale. Each consumer uses two logical shards on one device, "
        "fixed caches and 64-dimensional keys and values. Direct labels give latency to one decimal place. "
        "Bars are medians of three process-session medians; timings include dispatch, allocation, execution and synchronization. "
        "Warmup and first-use compilation are excluded. No confidence intervals are implied. These are not full-model or distributed results. "
        f"Source: {APPLE_RECORD.as_posix()}/summary.json and consumer-summary.json.",
    )


def render_cuda(*, narrow=False) -> None:
    segments = json.loads((ROOT / "evidence/05-full-decode-canary/segments.json").read_text())
    analysis = json.loads((ROOT / "evidence/05-full-decode-canary/analysis.json").read_text())
    fig = _figure(790, 420) if narrow else _figure(422)
    ax = _axis(
        fig,
        [120 / 420, 1 - 320 / 790, 270 / 420, 220 / 790] if narrow else [0.165, 0.21, 0.292, 0.56],
    )
    ax2 = _axis(
        fig,
        [170 / 420, 1 - 720 / 790, 224 / 420, 240 / 790] if narrow else [0.689, 0.21, 0.285, 0.56],
    )
    _panel_title(fig, 24, 36, "a", "Converter, 32K context")
    _panel_title(
        fig, 24 if narrow else 626, 416 if narrow else 36, "b", "Complete decode vs atomic"
    )
    arms = [
        ("row_stable", "Rowwise", ACCENT, "o"),
        ("pinned_atomic", "Atomic", NEUTRAL, "s"),
        ("hierarchical_stable", "Hierarchical", NEUTRAL, "D"),
    ]
    for y, (arm, _label, color, _marker) in zip([2, 1, 0], arms, strict=True):
        value = segments["contexts"]["32768"][f"converter.{arm}"]["pooled_median_us"]
        ax.barh(y, value, height=0.52, color=color if arm == "row_stable" else COMPARISON, zorder=3)
        ax.annotate(
            f"{value:.1f}",
            (value, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=LABEL,
            color=INK,
        )
    ax.set_yticks([2, 1, 0], [arm[1] for arm in arms])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, 275)
    ax.set_xticks([0, 100, 200] if narrow else [0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Converter latency (µs)", labelpad=10, fontsize=LABEL)
    rows = [
        ("row_stable", 32768, "Rowwise"),
        ("hierarchical_stable", 32768, "Hierarchical"),
        ("row_stable", 65536, "Rowwise"),
        ("hierarchical_stable", 65536, "Hierarchical"),
    ]
    margin = analysis["tier1_margin"]
    for y, (arm, ctx, _label) in zip([3, 2, 1, 0], rows, strict=True):
        contrast = analysis["contrasts"][f"{arm}_over_atomic.c{ctx}"]
        color = ADVERSE if contrast["hi"] > margin else ACCENT if arm == "row_stable" else NEUTRAL
        ax2.errorbar(
            contrast["point"],
            y,
            xerr=[[contrast["point"] - contrast["lo"]], [contrast["hi"] - contrast["point"]]],
            color=color,
            marker="o" if arm == "row_stable" else "D",
            markersize=8,
            markerfacecolor=color if arm == "row_stable" else "none",
            markeredgewidth=1.2,
            linestyle="none",
            elinewidth=1.5,
            capsize=3,
            capthick=1,
            zorder=3,
        )
    ax2.axvline(1.0, color=NEUTRAL, linewidth=0.9)
    ax2.axvline(margin, color=NEUTRAL, linewidth=0.9, linestyle=(0, (3, 3)))
    ax2.text(
        margin,
        1.015,
        f"{margin:.2f} margin",
        transform=ax2.get_xaxis_transform(),
        color=MUTED,
        fontsize=LABEL,
        ha="center",
        va="bottom",
    )
    ax2.set_yticks([3, 2, 1, 0], [f"{r[1] // 1024}K · {r[2]}" for r in rows])
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlim(0.993, 1.022)
    ax2.set_xticks([1.00, 1.01, 1.02])
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.set_xlabel("Complete-step latency ratio", labelpad=10, fontsize=LABEL)
    _save_svg(
        fig,
        "fig-cost-narrow.svg" if narrow else "fig-cost.svg",
        "Archived CUDA converter latency and complete decode cost",
        "Panel (a): at 32K, the 48-layer converter segment takes 119.996 microseconds rowwise, 194.393 atomic, "
        "and 239.727 hierarchical; rowwise is 38.3 percent lower than atomic. Horizontal bars use a zero baseline; "
        "visible labels round to one decimal place. Panel (b): complete-step ratio estimates and 98.75 percent intervals "
        "are shown against atomic on a 0.993 to 1.022 axis. Solid vertical line: equal latency; dashed line: prespecified 1.01 margin. "
        "Rust marks rowwise at 64K, which is 1.014006 times atomic and whose interval exceeds the margin. "
        "All other contrasts meet the margin. Lower latency or ratio is better. Five sessions; panel (a) uses pooled segment medians, "
        "and panel (b) uses session-level complete-step estimates. These are historical two-H100 live-DCP-2 reference-executor results, "
        "not serving-runtime performance. No NVIDIA experiments were run for this update. "
        "Source: evidence/05-full-decode-canary/segments.json and analysis.json.",
    )


def main() -> None:
    _configure()
    render_cuda()
    render_apple()
    render_cuda(narrow=True)
    render_apple(narrow=True)


if __name__ == "__main__":
    main()
