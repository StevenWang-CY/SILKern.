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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
INK = "#171717"
MUTED = "#525252"
RULE = "#e5e5e5"
NEUTRAL = "#737373"
ACCENT = "#245a96"
ADVERSE = "#a34e33"
COLORS = {
    INK: "ink",
    MUTED: "muted",
    RULE: "rule",
    NEUTRAL: "neutral",
    ACCENT: "accent",
    ADVERSE: "adverse",
}
WIDTH = 1120


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
        "<style>svg{--ink:#171717;--muted:#525252;--rule:#e5e5e5;"
        "--neutral:#737373;--accent:#245a96;--adverse:#a34e33}"
        "@media(prefers-color-scheme:dark){svg{--ink:#e5e5e5;"
        "--muted:#a3a3a3;--rule:#303030;--neutral:#a3a3a3;"
        "--accent:#8eb5de;--adverse:#e49c81}}</style>"
    )
    svg = svg[:end] + "\n" + accessibility + svg[end:]
    for color, variable in COLORS.items():
        svg = svg.replace(f"fill: {color}", f"fill: var(--{variable})")
        svg = svg.replace(f"stroke: {color}", f"stroke: var(--{variable})")
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 14,
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
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
        }
    )


def _figure(height: int):
    # One SVG point per layout unit; the README can scale the whole figure.
    return plt.figure(figsize=(WIDTH / 72, height / 72), dpi=72)


def _axis(fig, rectangle):
    ax = fig.add_axes(rectangle)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=4, pad=6)
    return ax


def _dumbbell(ax, cell, y, *, label_offset):
    native = cell["median_us"]["mlx_compiled"]
    metal = cell["median_us"]["metal_compiled"]
    ax.plot([metal, native], [y, y], color=NEUTRAL, linewidth=1.15, zorder=2)
    ax.plot(
        native,
        y,
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor=NEUTRAL,
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.plot(metal, y, marker="o", markersize=6, color=ACCENT, zorder=3)
    ax.annotate(
        f"{cell['compiled_ratio']:.2f}×",
        (native, y),
        xytext=(label_offset, 0),
        textcoords="offset points",
        color=MUTED,
        fontsize=13,
        va="center",
    )


def render_apple() -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    consumer = json.loads((record / "consumer-summary.json").read_text())
    first_session = json.loads((record / data["sessions"][0]).read_text())
    metadata = first_session["metadata"]
    device = metadata["device"]["device_name"]
    mlx_version = metadata["versions"]["mlx"]
    ratios = [cell["compiled_ratio"] for cell in data["cells"]]
    fig = _figure(410)
    ax = _axis(fig, [0.145, 0.16, 0.415, 0.65])
    ax2 = _axis(fig, [0.735, 0.16, 0.235, 0.65])
    fig.text(0.3525, 0.965, "(a) Index localization", fontsize=16, ha="center", va="top")
    fig.text(0.8525, 0.965, "(b) Attention consumer", fontsize=16, ha="center", va="top")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                markersize=6,
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=NEUTRAL,
                markeredgewidth=1.2,
                label="Compiled MLX",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markersize=6,
                linestyle="none",
                color=ACCENT,
                label="Compiled Metal",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.565, 0.905),
        ncols=2,
        frameon=False,
        fontsize=13,
        handletextpad=0.4,
        columnspacing=1.4,
        borderpad=0,
    )
    cells = sorted(data["cells"], key=lambda c: (c["geometry"]["width"], c["geometry"]["batch"]))
    positions = list(reversed(range(len(cells))))
    for y, cell in zip(positions, cells, strict=True):
        _dumbbell(ax, cell, y, label_offset=9)
    ax.set_yticks(
        positions, [f"{c['geometry']['batch']} × {c['geometry']['width']:,}" for c in cells]
    )
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.set_xlim(150, 218)
    ax.set_xticks([150, 160, 170, 180, 190, 200, 210])
    ax.set_xlabel("Latency (µs; lower is better)", labelpad=10, fontsize=14)
    ax.set_ylabel("Batch × selection width", labelpad=16, fontsize=14)
    consumer_cells = consumer["cells"]
    consumer_positions = list(reversed(range(len(consumer_cells))))
    for y, cell in zip(consumer_positions, consumer_cells, strict=True):
        _dumbbell(ax2, cell, y, label_offset=9)
    ax2.set_yticks(
        consumer_positions,
        [f"{c['geometry']['batch']} × {c['geometry']['width']:,}" for c in consumer_cells],
    )
    ax2.set_ylim(-0.5, len(consumer_cells) - 0.5)
    ax2.set_xlim(240, 410)
    ax2.set_xticks([240, 280, 320, 360, 400])
    ax2.set_xlabel("Latency (µs; lower is better)", labelpad=10, fontsize=14)
    _save_svg(
        fig,
        "fig-apple-performance.svg",
        f"{device} compiled localization and selected-attention measurements",
        f"Panel (a): compiled custom Metal is {min(ratios):.2f} to {max(ratios):.2f} times faster than compiled MLX "
        f"across {len(data['cells'])} geometries on {device} with MLX {mlx_version}. Geometry labels are batch × selection width. "
        "Open neutral circles denote MLX and blue circles denote Metal; annotations give the MLX / Metal speedup. "
        "The localization point-plot axis spans 150 to 218 microseconds. Panel (b) uses a separate 240 to 410 microsecond axis: "
        "the complete selected-attention consumer speedup is 1.12 times at batch 1 / width 128 and batch 8 / width 2048. "
        "Points are medians of three process-session medians. Both paths are compiled; timings include dispatch, allocation, "
        "execution and synchronization, excluding warmup. Consumer page tables and caches are static, with two logical shards "
        "on one device. Descriptive measurements with no confidence interval; not full-model or distributed performance. "
        f"Source: {APPLE_RECORD.as_posix()}/summary.json and consumer-summary.json.",
    )


def render_cuda() -> None:
    segments = json.loads((ROOT / "evidence/05-full-decode-canary/segments.json").read_text())
    analysis = json.loads((ROOT / "evidence/05-full-decode-canary/analysis.json").read_text())
    fig = _figure(370)
    ax = _axis(fig, [0.135, 0.19, 0.32, 0.66])
    ax2 = _axis(fig, [0.675, 0.19, 0.295, 0.66])
    fig.text(0.295, 0.965, "(a) Converter · 32K context", fontsize=16, ha="center", va="top")
    fig.text(0.8225, 0.965, "(b) Complete decode vs atomic", fontsize=16, ha="center", va="top")
    arms = [
        ("row_stable", "Rowwise", ACCENT, "o"),
        ("pinned_atomic", "Atomic", NEUTRAL, "s"),
        ("hierarchical_stable", "Hierarchical", NEUTRAL, "D"),
    ]
    for y, (arm, _label, color, marker) in zip([2, 1, 0], arms, strict=True):
        value = segments["contexts"]["32768"][f"converter.{arm}"]["pooled_median_us"]
        ax.plot([0, value], [y, y], color=color, linewidth=1.6, zorder=2)
        ax.plot(
            value,
            y,
            marker=marker,
            markersize=6,
            color=color,
            markerfacecolor="none" if arm == "hierarchical_stable" else color,
            markeredgewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{value:.1f}",
            (value, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=13,
            color=INK,
        )
    ax.set_yticks([2, 1, 0], [arm[1] for arm in arms])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, 275)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Converter segment latency (µs)", labelpad=10, fontsize=14)
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
            markersize=6,
            markerfacecolor=color if arm == "row_stable" else "none",
            markeredgewidth=1.2,
            linestyle="none",
            elinewidth=1.2,
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
        fontsize=12,
        ha="center",
        va="bottom",
    )
    ax2.set_yticks([3, 2, 1, 0], [f"{r[1] // 1024}K · {r[2]}" for r in rows])
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlim(0.993, 1.022)
    ax2.set_xticks([0.995, 1.000, 1.005, 1.010, 1.015, 1.020])
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax2.set_xlabel("Complete-step latency ratio", labelpad=10, fontsize=14)
    _save_svg(
        fig,
        "fig-cost.svg",
        "Archived CUDA converter latency and complete decode cost",
        "Panel (a): at 32K, the 48-layer converter segment takes 119.996 microseconds rowwise, 194.393 atomic, "
        "and 239.727 hierarchical; rowwise is 38.3 percent lower than atomic. Lollipop segments use a zero baseline; "
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


if __name__ == "__main__":
    main()
