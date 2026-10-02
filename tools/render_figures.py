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
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
INK = "#202832"
MUTED = "#58616c"
RULE = "#d4dbe1"
NEUTRAL = "#84909b"
ACCENT = "#32669b"
ADVERSE = "#a16a3b"
COMPARISON = "#b5bfca"
COLORS = {
    INK: "ink",
    MUTED: "muted",
    RULE: "rule",
    NEUTRAL: "neutral",
    ACCENT: "accent",
    ADVERSE: "adverse",
    COMPARISON: "comparison",
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
        "<style>svg{--ink:#202832;--muted:#58616c;--rule:#d4dbe1;"
        "--neutral:#84909b;--accent:#32669b;--adverse:#a16a3b;--comparison:#b5bfca}"
        "@media(prefers-color-scheme:dark){svg{--ink:#e5e5e5;"
        "--muted:#a3a3a3;--rule:#303030;--neutral:#a3a3a3;"
        "--accent:#8ab7e6;--adverse:#dfb17e;--comparison:#778696}}</style>"
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


def _bar_panel(fig, rectangle, cells, *, title, limit, ticks, labels, show_y=True):
    ax = fig.add_axes(rectangle)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7, labelleft=show_y)
    ax.tick_params(axis="x", length=0, pad=10)
    x = list(range(len(cells)))
    for arm, offset, color in (
        ("mlx_compiled", -0.19, COMPARISON),
        ("metal_compiled", 0.19, ACCENT),
    ):
        heights = [cell["median_us"][arm] for cell in cells]
        bars = ax.bar(
            [v + offset for v in x],
            heights,
            width=0.34,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            zorder=3,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in heights], padding=5, color=INK, fontsize=12)
    ax.set_xticks(
        x,
        [
            f"{label}\n{cell['compiled_ratio']:.2f}×"
            for label, cell in zip(labels, cells, strict=True)
        ],
    )
    ax.set_yticks(ticks)
    ax.set_ylim(0, limit)
    ax.set_xlim(-0.62, len(cells) - 0.38)
    ax.set_title(title, fontsize=15, color=INK, pad=16)
    return ax


def render_apple() -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    consumer = json.loads((record / "consumer-summary.json").read_text())
    first_session = json.loads((record / data["sessions"][0]).read_text())
    metadata = first_session["metadata"]
    device = metadata["device"]["device_name"]
    mlx_version = metadata["versions"]["mlx"]
    ratios = [cell["compiled_ratio"] for cell in data["cells"]]
    fig = _figure(414)
    fig.text(0.022, 0.965, "a  Index localization", fontsize=18, weight="bold", va="top")
    fig.text(0.765, 0.965, "b  Attention consumer", fontsize=18, weight="bold", va="top")
    fig.legend(
        handles=[
            Patch(facecolor=COMPARISON, label="Compiled MLX"),
            Patch(facecolor=ACCENT, label="Compiled Metal"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.49, 0.975),
        ncols=2,
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handleheight=1,
        handletextpad=0.5,
        columnspacing=1.5,
        borderpad=0,
    )
    for index, width in enumerate((128, 2048, 4096)):
        cells = sorted(
            (c for c in data["cells"] if c["geometry"]["width"] == width),
            key=lambda c: c["geometry"]["batch"],
        )
        ax = _bar_panel(
            fig,
            [0.065 + index * 0.224, 0.205, 0.193, 0.58],
            cells,
            title=f"Selection width {width:,}",
            limit=235,
            ticks=[0, 50, 100, 150, 200],
            labels=[str(c["geometry"]["batch"]) for c in cells],
            show_y=index == 0,
        )
        if index == 0:
            ax.set_ylabel("Latency (µs)", labelpad=12, fontsize=14)
    fig.text(
        0.382, 0.035, "Batch size · speedup (MLX / Metal)", fontsize=13, color=MUTED, ha="center"
    )
    ax = _bar_panel(
        fig,
        [0.794, 0.205, 0.183, 0.58],
        consumer["cells"],
        title="Complete selected attention",
        limit=440,
        ticks=[0, 100, 200, 300, 400],
        labels=[
            f"{c['geometry']['batch']} × {c['geometry']['width']:,}" for c in consumer["cells"]
        ],
    )
    fig.text(0.89, 0.035, "Batch × width · speedup", fontsize=13, color=MUTED, ha="center")
    fig.add_artist(
        plt.Line2D(
            [0.745, 0.745], [0.12, 0.91], transform=fig.transFigure, color=RULE, linewidth=0.9
        )
    )
    _save_svg(
        fig,
        "fig-apple-performance.svg",
        f"{device} compiled localization and selected-attention measurements",
        f"Grouped bars compare compiled compositional MLX in gray with compiled custom Metal in blue on {device}, MLX {mlx_version}. "
        f"Panel a contains three equal-scale facets at selection widths 128, 2048, 4096, each with batches 1, 8, 32. "
        f"Metal is {min(ratios):.2f} to {max(ratios):.2f} times faster across the nine localization geometries. "
        "All bar axes start at zero: localization uses 0–235 microseconds and consumer uses 0–440 microseconds. "
        "Direct labels give latency to one decimal place; labels beneath each batch give MLX / Metal speedup to two decimals. "
        "Panel b covers two complete selected-attention consumers: batch 1 / width 128 and batch 8 / width 2048. "
        "Each has two logical shards on one device, fixed caches, and 64-dimensional keys and values. "
        "Bars are medians of three process-session medians; timings include dispatch, allocation, execution and synchronization. "
        "Warmup and first-use compilation are excluded. No confidence intervals are implied. These are not full-model or distributed results. "
        f"Source: {APPLE_RECORD.as_posix()}/summary.json and consumer-summary.json.",
    )


def render_cuda() -> None:
    segments = json.loads((ROOT / "evidence/05-full-decode-canary/segments.json").read_text())
    analysis = json.loads((ROOT / "evidence/05-full-decode-canary/analysis.json").read_text())
    fig = _figure(370)
    ax = _axis(fig, [0.135, 0.19, 0.32, 0.66])
    ax2 = _axis(fig, [0.675, 0.19, 0.295, 0.66])
    fig.text(
        0.295,
        0.965,
        "a  Converter · 32K context",
        fontsize=18,
        weight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.8225,
        0.965,
        "b  Complete decode vs atomic",
        fontsize=18,
        weight="bold",
        ha="center",
        va="top",
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


if __name__ == "__main__":
    main()
