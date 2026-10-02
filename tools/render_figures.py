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
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APPLE_RECORD = Path("evidence/09-apple-mlx-consumer")
COLORS = {
    "#f5f5f0": "paper",
    "#262a2a": "ink",
    "#676f6d": "muted",
    "#dadcd5": "grid",
    "#c6cdc8": "line",
}


def _save_svg(fig, filename: str, title: str, description: str) -> None:
    path = ROOT / "assets" / filename
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    svg = path.read_text()
    start = svg.index("<svg")
    end = svg.index(">", start)
    svg = svg[:end] + ' role="img" aria-labelledby="figure-title figure-desc"' + svg[end:]
    end = svg.index(">", start) + 1
    accessibility = (
        f'<title id="figure-title">{escape(title)}</title>'
        f'<desc id="figure-desc">{escape(description)}</desc>'
        "<style>svg{--paper:#f5f5f0;--ink:#262a2a;--muted:#676f6d;"
        "--grid:#dadcd5;--line:#c6cdc8}@media(prefers-color-scheme:dark){"
        "svg{--paper:#171d1c;--ink:#eef0eb;--muted:#abb8b1;"
        "--grid:#3b4842;--line:#53685e}}</style>"
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
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "silkern-evidence-figures",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "text.color": "#262a2a",
            "axes.labelcolor": "#676f6d",
            "xtick.color": "#676f6d",
            "ytick.color": "#262a2a",
            "figure.facecolor": "#f5f5f0",
            "axes.facecolor": "#f5f5f0",
            "savefig.facecolor": "#f5f5f0",
        }
    )


def render_cuda() -> None:
    segments = json.loads((ROOT / "evidence/05-full-decode-canary/segments.json").read_text())
    analysis = json.loads((ROOT / "evidence/05-full-decode-canary/analysis.json").read_text())
    fig = plt.figure(figsize=(11.2, 4.9), dpi=100)
    fig.text(0.04, 0.925, "ARCHIVED NVIDIA EVIDENCE", fontsize=10, color="#676f6d", weight="bold")
    fig.text(0.04, 0.85, "Measure the converter. Check the consumer.", fontsize=23, weight="bold")
    fig.text(
        0.04,
        0.795,
        "Two H100s · live DCP-2 · five sessions · complete 48-layer reference executor",
        fontsize=11,
        color="#676f6d",
    )
    ax = fig.add_axes([0.17, 0.285, 0.285, 0.38])
    ax2 = fig.add_axes([0.665, 0.285, 0.285, 0.38])
    colors = ["#286bb6", "#a3aaa5", "#c7754a"]
    labels = ["Rowwise", "Atomic", "Hierarchical"]
    vals = [
        segments["contexts"]["32768"]["converter." + name]["pooled_median_us"]
        for name in ["row_stable", "pinned_atomic", "hierarchical_stable"]
    ]
    ax.barh([2, 1, 0], vals, color=colors, height=0.38, zorder=3)
    ax.set_yticks([2, 1, 0], labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(0, 285)
    ax.set_xticks([0, 100, 200])
    ax.grid(axis="x", color="#dadcd5", zorder=0)
    for y, v in zip([2, 1, 0], vals, strict=True):
        ax.text(v + 5, y, f"{v:.1f}", va="center", fontsize=10, weight="bold")
    ax.set_xlabel("µs per converter segment at 32K", labelpad=12, fontsize=10)
    fig.text(0.04, 0.705, "01  CONVERTER SEGMENT", fontsize=10, weight="bold")
    fig.text(0.54, 0.705, "02  COMPLETE-STEP RATIO", fontsize=10, weight="bold")
    rows = [
        ("row_stable", 32768, "Rowwise · 32K", colors[0]),
        ("hierarchical_stable", 32768, "Hier. · 32K", colors[2]),
        ("row_stable", 65536, "Rowwise · 64K", colors[0]),
        ("hierarchical_stable", 65536, "Hier. · 64K", colors[2]),
    ]
    for y, (arm, ctx, _label, color) in zip([3, 2, 1, 0], rows, strict=True):
        c = analysis["contrasts"][f"{arm}_over_atomic.c{ctx}"]
        ax2.errorbar(
            c["point"],
            y,
            xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]],
            fmt="o",
            color=color,
            markersize=6,
            capsize=3,
            lw=1.7,
            zorder=3,
        )
    ax2.axvline(1.0, color="#676f6d", lw=1)
    ax2.axvline(1.01, color="#b85b30", lw=1, ls=(0, (3, 3)))
    ax2.set_yticks([3, 2, 1, 0], [r[2] for r in rows])
    ax2.tick_params(axis="y", length=0, pad=9)
    ax2.set_ylim(-0.55, 3.55)
    ax2.set_xlim(0.993, 1.021)
    ax2.set_xticks([1, 1.01, 1.02])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))
    ax2.set_xlabel("Ratio to atomic · 98.75% intervals", labelpad=12, fontsize=10)
    ax2.text(1.0105, 3.56, "1.01 margin", fontsize=9, color="#b85b30", va="bottom")
    fig.text(0.04, 0.12, "38.3% lower rowwise segment time at 32K.", fontsize=11, weight="bold")
    fig.text(0.54, 0.12, "Rowwise at 64K is about 1.4% slower.", fontsize=11, weight="bold")
    fig.text(
        0.04,
        0.055,
        "Source: evidence/05-full-decode-canary · Historical results; no new NVIDIA experiments for this update.",
        fontsize=9,
        color="#676f6d",
    )
    _save_svg(
        fig,
        "fig-cost.svg",
        "Archived CUDA converter and complete-step measurements",
        "At 32K, the complete 48-layer converter segment takes 119.996 microseconds "
        "rowwise, 194.393 atomic, and 239.727 hierarchical. Whole-step ratios with "
        "98.75 percent intervals show rowwise at 64K is 1.014006 times atomic, above "
        "the prespecified 1.01 margin. All other contrasts meet the margin. "
        "These are historical two-H100 results.",
    )


def render_apple() -> None:
    record = ROOT / APPLE_RECORD
    data = json.loads((record / "summary.json").read_text())
    first_session = json.loads((record / data["sessions"][0]).read_text())
    metadata = first_session["metadata"]
    device = metadata["device"]["device_name"]
    mlx_version = metadata["versions"]["mlx"]
    session_count = data["session_count"]
    geometry_count = len(data["cells"])
    ratios = [cell["compiled_ratio"] for cell in data["cells"]]
    ratio_range = f"{min(ratios):.2f}–{max(ratios):.2f}×"
    widths = sorted({cell["geometry"]["width"] for cell in data["cells"]})
    latencies = [
        cell["median_us"][arm]
        for cell in data["cells"]
        for arm in ("mlx_compiled", "metal_compiled")
    ]
    padding = max((max(latencies) - min(latencies)) * 0.2, 1)
    xlim = (min(latencies) - padding, max(latencies) + padding)
    fig = plt.figure(figsize=(11.2, 5.6), dpi=100)
    fig.text(
        0.04,
        0.93,
        "APPLE SILICON · NATIVE LOCALIZATION",
        fontsize=10,
        color="#676f6d",
        weight="bold",
    )
    fig.text(0.04, 0.86, f"{ratio_range} over compiled MLX.", fontsize=24, weight="bold")
    fig.text(
        0.04,
        0.807,
        f"{device} · MLX {mlx_version} · {session_count} process sessions · "
        f"{geometry_count} geometries · both paths compiled",
        fontsize=11,
        color="#676f6d",
    )
    handles = [
        Line2D([0], [0], marker="o", color="#9baba4", lw=0, markersize=7, label="MLX composition"),
        Line2D([0], [0], marker="o", color="#286bb6", lw=0, markersize=7, label="Custom Metal"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.028, 0.763),
        ncol=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    column_width = 0.945 / len(widths)
    for col, width in enumerate(widths):
        ax = fig.add_axes([0.11 + column_width * col, 0.29, column_width - 0.09, 0.34])
        ax.set_title(f"WIDTH {width:,}", fontsize=11, loc="left", pad=14, weight="bold")
        cells = sorted(
            (c for c in data["cells"] if c["geometry"]["width"] == width),
            key=lambda c: c["geometry"]["batch"],
        )
        positions = list(reversed(range(len(cells))))
        for y, c in zip(positions, cells, strict=True):
            native = c["median_us"]["mlx_compiled"]
            metal = c["median_us"]["metal_compiled"]
            ax.plot([metal, native], [y, y], color="#c6cdc8", lw=3, zorder=1)
            ax.scatter([native], [y], color="#9baba4", s=46, zorder=3)
            ax.scatter([metal], [y], color="#286bb6", s=46, zorder=3)
            ax.text(
                (native + metal) / 2,
                y + 0.19,
                f"{c['compiled_ratio']:.2f}×",
                ha="center",
                fontsize=10,
                weight="bold",
            )
        ax.set_yticks(positions, [f"Batch {c['geometry']['batch']}" for c in cells])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.45, len(cells) - 0.45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="x", color="#dadcd5", zorder=0)
        ax.set_xlabel("µs per evaluated call", labelpad=10, fontsize=10)
    fig.text(
        0.04,
        0.15,
        "Lower latency is better. Ratio = MLX / Metal; dots are medians of "
        f"{session_count} session medians.",
        fontsize=10,
        color="#676f6d",
    )
    fig.text(
        0.04,
        0.102,
        "Includes Python dispatch, output allocation, execution, and evaluation. Compilation warmup excluded.",
        fontsize=10,
        color="#676f6d",
    )
    fig.text(
        0.04,
        0.054,
        f"Source: {APPLE_RECORD.as_posix()} · Descriptive microbenchmark; "
        "no confidence interval or model-throughput claim.",
        fontsize=9,
        color="#676f6d",
    )
    _save_svg(
        fig,
        "fig-apple-performance.svg",
        f"{device} localization benchmark",
        f"Compiled custom Metal achieves a {min(ratios):.2f} to {max(ratios):.2f} "
        "times speedup relative to compiled compositional MLX across all "
        f"{geometry_count} measured geometries on {device} with MLX {mlx_version}. "
        f"Widths: {', '.join(map(str, widths))}. Latencies include "
        "Python dispatch, output allocation, execution, and synchronization. "
        f"{session_count} process sessions; primitive microbenchmark, not model performance.",
    )


def main() -> None:
    _configure()
    render_cuda()
    render_apple()


if __name__ == "__main__":
    main()
