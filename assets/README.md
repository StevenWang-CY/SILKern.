# Figures and brand assets

The six figure families are generated, accessible SVGs with wide and narrow
layouts. Edit their sources and regenerate them; do not edit exports by hand.

```bash
python -m pip install -e ".[docs]"
python tools/render_diagrams.py
python tools/render_figures.py
```

Run these commands from the repository root. They check the worked
localization examples against the Python oracle and read existing performance
records; they do not run hardware experiments.

| Figure | Source | Content |
|---|---|---|
| `fig-platforms.svg` | `tools/render_diagrams.py` | Ownership, logical-page to physical-block translation, stable compaction |
| `fig-contract.svg` | `tools/render_diagrams.py` | Worked coordinate table, equations, and output layouts |
| `fig-consumer.svg` | `tools/render_diagrams.py` | Layout masks, masked gathering, and shared normalization |
| `fig-problem.svg` | `tools/render_diagrams.py` | Illustrative tile-order replays |
| `fig-apple-performance.svg` | `tools/render_figures.py` | [Apple record 09](../evidence/09-apple-mlx-consumer/) |
| `fig-cost.svg` | `tools/render_figures.py` | [Historical CUDA record 05](../evidence/05-full-decode-canary/) |

## Typography

Figures are set in [STIX Two](https://github.com/stipub/stixfonts) Text and
Math, under the SIL Open Font License ([`tools/fonts/OFL.txt`](../tools/fonts/OFL.txt)).
Each export embeds a WOFF subset of exactly the glyphs it uses, so GitHub shows
the same typography on every platform without network requests, while labels
remain selectable SVG text. [`tools/figure_style.py`](../tools/figure_style.py)
measures the same font files to lay out and check every label.

Wide figures use a 1120-unit canvas, shown at about 840 CSS pixels in a README.
Labels, equations, and subcaptions use 21 units (about 16 pixels); tensor
entries use 20, and headers, ticks, and bar values use 18. Narrow layouts
restack the same panels, or redraw a panel, on a 420-unit canvas so phone text
stays at a similar size. Panels carry LaTeX-style subcaptions, "(a) Title",
below the drawing; explanation and measurement scope belong in the Markdown
caption.

Mathematics is typeset from a small LaTeX subset: italic variables, upright
operator names such as `mod` and `divmod`, true minus signs, and TeX spacing
around relations, binary operators, and sums.

## Color and marks

Ink, rules, and fills follow GitHub's own neutrals, with a separate dark
palette selected by `prefers-color-scheme`. In diagrams, blue, orange, and green
follow individual surviving tokens, derived from the identity's two threads;
violet marks the fourth tile in the replay figure. Pale washes tint only the
cells a hue owns.

- Continuous strips with hairline dividers are tensors; dots after an open end
  mean the strip continues.
- A light band across a strip is an operation applied to every column;
  connectors pass beneath it.
- Solid connectors carry values, dashed connectors carry masks, and a junction
  dot joins branches.
- A shaded zone holds physical KV memory.
- Booktabs rules, without vertical lines, organize tables.

In charts, blue is the measured implementation and gray the comparison; orange
marks the adverse result and the region beyond a margin. Every latency scale
starts at zero. The Apple dot plot places all eleven geometries on one scale
and lists each speedup in a column, so no value is repeated as a label; bar
labels in the CUDA chart repeat recorded medians to one decimal place. Interval
charts keep their recorded confidence levels and margins. Styling must never
alter a measurement or its interpretation.

## Checks

`Figure.save` rejects overlapping labels, connectors that cross a label, and
text outside the canvas. Regenerating must reproduce identical bytes: run both
generators twice and confirm that `git status` shows no further change. Then
inspect every figure at desktop and 340-pixel widths in light and dark themes,
and confirm that captions still match panel letters, units, scales, and source
evidence.

Typography and export choices follow the emphasis on editable text and
final-size legibility in [Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
Layout studies included the architecture and kernel figures of the
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) and
[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) reports, the
[Kimi Linear](https://arxiv.org/abs/2510.26692) and
[Kimi K2](https://arxiv.org/abs/2507.20534) reports,
[Native Sparse Attention](https://arxiv.org/abs/2502.11089), and
[FlashAttention](https://arxiv.org/abs/2205.14135), together with the release
charts for [gpt-oss](https://openai.com/index/introducing-gpt-oss/) and
[Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5). The
illustrations are original SILKern diagrams; their data comes only from this
repository's examples and evidence.

## Brand assets

The wordmark (`logo-text.svg`), icon (`logo.svg`), and K mark (`logo-k.svg`)
are self-contained SVGs with dark-theme variants. `banner.svg` is the
1280 × 640 social card; `cover.png` is its raster for the repository's social
preview setting, so re-export it at the same size whenever the banner changes.
