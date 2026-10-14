# Figures and brand assets

Every SVG here is generated. Edit the generators and regenerate; do not edit an
export by hand. [`tests/test_figures.py`](../tests/test_figures.py) regenerates
all of them into a scratch directory and fails if a committed file differs or
has no generator.

```bash
python -m pip install -e ".[docs]"
python tools/render_diagrams.py
python tools/render_figures.py
python tools/render_brand.py
```

Run these commands from the repository root. They check the worked
localization examples against the Python oracle and read existing performance
records; they do not run hardware experiments.

| Figure | Source | Content |
|---|---|---|
| `fig-system.svg` | `tools/render_diagrams.py` | One decode step on two ranks: selector, localization on each rank, attention, and the cross-rank combination |
| `fig-localization.svg` | `tools/render_diagrams.py` | Ownership, logical-page to physical-block translation, stable compaction |
| `fig-contract.svg` | `tools/render_diagrams.py` | Worked coordinate table, equations, and output layouts |
| `fig-consumer.svg` | `tools/render_diagrams.py` | Layout masks, masked gathering, and shared normalization |
| `fig-problem.svg` | `tools/render_diagrams.py` | Illustrative tile-order replays |
| `fig-scan.svg` | `tools/render_diagrams.py` | How the rowwise, hierarchical, and Metal kernels number survivors |
| `fig-apple-performance.svg` | `tools/render_figures.py` | [Apple record 09](../evidence/09-apple-mlx-consumer/) |
| `fig-cost.svg` | `tools/render_figures.py` | [Historical CUDA record 05](../evidence/05-full-decode-canary/) |

Each figure has a `-narrow.svg` companion for phone-width pages.

## Typography

Figures are set in [STIX Two](https://github.com/stipub/stixfonts) Text and
Math, under the SIL Open Font License ([`tools/fonts/OFL.txt`](../tools/fonts/OFL.txt)).
Each export embeds a WOFF subset of exactly the glyphs it uses, so GitHub shows
the same typography on every platform without network requests, while labels
remain selectable SVG text. [`tools/figure_style.py`](../tools/figure_style.py)
measures the same font files to lay out and check every label.

Wide figures use a 1120-unit canvas, shown at about 840 CSS pixels in a README.
Labels, equations, and panel titles use 21 units (about 16 pixels); tensor
entries and category labels use 20, and notes, headers, ticks, and bar values
use 18. A chart's title uses 23. The README figures state the 21-unit label size
on the canvas itself, so a label without its own size never falls back to a
browser's smaller default. Every glyph in the README figures is set in regular
weight; emphasis comes from position, hue, and size, never from bold. Narrow
layouts restack the same panels, or redraw a figure, on a 420-unit canvas so
phone text stays at a similar size.

In the README illustrations each panel carries its title at the top left and its
letter, "(a)", at the top right; the scan figure keeps LaTeX-style subcaptions
below its panels. Each README chart opens with a centred title over a gray
subtitle that names the measure and says which way is better. Explanation and
measurement scope belong in the Markdown caption.

Mathematics is typeset from a small LaTeX subset: italic variables, upright
operator names such as `mod` and `divmod`, true minus signs, and TeX spacing
around relations, binary operators, and sums.

## Color and marks

Text is set in ink, as in print. Gray is kept for quiet marks such as padding,
indices, and tick labels, and every label clears 4.5:1 contrast in both themes.
Neutrals follow GitHub's, with a separate dark palette selected by
`prefers-color-scheme`. Each figure paints its own page color behind everything:
a browser applies the system's theme inside an SVG image, which can differ from
the theme a reader chose on GitHub, and an opaque page keeps the figure legible
when they disagree.

In diagrams, blue, orange, and green follow individual surviving tokens, derived
from the logo's two threads; violet marks the fourth tile in the replay figure,
and gray marks tokens that belong to another rank. A hue fills the cells it owns
strongly enough to read as an object, with digits left in ink; large regions,
such as table columns and the margin band, take a pale area tint instead.

- Continuous strips with hairline dividers are tensors; dots after an open end
  mean the strip continues. In the README figures a tensor sits on an opaque
  page-colored base, so its open cells stay white on a panel's ground.
- In the README illustrations every component is a rounded box with a navy
  outline: SILKern's own in pale blue, everything else white. An operation
  applied to every column is such a box laid across the strip; connectors pass
  beneath it. The scan figure keeps the earlier form, a stage ruled above and
  below, with SILKern's own stage ruled in the accent color.
- Each panel of a README illustration sits on a pale ground inside a dashed
  border; neighbouring panels alternate a lavender and a gray ground, and the
  two ranks of the system figure share one.
- Dataflow between components runs on pale steel-blue arrows of one weight and
  one head size, and a junction dot joins or splits them. An arrow never ends in
  a gap shorter than about two heads: the generator refuses one.
- A selector, which keeps a few of many candidates, is drawn as a funnel.
- Solid connectors in a token's hue carry its values, dashed connectors carry
  masks. Where curves cross, the one drawn later is cased in its panel's ground
  and passes over.
- A dashed outline over a faint ground holds physical KV memory; addresses that
  point into it sit outside it.
- Booktabs rules, without vertical lines, organize tables.
- Figures carry no pictorial icons; names and geometry identify every part.

The README charts follow the layout of a model-release chart: a title and a gray
subtitle, flat bars with a hairline outline and slightly rounded tops, values
set just above their bars, light gridlines behind them, and a small swatch
legend. SILKern's measured arm is a saturated blue, its other kernel a pale
blue, and the comparison gray; orange marks the adverse result and the region
beyond a margin. Every latency scale starts at zero. The Apple chart sets each
geometry's two bars side by side on one scale for all eleven geometries, with
the speedup above each pair; its phone layout gives every geometry a row, with
the speedup at the right. Bar labels in the CUDA chart repeat recorded medians
to one decimal place, and each bar is named twice, by kernel and by owner.
Interval charts keep their recorded confidence levels and margins, with labeled
ticks only where the precision is uniform. Styling must never alter a
measurement or its interpretation.

## Checks

`Figure.save` rejects overlapping labels, connectors that cross a label, text
outside the canvas, a narrow-layout crop that would cut through a label, and
text set in a gray too light to read. Regenerating must reproduce identical
bytes: run the generators twice and confirm that `git status` shows no further
change. Then inspect every figure at desktop and 340-pixel widths in light and
dark themes, and confirm that captions still match panel letters, units, scales,
and source evidence.

Typography and export choices follow the emphasis on editable text and
final-size legibility in [Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
Layout studies included the architecture and kernel figures of the
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) and
[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) reports, the
[Kimi Linear](https://arxiv.org/abs/2510.26692) and
[Kimi K2](https://arxiv.org/abs/2507.20534) reports,
[Native Sparse Attention](https://arxiv.org/abs/2502.11089),
[FlashAttention](https://arxiv.org/abs/2205.14135), and the
[PagedAttention](https://arxiv.org/abs/2309.06180) paper, together with the release
charts for [gpt-oss](https://openai.com/index/introducing-gpt-oss/),
[Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5), and
[Gemini 4 Argon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/). The
illustrations are original SILKern diagrams; their data comes only from this
repository's examples and evidence.

## Brand assets

`tools/render_brand.py` draws the brand from code:

| File | Content |
|---|---|
| `logo.svg` | The mark: three index threads enter spread out and leave packed against a rail. The blue thread passes over the ink thread and under the orange one, so the three read as woven. |
| `logo-text.svg` | The mark beside the wordmark, "SIL" in Manrope ExtraBold and "Kern" in Manrope Medium, converted to outlines |
| `banner.svg` | The 1280 × 640 social card |
| `icons/cpu.svg`, `icons/soc.svg`, `icons/gpu.svg` | README icons for a processor, a system on a chip, and an accelerator module |

The mark and wordmark adapt to the reader's theme, and their gaps are masks, so
they work on any background. Manrope is used under the SIL Open Font License
([`tools/fonts/Manrope-OFL.txt`](../tools/fonts/Manrope-OFL.txt)); only the seven
glyphs of the name are kept, in `tools/fonts/Manrope-SILKern.ttf`.

The icons follow GitHub's own octicons: a 16-unit grid, one 1.5-unit stroke with
round caps and joins, at most one solid detail, and one ink color per theme. They
mark hardware classes, not vendors, and appear only beside platform names: in
the README's tables, at 16 pixels, and on the social card.

`cover.png` is the repository's social preview: a 1280 × 640 raster of
`banner.svg`. Re-export it whenever the banner changes, for example by opening
`banner.svg` in a browser at that size and saving a screenshot, then upload it in
the repository's settings.
