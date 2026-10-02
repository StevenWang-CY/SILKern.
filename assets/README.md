# Technical figures

The six `fig-*.svg` assets are generated, accessible vector figures. Edit their
sources and regenerate them; do not edit exported SVGs by hand.

```bash
python -m pip install -e ".[docs]"
python tools/render_diagrams.py
python tools/render_figures.py
```

Run these commands from the repository root. They check the worked localization
outputs against the Python oracle and read existing performance artifacts.
They do not run hardware experiments.

| Figure | Source | Content |
|---|---|---|
| `fig-platforms.svg` | `tools/render_diagrams.py` | Ownership, page-table translation, and stable compaction |
| `fig-contract.svg` | `tools/render_diagrams.py` | Coordinate equations and output layouts |
| `fig-consumer.svg` | `tools/render_diagrams.py` | Masked gathering and shared normalization |
| `fig-problem.svg` | `tools/render_diagrams.py` | Illustrative tile-order replays |
| `fig-apple-performance.svg` | `tools/render_figures.py` | [Apple record 09](../evidence/09-apple-mlx-consumer/) |
| `fig-cost.svg` | `tools/render_figures.py` | [Historical CUDA record 05](../evidence/05-full-decode-canary/) |

## Shared style

[`tools/figure_style.py`](../tools/figure_style.py) defines the font, two text
sizes, and light/dark palettes. Every figure uses a 1120-unit canvas, 18-unit
labels, and 20-unit panel headings. At an 840-pixel README width these display
at 13.5 and 15 pixels. Only panel letters and selected output values are bold.
Mathematical subscripts use 75% of the label size and remain attached to their
parent expressions. Put explanations and measurement scope in the adjacent
Markdown caption, keeping labels aligned to rows, axes, or objects.

Brackets denote vectors, fine outlines denote operations, and indexed matrix
rows denote K/V feature vectors. Filled and open bit marks reinforce numeric
validity. In the consumer figure, solid paths carry values and dashed paths
carry the mask; junction dots distinguish connected branches from crossings. Color follows
survivor identity, with short underlines emphasizing selected outputs. Keep the
canvas transparent and use subtle tints only inside selected memory rows.

In latency charts, blue denotes the highlighted implementation and gray the
comparison. Bar axes start at zero; the three Apple localization panels share
one scale. The consumer uses a separate scale, and the complete-decode ratio
chart retains its historical confidence intervals. Styling must never alter
the recorded measurements or their interpretation.

After regeneration, inspect all figures at README width in both light and dark
themes. Check connectors as well as text: a line must not cross a label, and
crossing routes must remain distinguishable. Confirm that captions still match
panel lettering, units, scales, and source evidence.

Typography and export checks follow the emphasis on editable text and final-size
legibility in [Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
Layout references included the [gpt-oss release charts](https://openai.com/index/introducing-gpt-oss/#evaluations)
and the architecture figures in the [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)
and [Kimi Linear report](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf).
The illustrations here are original SILKern diagrams; their data comes only
from this repository's examples and evidence.
