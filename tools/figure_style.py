"""Shared typography and theme for the documentation's native SVG figures.

The 1120-unit canvas is normally displayed at about 840 CSS pixels in a README.
Use two sizes only: 18-unit labels (13.5 displayed pixels) and 22-unit headings.
Explanations belong in Markdown captions, not in a third tier of tiny annotations.
"""

WIDTH = 1120
LABEL = 18
HEADING = 22
FONT_FAMILIES = ("Arial", "DejaVu Sans")
FONT_FAMILY = ",".join(f"'{family}'" for family in FONT_FAMILIES) + ",sans-serif"

LIGHT = {
    "ink": "#202832",
    "muted": "#58616c",
    "line": "#84909b",
    "rule": "#d4dbe1",
    "paper": "#ffffff",
    "neutral": "#f1f3f5",
    "comparison": "#b5bfca",
    "blue": "#32669b",
    "blue-wash": "#e8f0f8",
    "teal": "#287c72",
    "teal-wash": "#e7f2ef",
    "copper": "#a16a3b",
    "copper-wash": "#f7eddf",
    "violet": "#786a9e",
    "violet-wash": "#eeebf5",
    "knockout": "#ffffff",
}
DARK = {
    "ink": "#e8edf2",
    "muted": "#aab4bf",
    "line": "#8d9aa8",
    "rule": "#37424d",
    "paper": "#0d1117",
    "neutral": "#1b232c",
    "comparison": "#778696",
    "blue": "#8ab7e6",
    "blue-wash": "#172c43",
    "teal": "#80c4b4",
    "teal-wash": "#17322e",
    "copper": "#dfb17e",
    "copper-wash": "#392c20",
    "violet": "#b6a7da",
    "violet-wash": "#292338",
    "knockout": "#111820",
}


def _variables(palette: dict[str, str]) -> str:
    return ";".join(f"--{name}:{color}" for name, color in palette.items())


THEME_CSS = (
    f"svg{{{_variables(LIGHT)}}}@media(prefers-color-scheme:dark){{svg{{{_variables(DARK)}}}}}"
)
TEXT_CSS = (
    f"text{{font-family:{FONT_FAMILY};font-size:{LABEL}px;"
    "font-variant-numeric:tabular-nums;fill:var(--ink)}"
    f".heading{{font-size:{HEADING}px;font-weight:700}}"
    ".subhead{font-weight:700}.muted{fill:var(--muted)}"
    ".data{font-weight:400}.data.knockout{font-weight:700}"
)
