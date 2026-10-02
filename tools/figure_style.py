"""Shared typography and theme for the documentation's native SVG figures.

The 1120-unit canvas is normally displayed at about 840 CSS pixels in a README.
Use 18-unit labels (13.5 displayed pixels) and 20-unit panel titles.
Explanations belong in Markdown captions, not in a third tier of tiny annotations.
"""

WIDTH = 1120
LABEL = 18
HEADING = 20
FONT_FAMILIES = ("Arial", "DejaVu Sans")
FONT_FAMILY = ",".join(f"'{family}'" for family in FONT_FAMILIES) + ",sans-serif"

LIGHT = {
    "ink": "#242424",
    "muted": "#636363",
    "line": "#777777",
    "rule": "#d8d8d8",
    "paper": "#ffffff",
    "comparison": "#b9bec3",
    "blue": "#355e8a",
    "blue-wash": "#e8f0f8",
    "teal": "#47786d",
    "teal-wash": "#e7f2ef",
    "copper": "#9d6444",
    "copper-wash": "#f7eddf",
    "violet": "#786791",
    "violet-wash": "#eeebf5",
}
DARK = {
    "ink": "#e8edf2",
    "muted": "#aab4bf",
    "line": "#8d9aa8",
    "rule": "#37424d",
    "paper": "#0d1117",
    "comparison": "#778696",
    "blue": "#8ab7e6",
    "blue-wash": "#172c43",
    "teal": "#80c4b4",
    "teal-wash": "#17322e",
    "copper": "#dfb17e",
    "copper-wash": "#392c20",
    "violet": "#b6a7da",
    "violet-wash": "#292338",
}


def _variables(palette: dict[str, str]) -> str:
    return ";".join(f"--{name}:{color}" for name, color in palette.items())


THEME_CSS = (
    f"svg{{{_variables(LIGHT)}}}@media(prefers-color-scheme:dark){{svg{{{_variables(DARK)}}}}}"
)
TEXT_CSS = (
    f"text{{font-family:{FONT_FAMILY};font-size:{LABEL}px;"
    "font-variant-numeric:tabular-nums;fill:var(--ink)}"
    f".heading{{font-size:{HEADING}px;font-weight:400}}"
    ".panel,.key{font-weight:700}.subhead{font-weight:400}.math{font-style:italic}"
    ".muted{fill:var(--muted)}.data{font-weight:400}.emphasis{font-weight:700}"
)
