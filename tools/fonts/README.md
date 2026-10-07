# Figure fonts

These files set the text of the generated documentation figures. They are
[STIX Two](https://github.com/stipub/stixfonts) 2.13 b171, from the
[Google Fonts distribution](https://github.com/google/fonts/tree/main/ofl/stixtwotext),
licensed under the SIL Open Font License 1.1 ([OFL.txt](OFL.txt)).

| File | Source |
|---|---|
| `STIXTwoText-Regular.ttf` | `STIXTwoText[wght].ttf`, static instance at weight 400 |
| `STIXTwoText-SemiBold.ttf` | `STIXTwoText[wght].ttf`, static instance at weight 600 |
| `STIXTwoText-Italic.ttf` | `STIXTwoText-Italic[wght].ttf`, static instance at weight 400 |
| `STIXTwoMath-Regular.ttf` | `STIXTwoMath-Regular.ttf` |

Instances were made with fontTools and trimmed to the Unicode ranges the
figures can use: Latin, Greek, punctuation, letterlike symbols, and the minus
sign for the text faces; arrows, mathematical operators, and technical symbols
for the math face. Each exported SVG embeds a further subset containing only the
glyphs it draws.
