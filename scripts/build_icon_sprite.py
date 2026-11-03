"""Build banko_ai/static/vendor/lucide-sprite.svg from lucide-static icon files.

    uv run python scripts/build_icon_sprite.py /tmp/banko-vendor/lucide-static-*/package/icons banko_ai/static/vendor/lucide-sprite.svg

Only the icons in ICONS are included. Add a name here, rebuild, and commit both."""

import re
import sys
from pathlib import Path

ICONS = (
    "layout-dashboard", "landmark", "arrow-left-right", "chart-column", "piggy-bank", "repeat", "target",
    "sparkles", "lightbulb", "bot", "server", "settings", "search", "sun", "moon", "monitor", "x", "plus",
    "send", "mic", "volume-2", "copy", "square", "paperclip", "camera", "chevron-right", "chevron-down",
    "external-link", "trash-2", "pencil", "check", "log-out", "menu", "download", "refresh-cw",
    "circle-alert", "info", "receipt", "message-square", "maximize-2", "minimize-2",
)
_INNER = re.compile(r"<svg[^>]*>(.*)</svg>", re.S)


def build(icon_dir: Path) -> str:
    symbols = []
    for name in ICONS:
        svg = (icon_dir / f"{name}.svg").read_text()
        inner = _INNER.search(svg).group(1).strip()
        symbols.append(f'<symbol id="i-{name}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                       f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</symbol>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" style="display:none">\n'
            + "\n".join(symbols) + "\n</svg>\n")


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.write_text(build(src))
    print(f"wrote {out} with {len(ICONS)} icons")
