"""The design system: both themes define every token, and every icon a
template uses exists in the sprite."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
CSS = (ROOT / "static" / "css" / "app.css").read_text()
TOKENS = ("--bg", "--side", "--card", "--bd", "--bd2", "--tx", "--mu", "--ac", "--acsoft", "--pos", "--neg",
          "--warn", "--track", "--chip", "--sh", "--ins", "--instx", "--insbd", "--r", "--chart-1", "--chart-2",
          "--chart-grid")


def _block(selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\n\}", CSS, re.S)
    assert match, selector
    return match.group(1)


def test_both_themes_define_every_token():
    day, night = _block(":root"), _block('[data-theme="dark"]')
    for token in TOKENS:
        assert f"{token}:" in day, f"Daylight misses {token}"
        if token not in ("--r", "--ac"):
            assert f"{token}:" in night, f"Midnight misses {token}"


def test_single_accent_color_rule():
    day, night = _block(":root"), _block('[data-theme="dark"]')
    assert "--ac:" in day, "Accent color must be defined in :root"
    assert "--ac:" not in night, "Accent color must not be redefined in dark theme (set once on <html>)"


def test_daylight_and_midnight_use_the_approved_values():
    day, night = _block(":root"), _block('[data-theme="dark"]')
    assert "--bg: #F6F5F2" in day and "--card: #FFFFFF" in day and "--r: 14px" in day
    assert "--bg: #090B12" in night and "--card: #121626" in night


def test_no_external_urls_in_css():
    assert not re.search(r"url\((['\"]?)https?:", CSS)
    assert "@import" not in CSS


def test_every_icon_used_in_templates_exists():
    sprite = (ROOT / "static" / "vendor" / "lucide-sprite.svg").read_text()
    have = set(re.findall(r'<symbol id="i-([a-z0-9-]+)"', sprite))
    used = set()
    for tpl in (ROOT / "templates").rglob("*.html"):
        used |= set(re.findall(r"""icon\(\s*['"]([a-z0-9-]+)['"]""", tpl.read_text()))
    for js in (ROOT / "static" / "js").rglob("*.js"):
        used |= set(re.findall(r"""['"]#i-([a-z0-9-]+)['"]""", js.read_text()))
    assert used <= have, f"missing icons: {sorted(used - have)}"
