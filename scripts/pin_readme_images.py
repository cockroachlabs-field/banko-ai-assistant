#!/usr/bin/env python3
"""Pin what the README shows to a release's own tag, for the package PyPI gets.

PyPI keeps a release's description for good and loads its images from GitHub
whenever the page is viewed. The README links them on main, which is right
for the repository and wrong for a release: the 1.3.2 page lost both of its
images the day main dropped those files, and a published description cannot
be edited. The release workflow runs this before it builds, so the 2.0.0
page shows the 2.0.0 screenshots for as long as the tag exists.

    python scripts/pin_readme_images.py 2.0.0 [README.md]

Every link into this repository's files on main, the raw file or its page,
is moved to the tag v<version>. Links to the repository itself, its issues
and its workflows stay as they are. It fails when there was nothing to pin:
a README whose links took another form must not be published as if pinned.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "cockroachlabs-field/banko-ai-assistant"
ON_MAIN = re.compile(rf"(https://(?:raw\.githubusercontent\.com/{REPO}|github\.com/{REPO}/blob)/)main/")
VERSION = re.compile(r"\d+\.\d+\.\d+(?:[a-z0-9.]+)?")


def pin(text: str, version: str) -> tuple[str, int]:
    """The text with its links on main moved to the release's tag, and how many moved."""
    return ON_MAIN.subn(rf"\g<1>v{version}/", text)


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or not VERSION.fullmatch(argv[1]):
        print("usage: pin_readme_images.py <version, like 2.0.0> [README.md]", file=sys.stderr)
        return 2
    readme = Path(argv[2]) if len(argv) == 3 else Path(__file__).resolve().parents[1] / "README.md"
    pinned, moved = pin(readme.read_text(), argv[1])
    if not moved:
        print(f"nothing to pin in {readme}: no link to this repository's files on main", file=sys.stderr)
        return 1
    readme.write_text(pinned)
    print(f"pinned {moved} links in {readme.name} to v{argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
