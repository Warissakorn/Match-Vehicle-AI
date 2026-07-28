"""Generate the application icon.

Kept as a script rather than a checked-in binary alone so the mark can be
re-cut when the palette moves -- the colours come straight from
``app/theme.py``, so the icon cannot drift away from the rest of the UI.

The mark
--------
Two camera points joined by one path -- the whole product in one shape. A
hollow ring low-left (seen at A) and a filled dot high-right (matched at B),
connected by a bar.

Deliberately *not* a car silhouette: below about 32px a vehicle becomes an
unreadable smudge, whereas two nodes and a bar stay legible at 16px, which is
the size the taskbar and title bar actually use.

Three things here are the result of rendering the alternatives and looking at
them at 16px rather than reasoning about them:

* **No arrowhead.** An arrow after the right-hand node merged with it into a
  single teardrop blob at small sizes; the diagonal already carries direction.
* **Diagonal, not horizontal.** A horizontal pair leaves the tile as a thin
  band with most of its area empty, and the connecting bar is the first thing
  to vanish. The diagonal uses the full tile and keeps a distinct silhouette.
* **The hollow node is punched, not stroked onto the bar.** The bar is drawn
  first, then the node's interior is refilled with the tile colour before its
  ring goes on, so the bar cannot show through the hole.

Rejected: forward chevrons (legible, but reads as a generic "next" button and
says nothing about the product) and two stacked lanes (reads as a settings
slider, which is actively misleading).

Everything is drawn at 8x and downsampled with LANCZOS. PIL has no
antialiased vector drawing, so a 16px circle drawn directly is a jagged
lump; supersampling is what makes the small sizes clean.

Run:  python tools/make_icon.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "app"))

import palette  # noqa: E402

# Windows .ico carries several bitmaps and the shell picks the closest; these
# are the sizes it actually asks for (16 title bar / taskbar, 32 alt-tab,
# 48 explorer, 256 the large-icons view).
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
CANVAS = 256
SUPERSAMPLE = 8

# The tile keeps its indigo in either scheme: an app icon sits on the user's
# desktop, not inside our window, so it cannot follow our light/dark toggle.
TILE = palette.LIGHT["PRIMARY"]
MARK = "#FFFFFF"


def _rounded_tile(draw, size, radius, fill):
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=fill)


def render(size: int = CANVAS):
    """Draw the mark at ``size`` px, supersampled and downscaled."""
    from PIL import Image, ImageDraw

    s = size * SUPERSAMPLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Tile. 22% corner radius reads as "app icon" on every modern desktop
    # without going fully circular.
    _rounded_tile(draw, s, int(s * 0.22), TILE)

    stroke = max(1, int(s * 0.075))
    r = int(s * 0.145)
    ax, ay = int(s * 0.31), int(s * 0.66)   # point A, low-left
    bx, by = int(s * 0.69), int(s * 0.34)   # point B, high-right

    # Bar first, so the nodes drawn over it mask its ends -- simpler and more
    # exact than working out where the line should stop.
    draw.line((ax, ay, bx, by), fill=MARK, width=stroke)

    # Point A: hollow ring -- seen, not yet matched. Its interior is refilled
    # with the tile colour before the ring is stroked, otherwise the bar runs
    # straight through the hole and the node reads as filled at small sizes.
    draw.ellipse((ax - r, ay - r, ax + r, ay + r), fill=TILE)
    draw.ellipse((ax - r, ay - r, ax + r, ay + r), outline=MARK, width=stroke)

    # Point B: filled -- the same vehicle, identified.
    draw.ellipse((bx - r, by - r, bx + r, by + r), fill=MARK)

    return img.resize((size, size), Image.Resampling.LANCZOS)


def main():
    from PIL import Image

    out_dir = os.path.join(_ROOT, "assets")
    os.makedirs(out_dir, exist_ok=True)

    master = render(CANVAS)
    png_path = os.path.join(out_dir, "icon.png")
    master.save(png_path)

    # Each size is rendered from scratch rather than resized from the master:
    # a 16px icon downscaled from 256 loses the ring's hole entirely, while
    # drawing it at 16x8=128 and reducing keeps the stroke proportional.
    frames = [render(n) for n in ICO_SIZES]
    ico_path = os.path.join(out_dir, "icon.ico")
    frames[-1].save(ico_path, format="ICO",
                    sizes=[(n, n) for n in ICO_SIZES], append_images=frames[:-1])

    # 64px PNG for Tk's iconphoto on Linux/macOS, where .ico is not read.
    small_path = os.path.join(out_dir, "icon-64.png")
    render(64).save(small_path)

    for path in (png_path, ico_path, small_path):
        print(f"wrote {os.path.relpath(path, _ROOT)}  ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
