"""Colour palettes for the Genesis design system port.

Split out of ``theme`` so it can be read without importing tkinter: the icon
generator in ``tools/make_icon.py`` needs the brand indigo, and the palette
tests need the raw values, neither of which should require a display or a Tk
build to get at. ``theme`` re-exports everything here, so call sites keep
using ``theme.PRIMARY`` and friends.
"""

from __future__ import annotations

# --- Palettes (genesisDESIGN_1.md "Colors") ---------------------------------
#
# Two of them. The document says outright: "Do provide sufficient contrast in
# both light and dark modes -- test both", so a dark scheme is part of the
# system rather than a departure from it. Dark is not the light palette with
# background and text swapped -- that produces glaring text and an indigo too
# dark to read as interactive. Instead:
#
#   * text is #F5F5F7 rather than pure white, and the darkest surface is
#     #0A0A0B rather than pure black, following the document's rule against
#     using #000000/#FFFFFF for text;
#   * the accent moves up the indigo ramp (#6366F1 -> #818CF8) because the
#     original reads as near-black against a dark surface;
#   * because that lighter indigo is itself bright, text *on* it flips to the
#     dark background colour -- hence ON_PRIMARY, which is the one token the
#     document does not name but that a two-scheme port cannot do without
#     (white on #818CF8 is about 2.5:1, far under the readable floor).
#
# SECONDARY is deliberately absent from both: the document reserves #20970B
# strictly for one brand highlight on its own homepage, so there is nothing
# here it applies to, and using it anyway would break the "indigo only for
# interactive" rule it exists to protect.

LIGHT = {
    "PRIMARY": "#6366F1",
    "PRIMARY_HOVER": "#4F46E5",
    "ON_PRIMARY": "#FFFFFF",
    # Selection/pressed fill. Genesis specifies focus as a 3px translucent
    # indigo ring, which Tk cannot draw (no alpha on widget borders), so the
    # nearest opaque equivalent is used: a light indigo wash for selected rows
    # and a solid indigo border on the focused widget.
    "PRIMARY_WASH": "#EEEEFC",
    "NEUTRAL": "#9C9C9C",
    "BACKGROUND": "#FAFAFA",
    "SURFACE": "#FFFFFF",
    "TEXT": "#0A0A0A",
    "TEXT_SECONDARY": "#6B6B6B",
    "BORDER": "#E8E8EC",
    "SUCCESS": "#10B981",
    "WARNING": "#F59E0B",
    "ERROR": "#EF4444",
    # Darker siblings, for the semantic colours used as *text on a surface*.
    # The document's own values are specified for status chips -- a coloured
    # pill with dark text on it -- and only ever have to survive against that
    # fill. As body text on white they measure 2.5:1 (success) and 2.2:1
    # (warning), i.e. genuinely hard to read, which is exactly how the
    # per-point timestamp lines came out. Chips keep the original values;
    # anything drawn as coloured text uses these.
    "SUCCESS_TEXT": "#047857",
    "WARNING_TEXT": "#B45309",
    "ERROR_TEXT": "#DC2626",
    "DANGER_WASH": "#FEF2F2",
    # Backdrop behind an image in the zoom viewer. Dark in *both* schemes on
    # purpose: a photograph is judged against a neutral dark surround, and a
    # white mat around a night-time CCTV frame washes it out.
    "VIEWER_BACKDROP": "#0A0A0A",
}

DARK = {
    "PRIMARY": "#818CF8",
    "PRIMARY_HOVER": "#A5B4FC",
    "ON_PRIMARY": "#0A0A0B",
    "PRIMARY_WASH": "#2A2A45",
    "NEUTRAL": "#6E6E76",
    "BACKGROUND": "#0A0A0B",
    "SURFACE": "#151518",
    "TEXT": "#F5F5F7",
    "TEXT_SECONDARY": "#A1A1A8",
    "BORDER": "#2C2C33",
    "SUCCESS": "#34D399",
    "WARNING": "#FBBF24",
    "ERROR": "#F87171",
    # Already 6.5-11:1 against the dark surface, so the text variants are the
    # same values -- darkening them here would make them *less* readable.
    "SUCCESS_TEXT": "#34D399",
    "WARNING_TEXT": "#FBBF24",
    "ERROR_TEXT": "#F87171",
    "DANGER_WASH": "#3A1D1D",
    "VIEWER_BACKDROP": "#000000",
}

PALETTES = {"light": LIGHT, "dark": DARK}
DEFAULT_MODE = "light"
