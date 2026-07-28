"""Genesis design system, mapped onto Tkinter/ttk.

The palette, type scale and spacing grid come from ``genesisDESIGN_1.md``.
That document describes a *web* interface, so a faithful port is impossible
and pretending otherwise would produce a worse-looking window, not a better
one. What carries over and what does not:

  carried over  palette, type scale and hierarchy, the 4px spacing grid,
                1px recessive borders, the "one filled primary button per
                view" rule, indigo reserved strictly for interactive things.
  dropped       rounded corners, drop shadows, backdrop blur, hover lift.
                Tk draws none of them: ttk widget borders are square (the
                ``clam`` element geometry has no radius), and there is no
                compositor-level blur or shadow to reach for. Faking them
                with images would mean bitmap-scaled buttons that go blurry
                at any display scaling -- worse than a clean flat surface.
  substituted   General Sans / DM Sans / JetBrains Mono are web fonts loaded
                from Fontshare and Google Fonts. Tk can only use families
                installed on the machine, so ``_pick_family`` asks for the
                real ones first and falls back to the closest system face.

Sizes are converted from the document's CSS pixels to Tk points at roughly
0.75x (15px body -> 10pt), which is what makes a desktop window read at the
same visual weight as the web mock rather than a third larger.

``apply(root)`` is called once from ``main()``; everything after that is
plain ttk with the style names below.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

log = logging.getLogger("mash_reid.theme")

# --- Palette (genesisDESIGN_1.md "Colors") ----------------------------------
#
# SECONDARY is deliberately absent: the document reserves #20970B strictly for
# one brand highlight on its own homepage, so there is nothing here it applies
# to, and using it anyway would break the "indigo only for interactive" rule
# it exists to protect.

PRIMARY = "#6366F1"
PRIMARY_HOVER = "#4F46E5"
NEUTRAL = "#9C9C9C"
BACKGROUND = "#FAFAFA"
SURFACE = "#FFFFFF"
TEXT = "#0A0A0A"
TEXT_SECONDARY = "#6B6B6B"
BORDER = "#E8E8EC"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
ERROR = "#EF4444"

# Selection/pressed fills. Genesis specifies focus as a 3px translucent
# indigo ring, which Tk cannot draw (no alpha on widget borders), so the
# nearest opaque equivalent is used: a very light indigo wash for selected
# rows and a solid indigo border on the focused widget.
PRIMARY_WASH = "#EEEEFC"

# --- Spacing (4px base unit) ------------------------------------------------

SPACE = (0, 4, 8, 12, 16, 20, 24, 32, 40, 48)

PAD_S = 4
PAD_M = 8
PAD_L = 16

# --- Type scale -------------------------------------------------------------
#
# (name, css_px) from the document's "Type scale", converted to points below.
# Display/Headline (72/60px) have no place in a tool window and are omitted.

_SIZES_PX = {
    "heading": 32,
    "subhead": 24,
    "body": 15,
    "small": 13,
    "caption": 12,
    "overline": 11,
}

# Preference order per role. The genuine Genesis families come first so the
# window picks them up automatically on a machine that has them installed;
# everything after is a per-platform stand-in in descending order of fidelity.
_DISPLAY_FAMILIES = ("General Sans", "Inter Display", "Inter", "Segoe UI Semibold",
                     "SF Pro Display", "Helvetica Neue", "DejaVu Sans", "Arial")
_BODY_FAMILIES = ("DM Sans", "Inter", "Segoe UI", "SF Pro Text",
                  "Helvetica Neue", "DejaVu Sans", "Arial")
_CODE_FAMILIES = ("JetBrains Mono", "Cascadia Mono", "Consolas", "SF Mono",
                  "DejaVu Sans Mono", "Courier New")


def _px_to_pt(px: int) -> int:
    """CSS pixels -> Tk points, the conversion the module docstring explains."""
    return max(7, round(px * 0.75))


def _pick_family(preferences, installed: set[str], fallback: str) -> str:
    """First preference actually present on this machine.

    Matching is case-insensitive because ``font.families()`` reports the
    system's own capitalisation, which differs between platforms for the
    same face ("DejaVu Sans" vs "dejavu sans" on some X11 setups).
    """
    lowered = {name.lower(): name for name in installed}
    for name in preferences:
        hit = lowered.get(name.lower())
        if hit:
            return hit
    return fallback


class Fonts:
    """Resolved font tuples, one per role in the type scale.

    Held as ``(family, size)``/``(family, size, "bold")`` tuples rather than
    ``tkfont.Font`` objects so they can be passed straight to any widget's
    ``font=`` option and copied freely -- a shared Font object would make
    every widget using it change together, which is not what a scale wants.
    """

    def __init__(self, root: tk.Misc):
        try:
            installed = set(tkfont.families(root))
        except Exception:  # pragma: no cover - only on a broken Tk install
            log.debug("Could not enumerate font families", exc_info=True)
            installed = set()

        self.display_family = _pick_family(_DISPLAY_FAMILIES, installed, "TkDefaultFont")
        self.body_family = _pick_family(_BODY_FAMILIES, installed, "TkDefaultFont")
        self.code_family = _pick_family(_CODE_FAMILIES, installed, "TkFixedFont")

        # Headings use the display family at bold weight; body/UI text uses
        # the body family. Genesis: "never swap them".
        self.heading = (self.display_family, _px_to_pt(_SIZES_PX["heading"]), "bold")
        self.subhead = (self.display_family, _px_to_pt(_SIZES_PX["subhead"]), "bold")
        self.section = (self.display_family, _px_to_pt(_SIZES_PX["body"]), "bold")
        self.body = (self.body_family, _px_to_pt(_SIZES_PX["body"]))
        self.body_bold = (self.body_family, _px_to_pt(_SIZES_PX["body"]), "bold")
        self.small = (self.body_family, _px_to_pt(_SIZES_PX["small"]))
        self.caption = (self.body_family, _px_to_pt(_SIZES_PX["caption"]))
        self.overline = (self.body_family, _px_to_pt(_SIZES_PX["overline"]), "bold")
        self.code = (self.code_family, _px_to_pt(_SIZES_PX["small"]))


#: Populated by :func:`apply`. Widgets read ``theme.FONTS.small`` etc. rather
#: than hardcoding ``("TkDefaultFont", 8)`` at each call site, so the scale
#: stays in one place.
FONTS: Fonts | None = None


def apply(root: tk.Misc) -> Fonts:
    """Install the Genesis styles on ``root`` and return the resolved fonts.

    Uses ``clam`` as the base theme: it is the only ttk theme present on all
    three platforms whose elements are drawn by ttk itself rather than by the
    native toolkit, so ``background``/``foreground``/``bordercolor`` are
    actually honoured. On ``vista``/``aqua`` most colour options are silently
    ignored -- the window would keep its default grey and none of this would
    show up, which is the trap worth naming here.
    """
    global FONTS

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - clam ships with every Tk build
        log.warning("clam theme unavailable; colours will not apply")

    fonts = Fonts(root)
    FONTS = fonts

    # Default fonts, so untouched widgets (message boxes, file dialogs) still
    # follow the scale instead of standing out at Tk's own default size.
    for name, spec in (("TkDefaultFont", fonts.body), ("TkTextFont", fonts.body),
                       ("TkMenuFont", fonts.body), ("TkHeadingFont", fonts.body_bold),
                       ("TkFixedFont", fonts.code)):
        try:
            named = tkfont.nametofont(name, root)
            named.configure(family=spec[0], size=spec[1])
        except Exception:
            log.debug("Could not restyle named font %s", name, exc_info=True)

    root.option_add("*Font", fonts.body)
    # ttk styles cannot reach the classic widgets that remain: a Toplevel's
    # own background (every dialog in this app is one), and the Listbox /
    # Text / Entry classes ttk has no themed equivalent for at the call
    # sites that use them. The option database is the only way to set those
    # once instead of at each construction.
    root.option_add("*Toplevel.background", BACKGROUND)
    root.option_add("*Listbox.background", SURFACE)
    root.option_add("*Listbox.foreground", TEXT)
    root.option_add("*Listbox.selectBackground", PRIMARY_WASH)
    root.option_add("*Listbox.selectForeground", TEXT)
    root.option_add("*Listbox.highlightThickness", 1)
    root.option_add("*Listbox.highlightColor", PRIMARY)
    root.option_add("*Listbox.highlightBackground", BORDER)
    root.option_add("*Listbox.borderWidth", 0)
    try:
        root.configure(background=BACKGROUND)
    except tk.TclError:
        pass

    # --- Surfaces -----------------------------------------------------------
    style.configure(".", background=BACKGROUND, foreground=TEXT,
                    fieldbackground=SURFACE, bordercolor=BORDER,
                    font=fonts.body)
    style.configure("TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background=SURFACE, relief="solid", borderwidth=1)
    style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)

    # --- Text roles ---------------------------------------------------------
    # Every one of these sits on a card surface, so they carry the surface
    # background: a TLabel default background on a white card shows as a grey
    # rectangle around the text.
    style.configure("Section.TLabel", background=SURFACE, foreground=TEXT,
                    font=fonts.section)
    style.configure("Step.TLabel", background=SURFACE, foreground=PRIMARY,
                    font=fonts.overline)
    style.configure("Muted.TLabel", background=SURFACE, foreground=TEXT_SECONDARY,
                    font=fonts.small)
    # Live numeric readout beside a slider -- bold so the current value reads
    # as data rather than as another label.
    style.configure("Value.TLabel", background=SURFACE, foreground=TEXT,
                    font=fonts.body_bold)
    style.configure("Caption.TLabel", background=SURFACE, foreground=TEXT_SECONDARY,
                    font=fonts.caption)
    style.configure("Success.TLabel", background=SURFACE, foreground=SUCCESS,
                    font=fonts.small)
    style.configure("Warning.TLabel", background=SURFACE, foreground=WARNING,
                    font=fonts.small)
    style.configure("Error.TLabel", background=SURFACE, foreground=ERROR,
                    font=fonts.small)
    # Status-bar variants: these sit on the window background, not a card.
    style.configure("Status.TLabel", background=BACKGROUND, foreground=TEXT_SECONDARY,
                    font=fonts.small)
    style.configure("StatusCode.TLabel", background=BACKGROUND, foreground=TEXT_SECONDARY,
                    font=fonts.code)

    # --- Buttons ------------------------------------------------------------
    # Secondary is the default: transparent fill, 1px border. Genesis allows
    # only one filled indigo button per view section, which is why Accent
    # exists as a separate style rather than as the base.
    style.configure("TButton", background=SURFACE, foreground=TEXT,
                    bordercolor=BORDER, focuscolor=PRIMARY, font=fonts.small,
                    relief="solid", borderwidth=1, padding=(12, 5))
    style.map("TButton",
              background=[("pressed", PRIMARY_WASH), ("active", PRIMARY_WASH),
                          ("disabled", BACKGROUND)],
              foreground=[("disabled", NEUTRAL)],
              bordercolor=[("active", PRIMARY), ("focus", PRIMARY)])

    style.configure("Accent.TButton", background=PRIMARY, foreground=SURFACE,
                    bordercolor=PRIMARY, font=fonts.body_bold, padding=(20, 7))
    style.map("Accent.TButton",
              background=[("pressed", PRIMARY_HOVER), ("active", PRIMARY_HOVER),
                          ("disabled", NEUTRAL)],
              foreground=[("disabled", SURFACE)],
              bordercolor=[("pressed", PRIMARY_HOVER), ("active", PRIMARY_HOVER),
                           ("disabled", NEUTRAL)])

    style.configure("Ghost.TButton", background=SURFACE, foreground=TEXT_SECONDARY,
                    bordercolor=SURFACE, relief="flat", borderwidth=0,
                    padding=(8, 4))
    style.map("Ghost.TButton",
              foreground=[("active", PRIMARY), ("disabled", NEUTRAL)],
              background=[("active", SURFACE)])

    style.configure("Danger.TButton", background=SURFACE, foreground=ERROR,
                    bordercolor=ERROR)
    style.map("Danger.TButton",
              background=[("active", "#FEF2F2"), ("pressed", "#FEF2F2")],
              foreground=[("disabled", NEUTRAL)])

    # --- Inputs -------------------------------------------------------------
    for entry_style in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(entry_style, fieldbackground=SURFACE, background=SURFACE,
                        foreground=TEXT, bordercolor=BORDER, insertcolor=TEXT,
                        arrowcolor=TEXT_SECONDARY, padding=(6, 4))
        style.map(entry_style,
                  bordercolor=[("focus", PRIMARY), ("hover", NEUTRAL)],
                  lightcolor=[("focus", PRIMARY)],
                  darkcolor=[("focus", PRIMARY)],
                  fieldbackground=[("disabled", BACKGROUND), ("readonly", SURFACE)],
                  foreground=[("disabled", NEUTRAL)])

    style.configure("TCheckbutton", background=SURFACE, foreground=TEXT,
                    indicatorcolor=SURFACE, font=fonts.small, focuscolor=SURFACE)
    style.map("TCheckbutton",
              indicatorcolor=[("selected", PRIMARY), ("pressed", PRIMARY_WASH)],
              foreground=[("disabled", NEUTRAL)],
              background=[("active", SURFACE)])

    style.configure("TScale", background=SURFACE, troughcolor=BORDER,
                    bordercolor=BORDER, lightcolor=PRIMARY, darkcolor=PRIMARY)
    style.map("TScale", background=[("active", SURFACE)])

    # --- Containers ---------------------------------------------------------
    style.configure("TLabelframe", background=SURFACE, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=SURFACE, foreground=TEXT_SECONDARY,
                    font=fonts.overline)

    style.configure("TPanedwindow", background=BACKGROUND)
    style.configure("TSeparator", background=BORDER)

    style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE,
                    foreground=TEXT, bordercolor=BORDER, rowheight=24,
                    font=fonts.small)
    style.configure("Treeview.Heading", background=BACKGROUND, foreground=TEXT_SECONDARY,
                    font=fonts.overline, relief="flat", padding=(6, 4))
    style.map("Treeview",
              background=[("selected", PRIMARY_WASH)],
              foreground=[("selected", TEXT)])
    style.map("Treeview.Heading", background=[("active", BORDER)])

    style.configure("TScrollbar", background=BACKGROUND, troughcolor=BACKGROUND,
                    bordercolor=BACKGROUND, arrowcolor=TEXT_SECONDARY)
    style.map("TScrollbar", background=[("active", NEUTRAL)])

    style.configure("TProgressbar", background=PRIMARY, troughcolor=BORDER,
                    bordercolor=BORDER, lightcolor=PRIMARY, darkcolor=PRIMARY)

    return fonts


def card(parent, **kwargs) -> ttk.Frame:
    """A white 1px-bordered panel on the 4px grid -- the Genesis "Card"."""
    kwargs.setdefault("padding", PAD_L)
    return ttk.Frame(parent, style="Card.TFrame", **kwargs)


def section_header(parent, step: str, title: str, hint: str = "") -> ttk.Frame:
    """Overline step marker + section title + optional one-line hint.

    The step marker is the only thing that makes the window's top-to-bottom
    order readable as a *sequence* rather than as an undifferentiated stack
    of settings, which is the whole point of grouping them into cards.
    """
    head = ttk.Frame(parent, style="Card.TFrame")
    head.pack(fill="x", pady=(0, PAD_M))
    line = ttk.Frame(head, style="Card.TFrame")
    line.pack(fill="x")
    ttk.Label(line, text=step, style="Step.TLabel").pack(side="left", padx=(0, PAD_M))
    ttk.Label(line, text=title, style="Section.TLabel").pack(side="left")
    if hint:
        ttk.Label(head, text=hint, style="Caption.TLabel",
                  wraplength=760, justify="left").pack(anchor="w", pady=(2, 0))
    return head
