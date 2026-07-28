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

``apply(root, mode)`` installs a scheme; ``set_mode(root, mode)`` swaps
between light and dark at runtime, which works because ``style.configure``
on an existing style name updates every widget already using it. The few
classic (non-ttk) widgets that hold a literal colour re-register through
``on_change``.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

log = logging.getLogger("mash_reid.theme")

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

# Module-level colour tokens, rebound by _install_palette() on every mode
# change. Call sites read ``theme.SURFACE`` at *construction* time, so any
# classic (non-ttk) widget holding one of these needs to repaint itself when
# the mode flips -- that is what on_change() below is for. ttk widgets need
# nothing: they carry a style name, and restyling updates them in place.
PRIMARY = LIGHT["PRIMARY"]
PRIMARY_HOVER = LIGHT["PRIMARY_HOVER"]
ON_PRIMARY = LIGHT["ON_PRIMARY"]
PRIMARY_WASH = LIGHT["PRIMARY_WASH"]
NEUTRAL = LIGHT["NEUTRAL"]
BACKGROUND = LIGHT["BACKGROUND"]
SURFACE = LIGHT["SURFACE"]
TEXT = LIGHT["TEXT"]
TEXT_SECONDARY = LIGHT["TEXT_SECONDARY"]
BORDER = LIGHT["BORDER"]
SUCCESS = LIGHT["SUCCESS"]
WARNING = LIGHT["WARNING"]
ERROR = LIGHT["ERROR"]
SUCCESS_TEXT = LIGHT["SUCCESS_TEXT"]
WARNING_TEXT = LIGHT["WARNING_TEXT"]
ERROR_TEXT = LIGHT["ERROR_TEXT"]
DANGER_WASH = LIGHT["DANGER_WASH"]
VIEWER_BACKDROP = LIGHT["VIEWER_BACKDROP"]

_MODE = DEFAULT_MODE
_LISTENERS: list = []


def current_mode() -> str:
    """``"light"`` or ``"dark"`` -- whichever palette is currently installed."""
    return _MODE


def other_mode() -> str:
    return "dark" if _MODE == "light" else "light"


def on_change(callback) -> None:
    """Register a no-argument callable to run after every mode switch.

    For classic Tk widgets only (Canvas, Label used as an image holder, and
    Treeview *tag* colours, which are per-widget rather than per-style).
    Everything built from ttk styles repaints itself.

    Listeners are expected *not* to guard their widget access: a destroyed
    widget's ``TclError`` is what tells ``_notify`` to drop the listener, and
    swallowing it would leave dead callbacks on the list for the whole
    session. A raising listener never aborts the switch either way.
    """
    _LISTENERS.append(callback)


def _notify() -> None:
    """Run every listener, dropping the ones whose widget has gone.

    Dialogs register on construction and are opened repeatedly over a
    session, so without pruning the list would grow without bound and every
    switch would walk a longer trail of dead callbacks. A listener that
    raises is treated as dead, which is why the ones in this codebase touch
    their widgets unguarded.
    """
    alive = []
    for callback in list(_LISTENERS):
        try:
            callback()
        except Exception:
            log.debug("Theme listener failed; dropping it", exc_info=True)
            continue
        alive.append(callback)
    _LISTENERS[:] = alive


def _install_palette(mode: str) -> dict:
    """Rebind the module-level colour tokens to ``mode``'s palette."""
    global _MODE, PRIMARY, PRIMARY_HOVER, ON_PRIMARY, PRIMARY_WASH, NEUTRAL
    global BACKGROUND, SURFACE, TEXT, TEXT_SECONDARY, BORDER
    global SUCCESS, WARNING, ERROR, DANGER_WASH, VIEWER_BACKDROP
    global SUCCESS_TEXT, WARNING_TEXT, ERROR_TEXT

    palette = PALETTES.get(mode) or PALETTES[DEFAULT_MODE]
    _MODE = mode if mode in PALETTES else DEFAULT_MODE
    PRIMARY = palette["PRIMARY"]
    PRIMARY_HOVER = palette["PRIMARY_HOVER"]
    ON_PRIMARY = palette["ON_PRIMARY"]
    PRIMARY_WASH = palette["PRIMARY_WASH"]
    NEUTRAL = palette["NEUTRAL"]
    BACKGROUND = palette["BACKGROUND"]
    SURFACE = palette["SURFACE"]
    TEXT = palette["TEXT"]
    TEXT_SECONDARY = palette["TEXT_SECONDARY"]
    BORDER = palette["BORDER"]
    SUCCESS = palette["SUCCESS"]
    WARNING = palette["WARNING"]
    ERROR = palette["ERROR"]
    SUCCESS_TEXT = palette["SUCCESS_TEXT"]
    WARNING_TEXT = palette["WARNING_TEXT"]
    ERROR_TEXT = palette["ERROR_TEXT"]
    DANGER_WASH = palette["DANGER_WASH"]
    VIEWER_BACKDROP = palette["VIEWER_BACKDROP"]
    return palette


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


def set_mode(root: tk.Misc, mode: str) -> None:
    """Switch schemes at runtime and repaint everything that needs it.

    Re-running ``apply`` is what does the work: ``style.configure`` on an
    existing style name updates every widget already using it, so the whole
    ttk tree changes without being rebuilt. Only the classic widgets holding a
    literal colour need help, and they get it through the ``on_change``
    listeners.
    """
    apply(root, mode)
    _notify()


def apply(root: tk.Misc, mode: str = DEFAULT_MODE) -> Fonts:
    """Install the Genesis styles on ``root`` and return the resolved fonts.

    Uses ``clam`` as the base theme: it is the only ttk theme present on all
    three platforms whose elements are drawn by ttk itself rather than by the
    native toolkit, so ``background``/``foreground``/``bordercolor`` are
    actually honoured. On ``vista``/``aqua`` most colour options are silently
    ignored -- the window would keep its default grey and none of this would
    show up, which is the trap worth naming here. It is also what makes a
    dark mode possible at all: a native-drawn theme would keep painting light
    chrome no matter what this function sets.

    Safe to call repeatedly; ``set_mode`` does exactly that.
    """
    global FONTS

    _install_palette(mode)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - clam ships with every Tk build
        log.warning("clam theme unavailable; colours will not apply")

    # Fonts do not depend on the palette, so resolving them again on a mode
    # switch would just re-scan the system font list for the same answer.
    fonts = FONTS or Fonts(root)
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
                    # clam draws every bevel from lightcolor/darkcolor, and its
                    # own defaults are pale greys chosen for a light theme --
                    # left alone they outline every bordered widget in near
                    # white against a dark surface.
                    lightcolor=BORDER, darkcolor=BORDER,
                    font=fonts.body)
    style.configure("TFrame", background=BACKGROUND)
    # Card carries the 1px border; Surface is the same fill with *no* border,
    # for the rows, header strips and fillers that live inside a card.
    #
    # These were one style, and every inner container therefore drew its own
    # rectangle. Mostly invisible in light mode (#E8E8EC on #FFFFFF), it was
    # glaring in dark, and one case was a defect in either scheme: the zero
    # height filler that stretches a collapsible header to full width rendered
    # as a horizontal rule running from the section title to the card's edge,
    # straight through the row -- the "text on a line" the section captions
    # were already fixed for.
    style.configure("Card.TFrame", background=SURFACE, relief="solid", borderwidth=1,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("Surface.TFrame", background=SURFACE, relief="flat", borderwidth=0)
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
    style.configure("Success.TLabel", background=SURFACE, foreground=SUCCESS_TEXT,
                    font=fonts.small)
    style.configure("Warning.TLabel", background=SURFACE, foreground=WARNING_TEXT,
                    font=fonts.small)
    style.configure("Error.TLabel", background=SURFACE, foreground=ERROR_TEXT,
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

    # foreground is ON_PRIMARY rather than SURFACE: in dark mode the surface
    # is nearly black, and the accent is a light indigo -- reusing SURFACE
    # there would put black text on it by accident rather than by contrast.
    style.configure("Accent.TButton", background=PRIMARY, foreground=ON_PRIMARY,
                    bordercolor=PRIMARY, font=fonts.body_bold, padding=(20, 7))
    style.map("Accent.TButton",
              background=[("pressed", PRIMARY_HOVER), ("active", PRIMARY_HOVER),
                          ("disabled", NEUTRAL)],
              foreground=[("disabled", ON_PRIMARY)],
              bordercolor=[("pressed", PRIMARY_HOVER), ("active", PRIMARY_HOVER),
                           ("disabled", NEUTRAL)])

    style.configure("Ghost.TButton", background=SURFACE, foreground=TEXT_SECONDARY,
                    bordercolor=SURFACE, relief="flat", borderwidth=0,
                    padding=(8, 4))
    style.map("Ghost.TButton",
              foreground=[("active", PRIMARY), ("disabled", NEUTRAL)],
              background=[("active", SURFACE)])

    style.configure("Danger.TButton", background=SURFACE, foreground=ERROR_TEXT,
                    bordercolor=ERROR)
    style.map("Danger.TButton",
              background=[("active", DANGER_WASH), ("pressed", DANGER_WASH)],
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
    # ttk.Labelframe is deliberately not used anywhere in this app. Its label
    # is drawn *straddling* the top border, and clam gives that label the
    # frame's own background rather than punching a gap in the border, so the
    # 1px rule runs straight through the text. A plain overline caption
    # stacked above a Panel frame says the same thing with nothing crossing
    # anything. These styles exist only so a stray Labelframe elsewhere still
    # picks up the palette.
    style.configure("TLabelframe", background=SURFACE, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=SURFACE, foreground=TEXT_SECONDARY,
                    font=fonts.overline)

    # Panel: the Labelframe replacement -- a bordered surface with no label of
    # its own, captioned by a separate Panel.TLabel above it.
    style.configure("Panel.TFrame", background=SURFACE, relief="solid", borderwidth=1,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("Panel.TLabel", background=BACKGROUND, foreground=TEXT_SECONDARY,
                    font=fonts.overline)
    # Same caption, for a panel nested inside a card (surface, not window bg).
    style.configure("PanelOnCard.TLabel", background=SURFACE, foreground=TEXT_SECONDARY,
                    font=fonts.overline)

    # Disclosure triangle on a collapsible card's header. Borderless and
    # surface-coloured so the whole header row reads as one clickable strip
    # rather than as a button sitting next to a title.
    style.configure("Chevron.TButton", background=SURFACE, foreground=TEXT_SECONDARY,
                    bordercolor=SURFACE, relief="flat", borderwidth=0,
                    font=fonts.small, padding=(2, 0), width=2)
    style.map("Chevron.TButton",
              background=[("active", SURFACE), ("pressed", SURFACE)],
              foreground=[("active", PRIMARY)])

    style.configure("TPanedwindow", background=BACKGROUND)
    # The divider has to be both visible and wide enough to grab. clam's
    # default sash is a hairline in the frame colour, which on either scheme
    # reads as a seam rather than as something draggable.
    style.configure("Sash", sashthickness=7, gripcount=0,
                    background=BORDER, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, handlesize=0)
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


class CollapsibleCard(ttk.Frame):
    """A card whose body can be folded away to a single header line.

    The window has three of these and a results area underneath; on a laptop
    screen the settings ate most of the height, leaving a strip too short to
    compare vehicles in -- which is the actual job. Folding a step away once
    it is set gives that height back without hiding anything permanently, and
    the fold state is remembered between runs.

    Build into ``.body``; the header row (step marker, title, chevron) is
    managed here. Clicking anywhere on the header toggles, not just the
    chevron -- a 12px triangle is a needlessly small target for the thing the
    user will hit most often.
    """

    OPEN, CLOSED = "\u25be", "\u25b8"   # down / right pointing triangle

    def __init__(self, parent, step: str, title: str, hint: str = "",
                 expanded: bool = True, on_toggle=None):
        super().__init__(parent, style="Card.TFrame", padding=PAD_L)
        self._expanded = bool(expanded)
        self._on_toggle = on_toggle

        header = ttk.Frame(self, style="Surface.TFrame")
        header.pack(fill="x")
        self._chevron = ttk.Button(header, style="Chevron.TButton",
                                   text=self.OPEN if self._expanded else self.CLOSED,
                                   command=self.toggle)
        self._chevron.pack(side="left", padx=(0, PAD_M))
        step_lbl = ttk.Label(header, text=step, style="Step.TLabel")
        step_lbl.pack(side="left", padx=(0, PAD_M))
        title_lbl = ttk.Label(header, text=title, style="Section.TLabel")
        title_lbl.pack(side="left")
        # A spacer that fills the rest of the row, so clicking the empty space
        # to the right of the title also toggles.
        filler = ttk.Frame(header, style="Surface.TFrame")
        filler.pack(side="left", fill="x", expand=True)
        for widget in (header, step_lbl, title_lbl, filler):
            widget.bind("<Button-1>", lambda _e: self.toggle())
            widget.configure(cursor="hand2")

        self._hint = None
        if hint:
            self._hint = ttk.Label(self, text=hint, style="Caption.TLabel",
                                   wraplength=760, justify="left")

        self.body = ttk.Frame(self, style="Surface.TFrame")
        self._render()

    def _render(self):
        # The header is packed once in __init__ and never unpacked, so a plain
        # pack() here always lands hint-then-body beneath it, in that order.
        if self._expanded:
            if self._hint is not None:
                self._hint.pack(anchor="w", pady=(2, PAD_M))
            self.body.pack(fill="x")
            self._chevron.config(text=self.OPEN)
        else:
            if self._hint is not None:
                self._hint.pack_forget()
            self.body.pack_forget()
            self._chevron.config(text=self.CLOSED)

    @property
    def expanded(self) -> bool:
        return self._expanded

    def toggle(self):
        self._expanded = not self._expanded
        self._render()
        if self._on_toggle:
            self._on_toggle(self._expanded)


def panel(parent, caption: str, on_card: bool = False, **kwargs):
    """Captioned bordered panel -- the Labelframe replacement.

    Returns ``(container, body)``: pack/grid the container, build into the
    body. The caption sits *above* the border rather than across it, which is
    the whole reason this exists (see the TLabelframe comment in ``apply``).
    """
    container = ttk.Frame(parent, style="Surface.TFrame" if on_card else "TFrame")
    ttk.Label(container, text=caption.upper(),
              style="PanelOnCard.TLabel" if on_card else "Panel.TLabel"
              ).pack(anchor="w", pady=(0, PAD_S))
    kwargs.setdefault("padding", PAD_M)
    body = ttk.Frame(container, style="Panel.TFrame", **kwargs)
    body.pack(fill="both", expand=True)
    return container, body


class VScrollFrame(ttk.Frame):
    """Vertically scrolling container, with the scrollbar hidden when unused.

    ``max_height`` caps how tall the visible area is allowed to grow: the
    control stack uses it so the settings never push the results area off the
    bottom of the window, while a dialog passes ``None`` and simply fills
    whatever space it is given. Either way the content scrolls rather than
    being clipped with no way to reach it -- which is what happened on a
    laptop screen before.

    Build into ``.body``.
    """

    WHEEL_STEP_UNITS = 3

    def __init__(self, parent, max_height: int | None = None, background=None):
        super().__init__(parent)
        self._max_height = max_height
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                 background=background or BACKGROUND)
        self._bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scroll_set)
        self._canvas.pack(side="left", fill="both", expand=True)
        # The scrollbar is packed and unpacked by _on_scroll_set rather than
        # left permanently visible: a disabled full-length trough next to
        # content that fits reads as a broken control.
        self._bar_shown = False

        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        # Keep the inner frame exactly as wide as the canvas, so its children
        # can use fill="x" -- a canvas window otherwise takes its requested
        # width and everything inside stays left-hugging at its natural size.
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfigure(self._window, width=e.width))

        self._canvas.bind("<Enter>", self._grab_wheel)
        self._canvas.bind("<Leave>", self._release_wheel)
        on_change(self._repaint)

    def _repaint(self):
        # Deliberately unguarded: a destroyed widget raises TclError here, and
        # _notify treats that as the signal to drop this listener. Swallowing
        # it would keep dead frames on the list forever.
        self._canvas.configure(background=BACKGROUND)

    def _on_body_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        # Requested height follows the content (capped, when a cap is given).
        # Without this the canvas keeps Tk's own default of 7c, which is both
        # arbitrary and the number a Panedwindow would use to place its sash.
        # Where the frame is packed with expand=True the request is ignored
        # anyway, so tracking it costs nothing there.
        wanted = self.body.winfo_reqheight()
        if self._max_height is not None:
            wanted = min(wanted, self._max_height)
        self._canvas.configure(height=max(1, wanted))

    def _on_scroll_set(self, first, last):
        self._bar.set(first, last)
        needed = not (float(first) <= 0.0 and float(last) >= 1.0)
        if needed and not self._bar_shown:
            self._bar.pack(side="right", fill="y")
            self._bar_shown = True
        elif not needed and self._bar_shown:
            self._bar.pack_forget()
            self._bar_shown = False

    def _grab_wheel(self, _event=None):
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", self._on_wheel)
        self._canvas.bind_all("<Button-5>", self._on_wheel)

    def _release_wheel(self, _event=None):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.unbind_all(sequence)

    def _on_wheel(self, event):
        if not self._bar_shown:
            return  # nothing to scroll; let the event fall through
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(delta * self.WHEEL_STEP_UNITS, "units")
