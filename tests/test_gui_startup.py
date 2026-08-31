"""Build the whole GUI against a fake Tkinter, to catch startup crashes.

Why this exists: the app is developed and CI-tested where tkinter is not
installed, so nothing exercised ``main()`` at all -- a NameError or a bad
attribute in window construction shipped happily and showed up on the user's
machine as "the console window opens and closes again", with the traceback
scrolling past into a closed terminal.

The stub is deliberately permissive. It is not trying to emulate Tk; it only
has to let every call in the construction path *run*, so that Python's own
name resolution and argument checking do the work. That catches the whole
class of "referenced something that does not exist yet" and "called it with
the wrong arguments" bugs, which is what actually breaks startup. It cannot
catch anything about layout, geometry or rendering.

Variables are the exception and are modelled properly (get/set/trace_add),
because ``ReIDApp`` reads its own settings back out of them during
construction; a stub that returned a mock from ``get()`` would send garbage
into ``int()``/``os.path.isdir()`` and fail for the wrong reason.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- the fake toolkit ---------------------------------------------------------

class _TclError(Exception):
    pass


class _Widget:
    """Accepts any constructor, any method, any attribute."""

    def __init__(self, *args, **kwargs):
        self._children: list = []
        self._kwargs = kwargs
        parent = args[0] if args else None
        if isinstance(parent, _Widget):
            parent._children.append(self)

    # Layout / configuration / bindings: all no-ops that return something
    # harmless. Returning self keeps chained calls working.
    def __getattr__(self, name):
        def _anything(*args, **kwargs):
            return self
        return _anything

    # Explicit where a real return value is consumed by the code under test.
    def winfo_children(self):
        return list(self._children)

    def winfo_toplevel(self):
        return self

    def winfo_reqheight(self):
        return 100

    def winfo_height(self):
        return 800

    def winfo_exists(self):
        return True

    def cget(self, key):
        return self._kwargs.get(key, "")

    def sashpos(self, index, value=None):
        return 300

    def get(self):
        return ""

    def current(self, index=None):
        return -1

    def after(self, ms, func=None, *args):
        # Not run: a real after() defers, and running the callback inline
        # would recurse forever through the resource poll.
        return "after#1"

    def after_idle(self, func=None, *args):
        return "after#idle"

    def after_cancel(self, job):
        return None

    def bbox(self, *args):
        return (0, 0, 100, 100)

    # Treeview accessors the code subscripts or iterates. The catch-all
    # __getattr__ would hand back a widget here, and "sel[0]" on a widget
    # fails for a reason that has nothing to do with the code under test.
    def selection(self, *args):
        return ()

    def get_children(self, *args):
        return ()

    def keys(self):
        return []


class _Variable:
    def __init__(self, master=None, value=None, name=None):
        self._value = value if value is not None else self._default
        self._traces: list = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for callback in list(self._traces):
            callback("", "", "write")

    def trace_add(self, mode, callback):
        self._traces.append(callback)
        return "trace#1"


class _StringVar(_Variable):
    _default = ""


class _IntVar(_Variable):
    _default = 0


class _DoubleVar(_Variable):
    _default = 0.0


class _BooleanVar(_Variable):
    _default = False


class _Font:
    def __init__(self, *args, **kwargs):
        pass

    def configure(self, **kwargs):
        pass


def _install_fake_tkinter(monkeypatch, families=()):
    """Put a fake ``tkinter`` (and submodules) into ``sys.modules``."""
    tk = types.ModuleType("tkinter")
    tk.TclError = _TclError
    for name in ("Tk", "Toplevel", "Frame", "Label", "Button", "Canvas",
                 "Listbox", "Text", "Entry", "PhotoImage", "Menu", "Misc"):
        setattr(tk, name, _Widget)
    tk.StringVar, tk.IntVar = _StringVar, _IntVar
    tk.DoubleVar, tk.BooleanVar = _DoubleVar, _BooleanVar
    tk.Variable = _Variable

    ttk = types.ModuleType("tkinter.ttk")
    for name in ("Frame", "Label", "Button", "Entry", "Combobox", "Checkbutton",
                 "Scale", "Scrollbar", "Panedwindow", "PanedWindow", "Treeview",
                 "Labelframe", "LabelFrame", "Spinbox", "Separator", "Progressbar",
                 "Notebook", "Style"):
        setattr(ttk, name, _Widget)
    tk.ttk = ttk

    font = types.ModuleType("tkinter.font")
    font.Font = _Font
    font.families = lambda root=None: families
    font.nametofont = lambda name, root=None: _Font()
    tk.font = font

    for sub, names in (("filedialog", ("askdirectory", "askopenfilenames", "askopenfilename")),
                       ("messagebox", ("showerror", "showinfo", "showwarning", "askyesno")),
                       ("simpledialog", ("askstring",))):
        module = types.ModuleType(f"tkinter.{sub}")
        for name in names:
            setattr(module, name, lambda *a, **k: None)
        setattr(tk, sub, module)
        monkeypatch.setitem(sys.modules, f"tkinter.{sub}", module)

    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", ttk)
    monkeypatch.setitem(sys.modules, "tkinter.font", font)
    return tk


class _NotStartedThread:
    """A Thread whose ``start`` does nothing.

    Window construction kicks off a background CUDA probe, which imports
    torch -- seconds of work per test, for a result nothing here can observe
    (the fake ``after`` never runs the callback that would consume it). The
    thread object is still built and its target closure still defined, so the
    construction path is unchanged; only the work is skipped.
    """

    def __init__(self, *args, **kwargs):
        self.daemon = kwargs.get("daemon", False)

    def start(self):
        return None

    def join(self, timeout=None):
        return None


@pytest.fixture
def gui(monkeypatch):
    """Import ``app/gui.py`` fresh against the fake toolkit."""
    _install_fake_tkinter(monkeypatch)
    for name in ("gui", "theme", "i18n", "palette"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.path.insert(0, os.path.join(_ROOT, "app"))
    try:
        import gui as gui_module
        # Settings must not leak in from a real settings.json on the machine
        # running the tests, or the result depends on the developer's own
        # remembered folders.
        monkeypatch.setattr(gui_module.settings, "load", lambda *a, **k: {})
        monkeypatch.setattr(gui_module.settings, "save", lambda *a, **k: None)
        monkeypatch.setattr(gui_module.threading, "Thread", _NotStartedThread)
        yield gui_module
    finally:
        sys.path.remove(os.path.join(_ROOT, "app"))


# --- the tests ----------------------------------------------------------------

def test_gui_module_imports(gui):
    assert hasattr(gui, "ReIDApp")
    assert hasattr(gui, "main")


def test_building_the_main_window_does_not_raise(gui):
    """The regression this file was added for: every widget in the window is
    constructed here, so a missing name or a bad call fails the test instead
    of the user's launcher."""
    gui.theme.apply(_Widget(), "light", "en")
    gui.ReIDApp(_Widget())


def test_building_the_main_window_in_dark_thai_does_not_raise(gui):
    gui.i18n.set_language("th")
    try:
        gui.theme.apply(_Widget(), "dark", "th")
        gui.ReIDApp(_Widget())
    finally:
        gui.i18n.set_language("en")


def test_main_runs_through_to_mainloop(gui, monkeypatch):
    """Covers the part ReIDApp's constructor does not: theme/language
    installation order, the window icon, and the close handler."""
    monkeypatch.setattr(gui.logging_setup, "setup_logging", lambda *a, **k: "log.txt")
    gui.main()


def test_switching_language_rebuilds_without_raising(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    app._set_language("th")
    assert app.ui_language.get() == "th"
    app._set_language("en")
    assert app.ui_language.get() == "en"


def test_toggling_the_theme_does_not_raise(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    app._toggle_theme()
    app._toggle_theme()


def test_collapsing_a_step_does_not_raise(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    for key in ("1", "2", "3"):
        app._on_step_toggled(key, False)
        assert app.step_open[key].get() is False


def test_thai_font_scale_is_applied_when_sarabun_is_present(monkeypatch):
    """The size correction only helps if it actually reaches the fonts."""
    _install_fake_tkinter(monkeypatch, families=("TH SarabunPSK", "Segoe UI", "Consolas"))
    for name in ("theme", "palette"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.path.insert(0, os.path.join(_ROOT, "app"))
    try:
        import theme
        english = theme.Fonts(_Widget(), "en")
        thai = theme.Fonts(_Widget(), "th")
        assert thai.body_family == "TH SarabunPSK"
        assert thai.scale == 1.45
        # Same visual size, not the same point size -- that is the whole point.
        assert thai.body[1] > english.body[1]
        assert thai.body[1] == round(english.body[1] * 1.45)
    finally:
        sys.path.remove(os.path.join(_ROOT, "app"))


def test_font_falls_back_without_a_thai_face_and_does_not_rescale(monkeypatch):
    """Tahoma is normally proportioned; applying Sarabun's correction to it
    would make the Thai UI larger than the English one, not equal to it."""
    _install_fake_tkinter(monkeypatch, families=("Tahoma", "Segoe UI"))
    for name in ("theme", "palette"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.path.insert(0, os.path.join(_ROOT, "app"))
    try:
        import theme
        thai = theme.Fonts(_Widget(), "th")
        assert thai.body_family == "Tahoma"
        assert thai.scale == 1.0
    finally:
        sys.path.remove(os.path.join(_ROOT, "app"))


def test_the_video_extract_dialog_builds(gui):
    """One click from the main window, so it deserves the same guard."""
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    gui.VideoExtractDialog(app, "A", app.dir_a)


def test_the_about_dialog_builds(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    gui.AboutDialog(app)


def test_the_model_manager_dialog_builds(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    gui.ModelManagerDialog(app)


def test_the_pattern_manager_dialog_builds(gui):
    gui.theme.apply(_Widget(), "light", "en")
    app = gui.ReIDApp(_Widget())
    gui.PatternManagerDialog(app)


def test_every_ui_callback_is_a_real_method(gui):
    """Cheap net for the same failure mode in the paths not built above.

    ``command=self._foo`` with a typo is a NameError at *click* time, long
    after startup, so nothing else here would reach it.
    """
    import ast
    with open(os.path.join(_ROOT, "app", "gui.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    classes = {n.name: {m.name for m in n.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
               for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    # Inherited from tk.Toplevel / ttk.Frame rather than defined here.
    inherited = {"destroy", "quit", "focus_set", "lift", "withdraw", "deiconify",
                 "update", "update_idletasks"}
    missing = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "command":
                    continue
                v = kw.value
                if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                        and v.value.id == "self"
                        and v.attr not in classes[cls.name]
                        and v.attr not in inherited):
                    missing.append(f"{cls.name}.{v.attr} (line {v.lineno})")
    assert not missing, f"command= points at no such method: {missing}"
