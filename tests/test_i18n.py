"""Tests for the Thai/English string catalog.

These run in CI, which has no Tk build: ``app/i18n.py`` imports nothing from
tkinter precisely so the catalog stays testable. The GUI wiring around it
cannot be exercised here, so the checks that matter most are the structural
ones -- that every catalog key still corresponds to a real call site, and that
no entry can crash a render.
"""

from __future__ import annotations

import ast
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "app"))

import i18n  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_language():
    """Language is module-global; put it back so tests cannot leak into
    each other (or into whatever runs next in the same session)."""
    before = i18n.current_language()
    yield
    i18n.set_language(before)


#: Files allowed to originate a user-facing string. gui.py is nearly all of
#: it; timestamp_ocr contributes the per-point status templates, which are
#: assembled from the sidecar's shape and so cannot live in the GUI.
_STRING_SOURCES = (("app", "gui.py"), ("src", "mash_reid", "timestamp_ocr.py"))


def _ui_source() -> str:
    out = []
    for parts in _STRING_SOURCES:
        with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
            out.append(fh.read())
    return "\n".join(out)


# --- t() ----------------------------------------------------------------------

def test_english_returns_the_source_string():
    i18n.set_language("en")
    assert i18n.t("Process") == "Process"


def test_thai_returns_the_translation():
    i18n.set_language("th")
    assert i18n.t("Process") == "ประมวลผล"


def test_untranslated_string_falls_back_to_english():
    # The whole point of keying on English source strings: a label added
    # without touching the catalog renders readably instead of raising or
    # showing a placeholder.
    i18n.set_language("th")
    assert i18n.t("A string nobody has translated") == "A string nobody has translated"


def test_unknown_language_falls_back_rather_than_sticking():
    i18n.set_language("kl")
    assert i18n.current_language() == i18n.DEFAULT_LANGUAGE
    assert i18n.t("Process") == "Process"


def test_display_name_round_trips_through_the_picker():
    for code in i18n.available():
        assert i18n.code_for_display(i18n.display_name(code)) == code


def test_unknown_display_name_falls_back():
    assert i18n.code_for_display("Klingon") == i18n.DEFAULT_LANGUAGE


def test_on_change_fires_and_a_broken_listener_cannot_block_the_switch():
    seen = []
    i18n.on_change(lambda: seen.append(i18n.current_language()))
    i18n.on_change(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    i18n.on_change(lambda: seen.append("second"))
    i18n.set_language("th")
    assert seen == ["th", "second"], seen


# --- tf() ---------------------------------------------------------------------

def test_tf_fills_named_fields():
    i18n.set_language("en")
    assert i18n.tf("A: {a} vehicles", a=12) == "A: 12 vehicles"


def test_tf_falls_back_when_a_translation_drops_a_field():
    # A bad catalog entry must show the untranslated line, not raise mid-render.
    i18n.CATALOGS["th"]["{count} frames"] = "{nope} เฟรม"
    try:
        i18n.set_language("th")
        assert i18n.tf("{count} frames", count=3) == "3 frames"
    finally:
        del i18n.CATALOGS["th"]["{count} frames"]


# --- catalog integrity --------------------------------------------------------

def test_no_duplicate_keys_in_the_catalog_literal():
    """A repeated key in a dict literal is silently collapsed by Python, so
    only the source can reveal it -- and the survivor is the *last* one, which
    is rarely the one the author was looking at."""
    with open(i18n.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = {}
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value in seen:
                    duplicates.append((key.value, seen[key.value], key.lineno))
                seen[key.value] = key.lineno
    assert not duplicates, f"duplicate catalog keys: {duplicates}"


def test_every_entry_is_a_non_empty_string_pair():
    for key, value in i18n.TH.items():
        assert isinstance(key, str) and key, key
        assert isinstance(value, str) and value, key


def test_translations_keep_their_format_fields():
    """A translation that adds or drops a ``{field}`` breaks at render time,
    and ``tf`` can only fall back -- catching it here is better."""
    field = re.compile(r"\{(\w+)\}")
    for key, value in i18n.TH.items():
        assert set(field.findall(value)) <= set(field.findall(key)), key


def test_every_catalog_key_still_appears_in_the_ui():
    """Guards the cost of keying on English source strings: editing a label
    silently orphans its translation, and nothing else would say so.

    Matching is on a distinctive slice rather than the whole string because
    long strings are written as implicit concatenation across lines in both
    files, with different line breaks.
    """
    source = _ui_source()
    orphans = []
    for key in i18n.TH:
        probe = key[:28] if len(key) > 28 else key
        if probe not in source:
            orphans.append(key)
    assert not orphans, f"catalog keys with no call site: {orphans}"


def test_thai_entries_actually_contain_thai():
    """Catches a half-finished entry copied from the English side. Keys that
    are pure punctuation or symbols are legitimately identical."""
    def has_thai(s):
        return any("฀" <= ch <= "๿" for ch in s)

    untranslated = [k for k, v in i18n.TH.items()
                    if v == k and not has_thai(v) and any(ch.isalpha() for ch in k)]
    assert not untranslated, f"entries left in English: {untranslated}"
