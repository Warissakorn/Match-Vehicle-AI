"""Thai / English interface strings.

Design notes worth knowing before adding to this file:

**English source strings are the keys.** ``t("Process")`` returns the Thai
when Thai is selected and falls back to the English it was given otherwise --
including when a string simply has no translation yet. That means a missing
entry degrades to a readable English label rather than to a raised
``KeyError`` or a ``???`` placeholder, and it means call sites stay readable
(``t("Fix times...")`` says what it renders, unlike ``t("btn.fix_times")``).
The cost is that editing an English string silently drops its translation,
which ``tests/test_i18n.py`` guards against by checking every key still has a
call site.

**Interpolated text uses named fields**, never positional ones -- Thai word
order differs from English, so a translator has to be free to move the values
around. ``tf("A: {a} vehicles", a=n)`` works; ``t("A: %d") % n`` would not.

**Numerals stay Arabic (0-9).** Thai numerals exist but are not what anyone
reads timestamps or counts in on a computer, and the values here are mostly
clock times and file counts.

Left untranslated on purpose: file paths, timestamps, OCR pattern tokens
(``YYYY-MO-DD``), model names, device names, and the ``A#12``/``B#7`` record
labels -- all identifiers rather than prose.
"""

from __future__ import annotations

import logging

log = logging.getLogger("mash_reid.i18n")

#: Language code -> the name shown in the picker, in that language.
LANGUAGES = {"en": "English", "th": "ไทย"}
DEFAULT_LANGUAGE = "en"

# --- Thai catalog -------------------------------------------------------------

TH: dict[str, str] = {
    # Window / chrome
    "Match-Vehicle-AI  |  Cross-Point Vehicle Re-ID":
        "Match-Vehicle-AI  |  จับคู่ยานพาหนะข้ามจุดตรวจ",
    "Language": "ภาษา",
    "◑ Dark": "◑ มืด",
    "◐ Light": "◐ สว่าง",

    # Step 1
    "STEP 1": "ขั้นที่ 1",
    "Frames": "ภาพนิ่ง",
    "Point a folder of extracted frames at each camera position, or "
    "pull them straight out of a video.":
        "เลือกโฟลเดอร์ภาพนิ่งของกล้องแต่ละจุด หรือดึงออกมาจากไฟล์วิดีโอโดยตรง",
    "Point A": "จุด A",
    "Point B": "จุด B",
    "Browse...": "เลือกโฟลเดอร์...",
    "From video...": "ดึงจากวิดีโอ...",

    # Step 2
    "STEP 2": "ขั้นที่ 2",
    "Timestamps and model": "เวลาและโมเดล",
    "Matching is gated on travel time, so the clock has to be right "
    "before Process. Each point's current state is shown below.":
        "การจับคู่ใช้ช่วงเวลาเดินทางเป็นตัวกรอง เวลาจึงต้องถูกต้องก่อนกด "
        "ประมวลผล สถานะเวลาของแต่ละจุดแสดงอยู่ด้านล่าง",
    "Fix times...": "แก้ไขเวลา...",
    "Model": "โมเดล",
    "Device": "อุปกรณ์ประมวลผล",
    "Manage models...": "จัดการโมเดล...",
    "✓ downloaded": "✓ ดาวน์โหลดแล้ว",
    "not downloaded": "ยังไม่ได้ดาวน์โหลด",
    "probing...": "กำลังตรวจสอบ...",
    # Background first-run model prefetch (gui._start_prefetch)
    "detection model": "โมเดลตรวจจับ",
    "embedder": "โมเดลฝังเวกเตอร์",
    "Preparing {name} ({step} of {total})...":
        "กำลังเตรียม{name} ({step} จาก {total})...",
    "model download paused - will retry when you press Process":
        "หยุดดาวน์โหลดโมเดลชั่วคราว - จะลองอีกครั้งเมื่อกดประมวลผล",

    # Per-point timestamp state (timestamp_ocr.describe_sidecar)
    "pick a folder first": "เลือกโฟลเดอร์ก่อน",
    "could not read this folder's timestamps": "อ่านเวลาของโฟลเดอร์นี้ไม่ได้",
    "no timestamp review yet - times come from filenames":
        "ยังไม่ได้ตรวจสอบเวลา - ใช้เวลาจากชื่อไฟล์",
    "{frames} frame time(s) from an every-frame OCR pass":
        "เวลาของภาพ {frames} ภาพ จากการ OCR ทุกภาพ",
    "{frames} frame time(s) confirmed from a fitted clock, {fps} fps":
        "เวลาของภาพ {frames} ภาพ ยืนยันแล้วจากนาฬิกาที่คำนวณได้ {fps} fps",
    "{frames} frame time(s) confirmed from a fitted clock":
        "เวลาของภาพ {frames} ภาพ ยืนยันแล้วจากนาฬิกาที่คำนวณได้",
    "{frames} frame time(s) fitted but not confirmed, {fps} fps":
        "เวลาของภาพ {frames} ภาพ คำนวณแล้วแต่ยังไม่ได้ยืนยัน {fps} fps",
    "{frames} frame time(s) fitted but not confirmed":
        "เวลาของภาพ {frames} ภาพ คำนวณแล้วแต่ยังไม่ได้ยืนยัน",

    # Step 3
    "STEP 3": "ขั้นที่ 3",
    "Match": "จับคู่",
    "These re-filter an existing run instantly; only \"Detect conf\" "
    "needs another Process to take effect.":
        "ค่าเหล่านี้กรองผลลัพธ์เดิมได้ทันที มีเพียง \"ความเชื่อมั่นการตรวจจับ\" "
        "ที่ต้องกดประมวลผลใหม่",
    "Similarity": "ความคล้าย",
    "Detect conf": "ความเชื่อมั่นการตรวจจับ",
    "Min travel (s)": "เวลาเดินทางต่ำสุด (วิ)",
    "Max travel (s)": "เวลาเดินทางสูงสุด (วิ)",
    "Use time gate": "ใช้ตัวกรองเวลา",
    "One-to-one": "จับคู่หนึ่งต่อหนึ่ง",
    "Group repeat sightings (slow)": "รวมคันที่เห็นซ้ำ (ช้า)",
    "Process": "ประมวลผล",

    # Galleries
    "Point A - click a vehicle": "จุด A - คลิกเลือกยานพาหนะ",
    "Show all B": "แสดงจุด B ทั้งหมด",
    "No results yet - run Process": "ยังไม่มีผลลัพธ์ - กดประมวลผล",
    "Showing: all point B vehicles": "กำลังแสดง: ยานพาหนะจุด B ทั้งหมด",
    "✓ Same": "✓ คันเดียวกัน",
    "✗ Diff": "✗ คนละคัน",
    "saved: same vehicle": "บันทึกแล้ว: คันเดียวกัน",
    "saved: different vehicle": "บันทึกแล้ว: คนละคัน",

    # Status bar / progress
    "Select folders for point A and B, then Process.":
        "เลือกโฟลเดอร์ของจุด A และ B แล้วกดประมวลผล",
    "Loading models and processing frames... (first run downloads weights)":
        "กำลังโหลดโมเดลและประมวลผลภาพ... (รอบแรกจะดาวน์โหลดไฟล์โมเดล)",
    "Processing point A...": "กำลังประมวลผลจุด A...",
    "Processing point B...": "กำลังประมวลผลจุด B...",
    "Grouping repeat sightings...": "กำลังรวมคันที่เห็นซ้ำ...",
    "Error.": "เกิดข้อผิดพลาด",

    # Video extraction dialog
    "Video(s)": "ไฟล์วิดีโอ",
    "Output": "โฟลเดอร์ผลลัพธ์",
    "Start time": "เวลาเริ่ม",
    "Interval (s)": "ระยะห่าง (วิ)",
    "Extract": "สกัดภาพ",
    "Clear": "ล้าง",
    "Close": "ปิด",
    "Cancel": "ยกเลิก",
    "(auto from filename; edit as YYYY-MM-DD HH:MM:SS)":
        "(อ่านจากชื่อไฟล์อัตโนมัติ แก้ไขในรูปแบบ YYYY-MM-DD HH:MM:SS)",
    "Choose one or more videos to begin.": "เลือกไฟล์วิดีโอหนึ่งไฟล์ขึ้นไปเพื่อเริ่ม",
    "No video": "ไม่มีไฟล์วิดีโอ",
    "Please choose at least one video file.": "กรุณาเลือกไฟล์วิดีโออย่างน้อยหนึ่งไฟล์",
    "One or more selected videos no longer exist.":
        "ไฟล์วิดีโอที่เลือกไว้บางไฟล์ไม่มีอยู่แล้ว",
    "No output": "ไม่ได้ระบุโฟลเดอร์ผลลัพธ์",
    "Please choose an output folder.": "กรุณาเลือกโฟลเดอร์ผลลัพธ์",
    "Invalid start time": "เวลาเริ่มไม่ถูกต้อง",
    "Extracting frames...": "กำลังสกัดภาพ...",
    "Extraction complete": "สกัดภาพเสร็จแล้ว",
    "Extraction failed": "สกัดภาพไม่สำเร็จ",

    # Model manager
    "Manage detection models": "จัดการโมเดลตรวจจับ",
    "Family": "ตระกูล",
    "Size": "ขนาด",
    "Status": "สถานะ",
    "Download": "ดาวน์โหลด",
    "Update (latest)": "อัปเดต (ล่าสุด)",
    "Remove": "ลบ",
    "Model download failed": "ดาวน์โหลดโมเดลไม่สำเร็จ",

    # Timestamp review
    "Clips": "คลิป",
    "Clip": "คลิป",
    "Read": "อ่านได้",
    "Typical error": "ค่าคลาดเคลื่อนปกติ",
    "Worst": "แย่สุด",
    "Manual offset": "ปรับเองเพิ่ม",
    "Sampled frames": "ภาพตัวอย่างที่สุ่มมา",
    "Selected frame": "ภาพที่เลือก",
    "Filename clock": "เวลาจากชื่อไฟล์",
    "OCR text": "ข้อความที่ OCR อ่านได้",
    "Conf": "ความเชื่อมั่น",
    "Read clock": "เวลาที่อ่านได้",
    "Off by": "คลาดเคลื่อน",
    "Used": "ใช้",
    "Time pattern": "รูปแบบเวลา",
    "Time pattern:": "รูปแบบเวลา:",
    "Re-parse": "อ่านใหม่",
    "Save as...": "บันทึกเป็น...",
    "Manage...": "จัดการ...",
    "Shift selected clip": "เลื่อนคลิปที่เลือก",
    "Shift all clips": "เลื่อนทุกคลิป",
    "seconds": "วินาที",
    "Zoom in...": "ขยาย...",
    "Readings the OCR considered:": "ค่าที่ OCR พิจารณา:",
    "Or type the correct time (YYYY-MM-DD HH:MM:SS):":
        "หรือพิมพ์เวลาที่ถูกต้อง (YYYY-MM-DD HH:MM:SS):",
    "Use this": "ใช้ค่านี้",
    "Ignore frame": "ข้ามภาพนี้",
    "Reset": "คืนค่าเดิม",
    "Apply and write times": "ยืนยันและบันทึกเวลา",
    "Read every frame instead (slow)": "อ่านทุกภาพแทน (ช้า)",
    "Mark clock position...": "กำหนดตำแหน่งนาฬิกา...",
    "Re-sample clock (fresh OCR)...": "สุ่มอ่านนาฬิกาใหม่...",
    "Mark the clock's position": "กำหนดตำแหน่งนาฬิกา",
    "Click and drag a box around the on-screen clock, then click \"Use this box\".":
        "คลิกลากกรอบล้อมรอบนาฬิกาบนภาพ แล้วกด \"ใช้กรอบนี้\"",
    "Use this box": "ใช้กรอบนี้",
    "Scroll to zoom - drag the scrollbars to pan.":
        "เลื่อนล้อเมาส์เพื่อซูม ลากแถบเลื่อนเพื่อย้ายมุมมอง",
    "Manage saved time patterns": "จัดการรูปแบบเวลาที่บันทึกไว้",
    "Name": "ชื่อ",
    "Pattern": "รูปแบบ",
    "Delete": "ลบ",
    "Delete pattern": "ลบรูปแบบ",
    "Save time pattern": "บันทึกรูปแบบเวลา",
    "Name for this pattern:": "ตั้งชื่อให้รูปแบบนี้:",
    "Nothing to save": "ไม่มีอะไรให้บันทึก",
    "Type or pick a pattern first.": "พิมพ์หรือเลือกรูปแบบก่อน",
    "unread": "อ่านไม่ได้",
    "borrowed": "ยืมค่ามา",
    "yes": "ใช่",
    "no": "ไม่",

    # Errors / confirmations
    "Invalid folders": "โฟลเดอร์ไม่ถูกต้อง",
    "Please choose valid A and B folders.": "กรุณาเลือกโฟลเดอร์ A และ B ที่ถูกต้อง",
    "Processing failed": "ประมวลผลไม่สำเร็จ",
    "No folder": "ไม่มีโฟลเดอร์",
    "Bad time pattern": "รูปแบบเวลาไม่ถูกต้อง",
    "Could not read timestamps": "อ่านเวลาไม่สำเร็จ",
    "Could not re-read timestamps": "อ่านเวลาใหม่ไม่สำเร็จ",
    "Reading the clock failed.": "อ่านนาฬิกาไม่สำเร็จ",
    "Could not write times": "บันทึกเวลาไม่สำเร็จ",
    "Times written": "บันทึกเวลาแล้ว",
    "Could not save training pair": "บันทึกคู่ข้อมูลฝึกไม่สำเร็จ",
    "Bad time": "เวลาไม่ถูกต้อง",
    "Unreadable": "อ่านไม่ได้",
    "That reading did not parse as a time; type the correct one instead.":
        "ค่านี้แปลงเป็นเวลาไม่ได้ กรุณาพิมพ์เวลาที่ถูกต้องแทน",
    "No frame selected": "ยังไม่ได้เลือกภาพ",
    "Pick a row in the sampled frames list first.":
        "เลือกแถวในรายการภาพตัวอย่างก่อน",
    "No clip selected": "ยังไม่ได้เลือกคลิป",
    "Pick a clip row first, or use 'Shift all clips'.":
        "เลือกแถวคลิปก่อน หรือใช้ \"เลื่อนทุกคลิป\"",
    "No frame available": "ไม่มีภาพให้เลือก",
    "There are no sampled frames to pick from.": "ไม่มีภาพตัวอย่างให้เลือก",
    "Frame not found": "ไม่พบไฟล์ภาพ",
    "Fit looks wrong": "ผลการคำนวณดูผิดปกติ",
    "Read every frame?": "อ่านทุกภาพเลยหรือไม่?",
    "OCR complete": "OCR เสร็จแล้ว",
    "OCR failed": "OCR ไม่สำเร็จ",
    "?": "?",
}

CATALOGS = {"en": {}, "th": TH}

_LANGUAGE = DEFAULT_LANGUAGE
_LISTENERS: list = []


def current_language() -> str:
    return _LANGUAGE


def available() -> list[str]:
    return list(LANGUAGES)


def display_name(code: str) -> str:
    return LANGUAGES.get(code, code)


def code_for_display(name: str) -> str:
    """Reverse of :func:`display_name`, for reading a picker's selection."""
    for code, label in LANGUAGES.items():
        if label == name:
            return code
    return DEFAULT_LANGUAGE


def set_language(code: str) -> None:
    """Install ``code`` and notify listeners. Unknown codes fall back."""
    global _LANGUAGE
    _LANGUAGE = code if code in CATALOGS else DEFAULT_LANGUAGE
    for callback in list(_LISTENERS):
        try:
            callback()
        except Exception:
            log.debug("Language listener failed", exc_info=True)


def on_change(callback) -> None:
    _LISTENERS.append(callback)


def t(text: str) -> str:
    """Translate ``text``, falling back to ``text`` itself.

    The fallback is the whole point: an untranslated string renders as
    readable English rather than as an error or a placeholder, so adding a
    label without touching this file never breaks the Thai UI -- it just
    leaves one line in English until someone fills it in.
    """
    if _LANGUAGE == "en":
        return text
    return CATALOGS.get(_LANGUAGE, {}).get(text, text)


def tf(template: str, **values) -> str:
    """Translate a ``str.format`` template, then fill it in.

    Named fields only -- Thai reorders clauses relative to English, so a
    translation has to be able to move ``{a}`` and ``{b}`` past each other.
    A translation missing a field the caller passed would raise ``KeyError``
    mid-render, so that is caught and the English template used instead: a
    wrong catalog entry should show the untranslated line, not crash the app.
    """
    translated = t(template)
    try:
        return translated.format(**values)
    except (KeyError, IndexError):
        log.warning("Bad translation for %r; falling back to English", template)
        return template.format(**values)
