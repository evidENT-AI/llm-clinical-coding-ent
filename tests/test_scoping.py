#!/usr/bin/env python3
"""Tests for matching the model's Stage 2 selection to Stage 1 rows."""

import pandas as pd

from pipeline.labeller import _pick, format_episode

STAGE1 = pd.DataFrame([
    {"text": "chronic rhinosinusitis", "qualifier": "diagnoses", "icd10": "J32.9", "opcs4": None},
    {"text": "nasal polyps", "qualifier": "diagnoses", "icd10": "J33.9", "opcs4": None},
    {"text": "fess", "qualifier": "procedures", "icd10": None, "opcs4": "E14.1"},
])


def test_selection_by_id_code_and_text():
    selection = {"diagnoses": {"principal": 0, "secondary": ["J33.9"]},
                 "procedures": {"principal": "FESS", "secondary": []}}
    dx = _pick(STAGE1, selection, "diagnoses")
    px = _pick(STAGE1, selection, "procedures")
    assert [r["text"] for r in dx] == ["chronic rhinosinusitis", "nasal polyps"]
    assert [r["primary_secondary"] for r in dx] == ["primary", "secondary"]
    assert [r["text"] for r in px] == ["fess"]


def test_ids_from_the_wrong_category_are_ignored():
    assert _pick(STAGE1, {"procedures": {"principal": 0, "secondary": []}}, "procedures") == []


def test_episode_text_hides_the_anchor():
    text = format_episode({"notes": [{"note_date": "2024-01-15", "note_class": "Note",
                                      "is_anchor": True, "text": "FESS performed."}]})
    assert "anchor" not in text.lower()
    assert "--- Note 1 | 2024-01-15 | Note ---\nFESS performed." in text
