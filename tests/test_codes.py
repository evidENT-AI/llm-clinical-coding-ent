#!/usr/bin/env python3
"""Tests for code, hospital number and date normalisation."""

import datetime as dt

from analysis.codes import (code_set, norm_id, normalise_code, normalise_diagnosis_code,
                            swapped, truncate)


def test_codes_are_normalised():
    assert normalise_code("d14.4") == "D144"
    assert normalise_diagnosis_code("H71.X") == "H71"
    assert normalise_diagnosis_code("H71.9") == "H719"


def test_code_set_splits_cells():
    assert code_set(["D16.1 Y01.7", "n/a", None], normalise_code) == {"D161", "Y017"}


def test_truncate_to_category():
    assert truncate({"H719", "J350"}, "category") == {"H71", "J35"}


def test_hospital_numbers_drop_padding():
    assert norm_id("02004297") == norm_id(2004297.0) == "2004297"


def test_ambiguous_dates_swap():
    assert swapped(dt.date(2024, 10, 4)) == dt.date(2024, 4, 10)
    assert swapped(dt.date(2024, 4, 25)) is None
