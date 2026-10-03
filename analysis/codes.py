#!/usr/bin/env python3
"""Normalise classification codes, hospital numbers and dates for matching."""

import datetime as dt
import re

import pandas as pd

LEVEL_LENGTH = {"exact": None, "category": 3}


def normalise_code(value):
    """Return a code upper-cased without punctuation, e.g. D14.4 -> D144."""
    if pd.isna(value):
        return None
    code = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return code or None


def normalise_diagnosis_code(value):
    """As `normalise_code`, collapsing the ICD-10 trailing-X filler (H71X -> H71)."""
    code = normalise_code(value)
    if code and re.fullmatch(r"[A-Z][0-9]{2}X", code):
        return code[:3]
    return code


def split_codes(cell):
    """Split a cell that may hold several codes, e.g. 'D16.1 Y01.7'."""
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text or text.lower() == "n/a":
        return []
    return [part for part in re.split(r"\s+", text) if part and part.lower() != "n/a"]


def code_set(cells, normaliser):
    """Return the set of normalised codes found in some cells."""
    return {code for cell in cells for part in split_codes(cell)
            if (code := normaliser(part))}


def truncate(codes, level):
    """Truncate codes to the matching level ('exact' or 'category')."""
    length = LEVEL_LENGTH[level]
    return {code[:length] if length else code for code in codes}


def norm_id(value):
    """Return a hospital number as digits only, without zero padding."""
    text = re.sub(r"\.0$", "", str(value).strip())
    return re.sub(r"[^0-9]", "", text).lstrip("0")


def alnum_id(value):
    """Return an upper-case alphanumeric hospital number."""
    text = re.sub(r"\.0$", "", str(value).strip())
    return re.sub(r"[^0-9A-Z]", "", text.upper())


def parse_date(value):
    """Parse a datetime, an Excel serial number or a day-first date string.

    Datetimes are taken as they are; re-parsing them day-first would swap day
    and month for ambiguous dates.
    """
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(value).date()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")).date()
    parsed = pd.to_datetime(str(value).split(" ")[0], dayfirst=True, errors="coerce")
    return pd.NaT if pd.isna(parsed) else parsed.date()


def swapped(date):
    """Return the day/month swap of an ambiguous date, else None."""
    if not isinstance(date, dt.date) or date.day > 12 or date.day == date.month:
        return None
    return dt.date(date.year, date.day, date.month)
