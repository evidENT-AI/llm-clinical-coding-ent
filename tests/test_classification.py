#!/usr/bin/env python3
"""Tests for choosing ICD-10 and OPCS-4 targets from the complex maps."""

from pipeline.classification import dotted, select_target


def _row(target, priority=1, rule="", advice="", group=1):
    return {"group": group, "priority": priority, "rule": rule, "advice": advice,
            "target": target}


def test_dotted():
    assert dotted("J039") == "J03.9"
    assert dotted("H66") == "H66"


def test_prefers_standalone_unconditional_lowest_priority():
    candidates = [
        _row("H678", priority=1, advice="ADDITIONAL CODE MANDATORY"),
        _row("H669", priority=2),
        _row("H660", priority=1, rule="IFA 248153007"),
        _row("Z000", priority=0, group=2),
    ]
    assert select_target(candidates)["target"] == "H669"


def test_no_group_one_target():
    assert select_target([_row("Z000", group=2)]) is None
