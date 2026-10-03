#!/usr/bin/env python3
"""Tests for reading verdicts and computing the evaluation metrics."""

from analysis.adjudication import classify, evaluate


def test_verdicts():
    assert classify("TP") == "TP"
    assert classify("FP - additional not relevant") == "FP_soft"
    assert classify("FP - hallucination") == "FP_hard"
    assert classify("FN - correct extraction, incorrect code") == "FN_coding"
    assert classify("FN (no diagnosis)") == "FN_miss"
    assert classify("Could not find MRN") is None


def test_metrics():
    rows = ([("TP", "diagnoses", "primary")] * 6 + [("FP_soft", "", "")] * 3
            + [("FP_hard", "", "")] + [("FN_coding", "", "")] * 2 + [("FN_miss", "", "")] * 2)
    m = evaluate(rows)
    assert m["validity_precision"] == 0.9
    assert m["alignment_precision"] == 0.6
    assert m["alignment_recall"] == 0.6
    assert m["extraction_recall"] == 0.8
    assert m["coding_accuracy"] == 0.75
    assert m["hallucination_rate"] == 0.1
