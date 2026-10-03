#!/usr/bin/env python3
"""Tests for the agreement measures."""

import numpy as np
import pandas as pd

from analysis.agreement import category_average, episode_kappa, overlap, set_scores


def _episode(dx, px):
    return {"dx": set(dx), "px": set(px), "dx_rows": [{c} for c in dx],
            "px_rows": [{c} for c in px]}


def test_identical_annotations_agree_perfectly():
    assert episode_kappa([{"H71"}, {"J35"}], [{"J35"}, {"H71"}]) == 1.0


def test_kappa_is_one_when_match_vectors_coincide():
    assert episode_kappa([{"H71"}], []) == 1.0
    assert episode_kappa([{"H71"}], [{"J35"}]) == 1.0


def test_kappa_falls_when_match_counts_differ():
    assert episode_kappa([{"H71", "J35"}], [{"H71"}, {"J35"}]) < 1.0


def test_no_annotations_is_missing():
    assert np.isnan(episode_kappa([], []))


def test_category_average_ignores_missing_types():
    units = pd.DataFrame({"dx": [1.0, 0.5], "px": [np.nan, 1.0]})
    assert category_average(units) == (0.75 + 1.0) / 2


def test_overlap():
    assert overlap({"H71", "J35"}, {"J35"}) == {"jaccard": 0.5, "anymatch": 1.0}
    assert overlap(set(), {"J35"}) is None


def test_set_scores_are_directional():
    model = {"a": _episode(["H719", "J350"], [])}
    reference = {"a": _episode(["H710"], [])}
    combined = set_scores(model, reference, ["a"], np.random.default_rng(0), n_boot=10)
    row = combined[combined.code_type == "combined"].iloc[0]
    assert row.precision == 0.5 and row.recall == 1.0
