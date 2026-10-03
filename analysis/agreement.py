#!/usr/bin/env python3
"""Agreement measures between two raters' code sets."""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis.codes import truncate

SEED = 42
N_BOOT = 2000
CODE_TYPES = ("diagnosis", "procedure", "combined")


def code_sets(episode, code_type, level="category"):
    """Return an episode's codes of one type at a matching level."""
    if code_type == "combined":
        codes = episode["dx"] | episode["px"]
    else:
        codes = episode["dx" if code_type == "diagnosis" else "px"]
    return truncate(codes, level)


def annotation_sets(episode, code_type, level="category"):
    """Return one code set per annotation for an episode."""
    if code_type == "combined":
        rows = episode["dx_rows"] + episode["px_rows"]
    else:
        rows = episode["dx_rows" if code_type == "diagnosis" else "px_rows"]
    return [truncate(s, level) for s in rows]


def overlap(a, b):
    """Return Jaccard and any-match for two code sets, or None if either is empty."""
    if not a or not b:
        return None
    shared = a & b
    return {"jaccard": len(shared) / len(a | b), "anymatch": float(bool(shared))}


def episode_kappa(rows_a, rows_b):
    """
    Args:
        rows_a: One code set per annotation for rater A.
        rows_b: One code set per annotation for rater B.

    Returns:
        Cohen's kappa, or NaN if neither rater has annotations.
    """
    flat_a = set().union(*rows_a) if rows_a else set()
    flat_b = set().union(*rows_b) if rows_b else set()
    hits_a = [int(any(c in flat_b for c in r)) for r in rows_a]
    hits_b = [int(any(c in flat_a for c in r)) for r in rows_b]
    width = max(len(hits_a), len(hits_b))
    if width == 0:
        return np.nan
    hits_a += [0] * (width - len(hits_a))
    hits_b += [0] * (width - len(hits_b))
    if len(set(hits_a + hits_b)) > 1:
        return cohen_kappa_score(hits_a, hits_b)
    return 1.0 if hits_a == hits_b else 0.0


def kappa_units(episodes_a, episodes_b, keys, level="category"):
    """Return per-episode diagnosis and procedure kappas.

    Args:
        episodes_a: Rater A episodes.
        episodes_b: Rater B episodes.
        keys: Episode keys to score.
        level: Matching level.

    Returns:
        DataFrame with columns `dx` and `px` (NaN where neither rater coded).
    """
    return pd.DataFrame([{
        "dx": episode_kappa(annotation_sets(episodes_a[k], "diagnosis", level),
                            annotation_sets(episodes_b[k], "diagnosis", level)),
        "px": episode_kappa(annotation_sets(episodes_a[k], "procedure", level),
                            annotation_sets(episodes_b[k], "procedure", level)),
    } for k in keys])


def category_average(units):
    """Mean of the diagnosis and procedure mean kappas."""
    return (np.nanmean(units.dx) + np.nanmean(units.px)) / 2


def category_average_ci(units, n_boot=N_BOOT, seed=SEED):
    """Category-average kappa with a 95% episode bootstrap interval.

    Returns:
        Tuple of (estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    draws = [category_average(units.iloc[rng.integers(0, len(units), len(units))])
             for _ in range(n_boot)]
    return (category_average(units),
            np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5))


def bootstrap_mean(values, rng, n_boot=N_BOOT):
    """Mean of non-missing values with a 95% bootstrap interval.

    Returns:
        Dict with n, mean, lo and hi.
    """
    v = np.asarray([x for x in values if not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n": len(v), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def set_scores(model, reference, keys, rng, n_boot=N_BOOT, level="category"):
    """Directional set precision, recall and F1 against a reference rater.

    Precision is averaged over episodes where the model coded that type, recall
    over episodes where the reference did, and F1 where both did.

    Args:
        model: Model episodes.
        reference: Reference episodes.
        keys: Episode keys to score.
        rng: Random generator, shared across code types.
        n_boot: Bootstrap draws.
        level: Matching level.

    Returns:
        DataFrame with one row per code type.
    """
    per = {ct: {"p": [], "r": [], "f": []} for ct in CODE_TYPES}
    for key in keys:
        for ct in CODE_TYPES:
            a = code_sets(model[key], ct, level)
            b = code_sets(reference[key], ct, level)
            hit = len(a & b)
            if a:
                per[ct]["p"].append(hit / len(a))
            if b:
                per[ct]["r"].append(hit / len(b))
            if a and b:
                p, r = hit / len(a), hit / len(b)
                per[ct]["f"].append(2 * p * r / (p + r) if p + r else 0.0)
    rows = []
    for ct in CODE_TYPES:
        p, r, f = (bootstrap_mean(per[ct][k], rng, n_boot) for k in ("p", "r", "f"))
        rows.append({"code_type": ct,
                     "precision": p["mean"], "precision_lo": p["lo"], "precision_hi": p["hi"],
                     "recall": r["mean"], "recall_lo": r["lo"], "recall_hi": r["hi"],
                     "f1": f["mean"], "f1_lo": f["lo"], "f1_hi": f["hi"], "n": f["n"]})
    return pd.DataFrame(rows)


def outlier_contrast(clinicians, triples, rng, n_boot=N_BOOT):
    """Within-episode check that no clinician diverges from the other two.
    
    Args:
        clinicians: `{name: episodes}` for the three clinicians.
        triples: Keys annotated by all three.
        rng: Random generator.
        n_boot: Bootstrap draws.

    Returns:
        DataFrame with one row per clinician.
    """
    def jaccard(a, b, key):
        result = overlap(code_sets(clinicians[a][key], "combined"),
                         code_sets(clinicians[b][key], "combined"))
        return None if result is None else result["jaccard"]

    names = list(clinicians)
    rows = []
    for target in names:
        others = [n for n in names if n != target]
        contrasts = []
        for key in triples:
            with_target = [v for v in (jaccard(target, o, key) for o in others) if v is not None]
            between = jaccard(others[0], others[1], key)
            if between is not None and with_target:
                contrasts.append(np.mean(with_target) - between)
        contrasts = np.asarray(contrasts)
        if len(contrasts) == 0:
            continue
        draws = rng.choice(contrasts, size=(n_boot, len(contrasts)), replace=True).mean(axis=1)
        rows.append({"clinician": target, "n_triples": len(contrasts),
                     "contrast": float(contrasts.mean()),
                     "lo": float(np.percentile(draws, 2.5)),
                     "hi": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)
