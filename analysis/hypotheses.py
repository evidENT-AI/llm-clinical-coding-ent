#!/usr/bin/env python3
"""Agreement with the reference clinician and the tests of H1 and H2.

Writes, to the study output directory:
    agreement_clinicians.csv  clinician pairs, the ceiling and the coder
    agreement_models.csv      each model against the clinician and the coder
    h1.csv                    model versus coder, paired by episode
    h2.csv                    the ceiling versus the coder and each model
    outlier_contrast.csv      reference clinician versus her peers

    python -m analysis.hypotheses --study config/study.toml
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from analysis.agreement import (N_BOOT, SEED, category_average, category_average_ci,
                                code_sets, kappa_units, outlier_contrast, overlap,
                                set_scores)
from analysis.data import (annotated, coder_pairs, load_clinicians, load_coder,
                           load_model, load_study, shared)


def kappa_row(label, units, **extra):
    """Category-average kappa and interval as a table row."""
    estimate, lo, hi = category_average_ci(units)
    return {"comparison": label, "n": len(units), "kappa": estimate, "lo": lo, "hi": hi, **extra}


def clinician_agreement(clinicians, coder):
    """Kappa for each clinician pair, the ceiling and the coder.

    The ceiling pools the two pairs that involve the reference clinician, since
    every comparator is scored against her.

    Returns:
        Tuple of (table, ceiling units, coder units).
    """
    ref = clinicians["reference"]
    rows, pair_units = [], {}
    for a, b in combinations(clinicians, 2):
        units = kappa_units(clinicians[a], clinicians[b], shared(clinicians[a], clinicians[b]))
        pair_units[(a, b)] = units
        rows.append(kappa_row(f"{a} vs {b}", units))
    ceiling = pd.concat([pair_units[("reference", "clinician_2")],
                         pair_units[("reference", "clinician_3")]], ignore_index=True)
    rows.append(kappa_row("ceiling (pairs with reference)", ceiling))
    rows.append(kappa_row("all clinician pairs", pd.concat(pair_units.values(),
                                                           ignore_index=True)))
    pairs = coder_pairs(ref, coder)
    coder_units = kappa_units(ref, pairs, shared(ref, pairs))
    rows.append(kappa_row("coder vs reference", coder_units))
    return pd.DataFrame(rows), ceiling, coder_units


def model_agreement(name, model, ref, coder):
    """Kappa and set scores for one model against the clinician and the coder.

    Returns:
        Tuple of (rows, units against the clinician).
    """
    rows, ref_units = [], None
    for label, other in (("clinician", ref), ("coder", coder_pairs(model, coder))):
        keys = shared(model, other)
        units = kappa_units(model, other, keys)
        if label == "clinician":
            ref_units = units
        estimate, lo, hi = category_average_ci(units)
        scores = set_scores(model, other, keys, np.random.default_rng(SEED))
        combined = scores[scores.code_type == "combined"].iloc[0]
        rows.append({"model": name, "reference": label, "n": len(units),
                     "kappa": estimate, "kappa_lo": lo, "kappa_hi": hi,
                     **{k: combined[k] for k in ("precision", "precision_lo", "precision_hi",
                                                 "recall", "recall_lo", "recall_hi",
                                                 "f1", "f1_lo", "f1_hi")}})
    return rows, ref_units


def test_h1(name, model, ref, coder):
    """Compare a model with the coder in agreement with the clinician.

    Uses episodes coded by all three. Reports McNemar's test on episode-level
    any-match and the paired bootstrap difference in category-average kappa.
    """
    pairs = coder_pairs(model, coder)
    keys = [k for k in model if annotated(model[k]) and k in ref and annotated(ref[k])
            and k in pairs and annotated(pairs[k])]

    def anymatch(episodes, key):
        result = overlap(code_sets(episodes[key], "combined"), code_sets(ref[key], "combined"))
        return np.nan if result is None else result["anymatch"]

    m = np.array([anymatch(model, k) for k in keys], float)
    c = np.array([anymatch(pairs, k) for k in keys], float)
    keep = ~(np.isnan(m) | np.isnan(c))
    m, c, keys = m[keep], c[keep], [k for k, ok in zip(keys, keep) if ok]
    n = len(m)

    rng = np.random.default_rng(SEED)
    dp_draws = np.array([(lambda i: m[i].mean() - c[i].mean())(rng.integers(0, n, n))
                         for _ in range(N_BOOT)])
    b, d = int(np.sum((m == 1) & (c == 0))), int(np.sum((m == 0) & (c == 1)))
    stat = (abs(b - d) - 1) ** 2 / (b + d) if b + d else 0.0
    p_mcnemar = 1 - chi2.cdf(stat, 1) if b + d else 1.0

    model_units, coder_units = kappa_units(model, ref, keys), kappa_units(pairs, ref, keys)
    rng = np.random.default_rng(SEED)
    dk_draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        dk_draws.append(category_average(model_units.iloc[i])
                        - category_average(coder_units.iloc[i]))
    dk_draws = np.array(dk_draws)
    return {
        "model": name, "n": n, "model_only": b, "coder_only": d,
        "delta_p": m.mean() - c.mean(),
        "delta_p_lo": np.percentile(dp_draws, 2.5), "delta_p_hi": np.percentile(dp_draws, 97.5),
        "mcnemar_chi2": stat, "mcnemar_p": p_mcnemar,
        "model_kappa": category_average(model_units), "coder_kappa": category_average(coder_units),
        "delta_kappa": category_average(model_units) - category_average(coder_units),
        "delta_kappa_lo": np.nanpercentile(dk_draws, 2.5),
        "delta_kappa_hi": np.nanpercentile(dk_draws, 97.5),
        "delta_kappa_p": 2 * min((dk_draws <= 0).mean(), (dk_draws >= 0).mean()),
    }


def test_h2(label, ceiling, other):
    """One-sided bootstrap test that the ceiling exceeds another agreement."""
    rng = np.random.default_rng(SEED)
    draws = np.array([
        category_average(ceiling.iloc[rng.integers(0, len(ceiling), len(ceiling))])
        - category_average(other.iloc[rng.integers(0, len(other), len(other))])
        for _ in range(N_BOOT)])
    return {"comparator": label, "comparator_kappa": category_average(other),
            "delta_kappa": category_average(ceiling) - category_average(other),
            "lo": np.nanpercentile(draws, 2.5), "hi": np.nanpercentile(draws, 97.5),
            "p_one_sided": (draws <= 0).mean()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="config/study.toml")
    args = parser.parse_args()

    study = load_study(args.study)
    out = study["output_dir"]
    clinicians = load_clinicians(study)
    coder = load_coder(study)
    ref = clinicians["reference"]

    clin_table, ceiling, coder_units = clinician_agreement(clinicians, coder)
    clin_table.to_csv(out / "agreement_clinicians.csv", index=False)

    agreement, h1, h2 = [], [], [test_h2("coder", ceiling, coder_units)]
    for name, paths in study["models"].items():
        model = load_model(paths["run"])
        rows, ref_units = model_agreement(name, model, ref, coder)
        agreement += rows
        h1.append(test_h1(name, model, ref, coder))
        h2.append(test_h2(name, ceiling, ref_units))

    h1 = pd.DataFrame(h1)
    h1["mcnemar_p_fdr"] = multipletests(h1.mcnemar_p, method="fdr_bh")[1]
    h1["delta_kappa_p_fdr"] = multipletests(h1.delta_kappa_p, method="fdr_bh")[1]
    h2 = pd.DataFrame(h2)
    h2["p_fdr"] = multipletests(h2.p_one_sided, method="fdr_bh")[1]

    triples = [k for k in ref if all(k in clinicians[n] and annotated(clinicians[n][k])
                                     for n in clinicians)]
    contrast = outlier_contrast(clinicians, triples, np.random.default_rng(SEED))

    pd.DataFrame(agreement).to_csv(out / "agreement_models.csv", index=False)
    h1.to_csv(out / "h1.csv", index=False)
    h2.to_csv(out / "h2.csv", index=False)
    contrast.to_csv(out / "outlier_contrast.csv", index=False)

    pd.set_option("display.width", 160)
    for title, table in (("Clinicians and coder", clin_table),
                         ("Models", pd.DataFrame(agreement)), ("H1", h1), ("H2", h2),
                         ("Outlier contrast", contrast)):
        print(f"\n{title}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
