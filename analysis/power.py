#!/usr/bin/env python3
"""Sample size and power (Supplementary Material A).

    python -m analysis.power apriori              # Tables S1 to S3, main Table 2
    python -m analysis.power design               # triple-annotation design
    python -m analysis.power posthoc --study config/study.toml

The model-versus-coder comparison is a paired binary outcome (does each
comparator's code fall in the clinician's set?) tested with McNemar's test.
Correlated successes are simulated with a bivariate normal threshold model.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

SEED = 42
P_CODER = 0.52
DELTAS = (0.10, 0.15, 0.20, 0.25, 0.30)
PHIS = (0.2, 0.4, 0.6)
TARGETS = (0.80, 0.90, 0.95)


def mcnemar_power(n, p_model, p_coder, phi, n_sims, rng, alpha=0.05):
    """Power of McNemar's test (continuity corrected) for paired successes.

    Args:
        n: Episodes.
        p_model: Model match probability.
        p_coder: Coder match probability.
        phi: Latent correlation between the two comparators' successes.
        n_sims: Simulated datasets.
        rng: Random generator.
        alpha: Significance level.

    Returns:
        Proportion of simulations that reject.
    """
    cov = np.array([[1.0, phi], [phi, 1.0]])
    draws = rng.multivariate_normal([0, 0], cov, size=(n_sims, n))
    model_hit = draws[:, :, 0] < stats.norm.ppf(p_model)
    coder_hit = draws[:, :, 1] < stats.norm.ppf(p_coder)
    b = (model_hit & ~coder_hit).sum(axis=1)
    c = (~model_hit & coder_hit).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (np.abs(b - c) - 1) ** 2 / (b + c)
    chi2 = np.where(b + c == 0, 0.0, chi2)
    return (chi2 > stats.chi2.ppf(1 - alpha, df=1)).mean()


def episodes_needed(p_model, p_coder, phi, n_sims, rng, target, cap=1000):
    """Smallest number of episodes reaching the target power (binary search)."""
    lo, hi = 10, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, p_model, p_coder, phi, n_sims, rng) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < cap else np.inf


def apriori(n_sims=20000):
    """Episodes needed by effect size and correlation at each power target.

    Returns:
        Long table of (power, delta, phi, episodes).
    """
    rows = []
    for target in TARGETS:
        rng = np.random.default_rng(SEED)
        for delta in DELTAS:
            for phi in PHIS:
                n = episodes_needed(min(P_CODER + delta, 0.98), P_CODER, phi, n_sims, rng, target)
                rows.append({"power": target, "delta": delta, "phi": phi, "episodes": n})
    return pd.DataFrame(rows)


def _episode_difficulty(rng, n, p, rho):
    concentration = 1.0 / rho - 1.0
    return rng.beta(p * concentration, (1.0 - p) * concentration, size=n)


def outlier_power(rng, n_triple, p_mm, rho, delta, items, n_sims, alpha=0.05):
    """Power to detect the reference clinician agreeing `delta` below her peers.

    Each triple episode shares a difficulty q; the two reference pairs agree
    with probability q - delta, the other pair with q. A one-sided t-test is
    applied to the within-episode contrast.
    """
    if n_triple < 2:
        return 0.0
    critical = stats.t.ppf(1 - alpha, n_triple - 1)
    hits = 0
    for _ in range(n_sims):
        q = _episode_difficulty(rng, n_triple, p_mm, rho)
        ref = rng.binomial(1, np.clip(q - delta, 0, 1)[:, None],
                           size=(n_triple, 2 * items)).mean(1)
        others = rng.binomial(1, q[:, None], size=(n_triple, items)).mean(1)
        contrast = ref - others
        sd = contrast.std(ddof=1)
        if sd == 0:
            continue
        if contrast.mean() / (sd / np.sqrt(n_triple)) < -critical:
            hits += 1
    return hits / n_sims


def _simulate_design(rng, n_triple, n_double, p_mm, rho, items):
    agree, episode = [], []
    for eid in range(n_triple + n_double):
        q = _episode_difficulty(rng, 1, p_mm, rho)[0]
        for _ in range(items):
            if eid < n_triple:
                agree += [rng.binomial(1, q), rng.binomial(1, q), rng.binomial(1, q)]
                episode += [eid] * 3
            else:
                agree.append(rng.binomial(1, q))
                episode.append(eid)
    return np.asarray(agree), np.asarray(episode)


def _cluster_bootstrap(rng, agree, episode, n_boot):
    clusters = np.unique(episode)
    grouped = {c: agree[episode == c] for c in clusters}
    means = [np.concatenate([grouped[c] for c in rng.choice(clusters, len(clusters))]).mean()
             for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def ceiling_halfwidth(rng, n_triple, n_double, p_mm, rho, items, n_reps, n_boot):
    """Median 95% interval half-width of the pooled ceiling."""
    widths = []
    for _ in range(n_reps):
        lo, hi = _cluster_bootstrap(rng, *_simulate_design(rng, n_triple, n_double,
                                                           p_mm, rho, items), n_boot)
        widths.append((hi - lo) / 2)
    return float(np.median(widths))


def ceiling_above_coder(rng, n_triple, n_double, p_mm, rho, items, n_reps, n_boot):
    """Power for the ceiling's lower 95% bound to exceed the coder's 0.52."""
    hits = 0
    for _ in range(n_reps):
        lo, _ = _cluster_bootstrap(rng, *_simulate_design(rng, n_triple, n_double,
                                                          p_mm, rho, items), n_boot)
        hits += lo > P_CODER
    return hits / n_reps


def design(p_mm=0.65, rho=0.30, items=2, n_sims=3000, n_reps=400, n_boot=800):
    """Simulate the triple-annotation design (Supplementary Material A, S1.4).

    Returns:
        Tuple of (outlier power, ceiling half-width, power to clear the coder).
    """
    rng = np.random.default_rng(SEED)
    outlier = pd.DataFrame([
        {"n_triple": nt, **{f"delta={d}": outlier_power(rng, nt, p_mm, rho, d, items, n_sims)
                            for d in (0.10, 0.15, 0.20)}}
        for nt in (15, 20, 25, 30, 40, 50)])
    halfwidth = pd.DataFrame([
        {"n_triple": nt, "n_double": nd,
         "halfwidth": ceiling_halfwidth(rng, nt, nd, p_mm, rho, items, n_reps, n_boot)}
        for nt, nd in ((0, 40), (20, 0), (20, 20), (20, 40), (30, 10), (30, 20),
                       (40, 0), (30, 30))])
    above = pd.DataFrame([
        {"n_triple": nt, "n_double": nd,
         **{f"p_mm={p}": ceiling_above_coder(rng, nt, nd, p, rho, items, n_reps, n_boot)
            for p in (0.58, 0.62, 0.65, 0.70)}}
        for nt, nd in ((20, 10), (30, 10), (30, 20), (40, 20))])
    return outlier, halfwidth, above


def tetrachoric(a, b, c, d):
    """Latent correlation behind a 2x2 table of paired successes.

    Args:
        a: Both succeed. b: First only. c: Second only. d: Neither.

    Returns:
        The correlation, or NaN if it cannot be solved.
    """
    n = a + b + c + d
    h = stats.norm.ppf(min(max((a + b) / n, 1e-6), 1 - 1e-6))
    k = stats.norm.ppf(min(max((a + c) / n, 1e-6), 1 - 1e-6))

    def gap(r):
        return stats.multivariate_normal.cdf([h, k], mean=[0, 0], cov=[[1, r], [r, 1]]) - a / n

    if gap(-0.999) * gap(0.999) > 0:
        return np.nan
    return brentq(gap, -0.999, 0.999)


def minimum_detectable(n, p_coder, phi, target, n_sims=8000):
    """Smallest difference in match probability detectable at `target` power."""
    rng = np.random.default_rng(SEED)
    for delta in np.arange(0.02, 0.60, 0.01):
        if mcnemar_power(n, min(p_coder + delta, 0.98), p_coder, phi, n_sims, rng) >= target:
            return round(delta, 2)
    return np.nan


def posthoc(study_path):
    """Observed correlation and minimum detectable effect per model.

    Returns:
        One row per model.
    """
    from analysis.agreement import code_sets, overlap
    from analysis.data import (annotated, coder_pairs, load_clinicians, load_coder,
                               load_model, load_study)

    study = load_study(study_path)
    ref = load_clinicians(study)["reference"]
    coder = load_coder(study)
    rows = []
    for name, paths in study["models"].items():
        model = load_model(paths["run"])
        pairs = coder_pairs(model, coder)
        keys = [k for k in model if annotated(model[k]) and k in ref and annotated(ref[k])
                and k in pairs and annotated(pairs[k])]

        def anymatch(episodes, key):
            result = overlap(code_sets(episodes[key], "combined"), code_sets(ref[key], "combined"))
            return np.nan if result is None else result["anymatch"]

        m = np.array([anymatch(model, k) for k in keys], float)
        c = np.array([anymatch(pairs, k) for k in keys], float)
        keep = ~(np.isnan(m) | np.isnan(c))
        m, c = m[keep], c[keep]
        both, model_only = int(((m == 1) & (c == 1)).sum()), int(((m == 1) & (c == 0)).sum())
        coder_only, neither = int(((m == 0) & (c == 1)).sum()), int(((m == 0) & (c == 0)).sum())
        phi = tetrachoric(both, model_only, coder_only, neither)
        rho = phi if not np.isnan(phi) else 0.2
        rows.append({"model": name, "n": len(m), "p_model": m.mean(), "p_coder": c.mean(),
                     "delta_p": m.mean() - c.mean(), "tetrachoric": phi,
                     **{f"mde_{int(t * 100)}": minimum_detectable(len(m), c.mean(), rho, t)
                        for t in TARGETS}})
    return pd.DataFrame(rows), study["output_dir"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=("apriori", "design", "posthoc"))
    parser.add_argument("--study", default="config/study.toml")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.analysis == "posthoc":
        table, out = posthoc(args.study)
        table.to_csv(out / "power_posthoc.csv", index=False)
        print(table.round(2).to_string(index=False))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    if args.analysis == "apriori":
        table = apriori()
        table.to_csv(args.out / "power_apriori.csv", index=False)
        for target, group in table.groupby("power"):
            print(f"\n{int(target * 100)}% power")
            print(group.pivot(index="delta", columns="phi", values="episodes").to_string())
    else:
        for name, table in zip(("outlier", "ceiling_halfwidth", "ceiling_above_coder"), design()):
            table.to_csv(args.out / f"power_design_{name}.csv", index=False)
            print(f"\n{name}\n{table.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
