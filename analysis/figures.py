#!/usr/bin/env python3
"""Figures 1 to 3, drawn from the CSVs written by the other analyses.

    python -m analysis.figures --study config/study.toml
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.data import load_study  # noqa: E402

COLOURS = {"Llama 3.1 8B": "#8b5cf6", "GPT-4o": "#0d9488", "GPT-5": "#5e81ac",
           "Llama 4 Scout": "#e89611", "Coder": "#4b5563"}
MARKERS = {"Llama 3.1 8B": "o", "GPT-4o": "s", "GPT-5": "^", "Llama 4 Scout": "D",
           "Coder": "X"}
METRIC_COLOURS = {"Precision": "#5e81ac", "Recall": "#688448", "F1": "#d97706"}
GREY, DARK = "#4b5563", "#374151"
ALPHA = 0.05

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "legend.fontsize": 11, "xtick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.linewidth": 0.9, "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#1f2933", "text.color": "#1f2933",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _fill_legend(filled, empty):
    return [Line2D([0], [0], marker="o", color=GREY, mfc=GREY, mec=GREY, ls="none", ms=9,
                   label=filled),
            Line2D([0], [0], marker="o", color=GREY, mfc="white", mec=GREY, ls="none", ms=9,
                   label=empty)]


def _forest(ax, names, estimates, lows, highs, filled):
    for y, (name, est, lo, hi, fill) in enumerate(zip(names, estimates, lows, highs, filled)):
        colour = COLOURS.get(name, GREY)
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="none", ecolor=colour,
                    elinewidth=1.8, capsize=4, capthick=1.8, zorder=2)
        ax.plot(est, y, marker=MARKERS.get(name, "o"), ms=10, mfc=colour if fill else "white",
                mec=colour, mew=1.8, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.grid(axis="x", color="#eef0f2", lw=0.8)
    ax.set_axisbelow(True)


def figure_h1(h1, path):
    """Figure 1: difference in kappa, model minus coder, paired by episode."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvline(0, color="#9ca3af", ls="--", lw=1.1, zorder=1)
    _forest(ax, list(h1.model), h1.delta_kappa, h1.delta_kappa_lo, h1.delta_kappa_hi,
            h1.delta_kappa_p_fdr < ALPHA)
    ax.set_xlabel("Difference in per-episode kappa (LLM minus coder)")
    ax.legend(handles=_fill_legend("Significant (FDR p < 0.05)", "Not significant"),
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    _save(fig, path)


def figure_h2(h2, clinicians, models, path):
    """Figure 2: agreement with the clinician against the clinician ceiling."""
    ceiling = clinicians[clinicians.comparison.str.startswith("ceiling")].iloc[0]
    coder = clinicians[clinicians.comparison == "coder vs reference"].iloc[0]
    vs_ref = models[models.reference == "clinician"]
    names = ["Coder"] + list(vs_ref.model)
    estimates = [coder.kappa] + list(vs_ref.kappa)
    lows = [coder.lo] + list(vs_ref.kappa_lo)
    highs = [coder.hi] + list(vs_ref.kappa_hi)
    below = list(h2.set_index("comparator").loc[["coder"] + list(vs_ref.model), "p_fdr"] < ALPHA)

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.axvspan(ceiling.lo, ceiling.hi, color=DARK, alpha=0.10, zorder=0)
    ax.axvline(ceiling.kappa, color=DARK, lw=1.4, zorder=1)
    _forest(ax, names, estimates, lows, highs, below)
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("Agreement with clinician (per-episode kappa)")
    handles = [Line2D([0], [0], color=DARK, lw=1.4,
                      label=f"Clinician ceiling (kappa {ceiling.kappa:.2f})"),
               Line2D([0], [0], marker="s", color=DARK, alpha=0.3, mec="none", ls="none",
                      ms=12, label="Ceiling 95% CI")]
    handles += _fill_legend("Below ceiling (FDR p < 0.05)", "Reaches ceiling")
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=1.4)
    _save(fig, path)


def figure_performance(adjudication, path):
    """Figure 3: precision, recall and F1 at the validity and alignment levels."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    names = list(adjudication.model)
    x = np.arange(len(names))
    width = 0.8 / 3
    tiers = (("validity", "Validity"), ("alignment", "Clinical alignment"))
    for ax, (tier, title) in zip(axes, tiers):
        for j, metric in enumerate(("precision", "recall", "f1")):
            label = "F1" if metric == "f1" else metric.title()
            ax.bar(x + (j - 1) * width, adjudication[f"{tier}_{metric}"], width=width * 0.92,
                   color=METRIC_COLOURS[label], edgecolor="white", linewidth=0.6, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#eef0f2", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    fig.legend(handles=[Patch(facecolor=c, label=m) for m, c in METRIC_COLOURS.items()],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def _save(fig, path):
    if fig.get_axes() and not fig.get_layout_engine():
        fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="config/study.toml")
    args = parser.parse_args()

    out = load_study(args.study)["output_dir"]
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    figure_h1(pd.read_csv(out / "h1.csv"), folder / "figure1_h1")
    figure_h2(pd.read_csv(out / "h2.csv"), pd.read_csv(out / "agreement_clinicians.csv"),
              pd.read_csv(out / "agreement_models.csv"), folder / "figure2_h2")
    figure_performance(pd.read_csv(out / "adjudication.csv"), folder / "figure3_performance")
    print(f"Figures written to {folder}")


if __name__ == "__main__":
    main()
