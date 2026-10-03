#!/usr/bin/env python3
"""Pilot agreement between the reference clinician and the clinical coder.

For each episode the outcome is whether the coder's code falls in the
clinician's code set (category level). Produces Table 1: kappa by code type
and the pooled match probability that anchors the power calculation.

    python -m analysis.pilot --study config/study.toml
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis.codes import normalise_code, normalise_diagnosis_code, split_codes
from analysis.data import load_study


def patient_ids(series):
    return (series.astype(str).str.strip()
            .str.replace(r"[^0-9]", "", regex=True).str.lstrip("0"))


def parse_dates(series, dayfirst):
    """Parse date text, dropping any time, and recover Excel serial numbers."""
    text = series.astype(str).str.strip().str.split(r"\s+", n=1).str[0]
    serial = text.str.fullmatch(r"[0-9]+(\.[0-9]+)?").fillna(False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if serial.any():
        days = pd.to_numeric(text[serial]).astype(int)
        out[serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(days, unit="D")
    out[~serial] = pd.to_datetime(text[~serial], dayfirst=dayfirst, format="mixed",
                                  errors="coerce")
    return out.dt.date


def load_pilot_coder(path):
    """One row per coded episode with its primary diagnosis and procedure."""
    raw = pd.read_csv(path, encoding="latin1", low_memory=False)
    return pd.DataFrame({
        "patient_id": patient_ids(raw.LocalPatientIdentifier),
        "admission": parse_dates(raw.AdmissionDate, dayfirst=False),
        "discharge": parse_dates(raw.DischargeDate, dayfirst=False),
        "diagnosis": raw.DiagnosisCode.map(normalise_diagnosis_code),
        "procedure": raw.ProcedureCode.map(normalise_code),
    })


def load_pilot_reference(path):
    """One row per episode with the clinician's diagnosis and procedure codes.

    A row counts as a diagnosis or procedure by which code column is filled,
    and codes keep the order the clinician wrote them.
    """
    raw = pd.read_csv(path, encoding="latin1")
    df = pd.DataFrame({
        "patient_id": patient_ids(raw.LocalPatientIdentifier),
        "admission": parse_dates(raw.AdmissionDate, dayfirst=True),
        "discharge": parse_dates(raw.DischargeDate, dayfirst=True),
        "icd10": raw["ICD-10"],
        "opcs": raw["OPCS 4.11"],
    })

    def ordered(cells, normaliser):
        codes = []
        for cell in cells:
            for part in split_codes(cell):
                code = normaliser(part)
                if code and code not in codes:
                    codes.append(code)
        return codes

    return (df.dropna(subset=["admission", "discharge"])
            .groupby(["patient_id", "admission", "discharge"], as_index=False)
            .apply(lambda g: pd.Series({
                "diagnosis_set": ordered(g.icd10, normalise_diagnosis_code),
                "procedure_set": ordered(g.opcs, normalise_code),
            }), include_groups=False))


def score(coder_codes, reference_sets):
    """Category-level match and Cohen's kappa for one code type.

    The clinician's rating is the coder's code on a match, otherwise her first
    code; this only labels a disagreement, never decides it.

    Returns:
        Dict with n, kappa and the per-episode match indicators.
    """
    hits, coder_rating, reference_rating = [], [], []
    for code, ref in zip(coder_codes, reference_sets):
        if code is None or not ref:
            continue
        code, ref = code[:3], [c[:3] for c in ref]
        matched = code in ref
        hits.append(matched)
        coder_rating.append(code)
        reference_rating.append(code if matched else ref[0])
    hits = np.array(hits)
    kappa = cohen_kappa_score(coder_rating, reference_rating) if len(hits) else np.nan
    return {"n": len(hits), "kappa": kappa, "hits": hits}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="config/study.toml")
    args = parser.parse_args()

    study = load_study(args.study)
    matched = load_pilot_coder(study["pilot"]["coder_csv"]).merge(
        load_pilot_reference(study["pilot"]["reference_csv"]),
        on=["patient_id", "admission", "discharge"], how="inner")

    rows, hits = [], []
    for label, coder_col, ref_col in (("Diagnosis", "diagnosis", "diagnosis_set"),
                                      ("Procedure", "procedure", "procedure_set")):
        result = score(matched[coder_col], matched[ref_col])
        hits.append(result["hits"])
        rows.append({"category": label, "n": result["n"], "kappa": result["kappa"]})
    hits = np.concatenate(hits)
    p = hits.mean()
    rows.append({"category": "Diagnosis and procedure", "n": len(hits), "match_p": p,
                 "match_p_ci_halfwidth": 1.96 * np.sqrt(p * (1 - p) / len(hits))})

    table = pd.DataFrame(rows)
    table.to_csv(study["output_dir"] / "pilot_agreement.csv", index=False)
    print(f"{len(matched)} linked episodes")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
