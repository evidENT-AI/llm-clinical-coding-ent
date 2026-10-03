#!/usr/bin/env python3
"""Load the study inputs: clinician workbooks, coder extract and model outputs.

Paths are read from a TOML study file (see `config/study.example.toml`). Every
loader returns `{episode key: {dx, px, dx_rows, px_rows, nid, adm}}`, where
`dx`/`px` are code sets and `*_rows` hold one code set per annotation.
"""

import json
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis.codes import (alnum_id, code_set, norm_id, normalise_code,
                            normalise_diagnosis_code, parse_date, swapped)

CLINICIANS = ("reference", "clinician_2", "clinician_3")
CODER_DX = ("DiagnosisCode", "DiagnosisCode5", "DiagnosisCode6")
CODER_PX = ("ProcedureCode", "ProcedureCode2", "ProcedureCode3", "ProcedureCode4")


def load_study(path):
    """Read the study TOML file.

    Args:
        path: Path to the study file.

    Returns:
        Parsed settings, with `output_dir` as a Path.
    """
    with open(path, "rb") as f:
        study = tomllib.load(f)
    study["output_dir"] = Path(study.get("output_dir", "results"))
    study["output_dir"].mkdir(parents=True, exist_ok=True)
    return study


def _episode(dx_cells, px_cells, nid, adm):
    return {
        "dx": code_set(dx_cells, normalise_diagnosis_code),
        "px": code_set(px_cells, normalise_code),
        "dx_rows": [s for s in (code_set([c], normalise_diagnosis_code) for c in dx_cells) if s],
        "px_rows": [s for s in (code_set([c], normalise_code) for c in px_cells) if s],
        "nid": nid,
        "adm": adm,
    }


def load_clinician(path, sheet):
    """Load one clinician's annotations, one entry per patient episode.

    Args:
        path: Annotation workbook.
        sheet: Worksheet name.

    Returns:
        Episodes keyed by alphanumeric hospital number.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[df.LocalPatientIdentifier.notna()].copy()
    df["key"] = df.LocalPatientIdentifier.map(alnum_id)
    episodes = {}
    for key, group in df.groupby("key"):
        adm = next((parse_date(x) for x in group.AdmissionDate if pd.notna(x)), None)
        episodes[key] = _episode(list(group["ICD-10"]), list(group["OPCS 4.11"]),
                                 norm_id(group.LocalPatientIdentifier.iloc[0]), adm)
    return episodes


def load_clinicians(study):
    """Load the reference clinician and the two additional clinicians."""
    return {name: load_clinician(study["clinicians"][name]["path"],
                                 study["clinicians"][name]["sheet"])
            for name in CLINICIANS}


def load_coder(study):
    """Load the clinical coding extract, grouped by hospital number.

    Args:
        study: Study settings.

    Returns:
        `{nid: [episode, ...]}`; a patient may have several admissions.
    """
    df = pd.read_excel(study["coder"]["path"], engine="pyxlsb",
                       sheet_name=study["coder"]["sheet"])
    by_patient = defaultdict(list)
    for row in df.itertuples(index=False):
        by_patient[norm_id(row.LocalPatientIdentifier)].append(_episode(
            [getattr(row, c) for c in CODER_DX], [getattr(row, c) for c in CODER_PX],
            norm_id(row.LocalPatientIdentifier), parse_date(row.AdmissionDate)))
    return by_patient


def coder_episode(coder, nid, adm):
    """Return the coder episode for a patient and admission date.

    Matches the date or its day/month swap; a patient with a single coder
    episode matches regardless of date.
    """
    episodes = coder.get(nid, [])
    if not episodes:
        return None
    valid = adm is not None and not pd.isna(adm)
    dates = ({adm, swapped(adm)} - {None}) if valid else set()
    hits = [e for e in episodes if e["adm"] in dates]
    if hits:
        return hits[0]
    return episodes[0] if len(episodes) == 1 else None


def coder_pairs(episodes, coder):
    """Return the coder episode matched to each annotated episode."""
    pairs = {}
    for key, episode in episodes.items():
        if not annotated(episode):
            continue
        match = coder_episode(coder, episode["nid"], episode["adm"])
        if match is not None:
            pairs[key] = match
    return pairs


def load_model(run_dir, stage="stage2"):
    """Load a model's coded output for every episode in a run directory.

    Args:
        run_dir: Directory of `<patient>_<date>/` episode folders.
        stage: "stage1" (wide extraction) or "stage2" (scoped).

    Returns:
        Episodes keyed by alphanumeric hospital number.
    """
    episodes = {}
    for folder in sorted(Path(run_dir).glob("*_*")):
        path = folder / f"{stage}_output.json"
        if not path.exists():
            continue
        labels = json.loads(path.read_text()).get("labels", [])
        patient, _, date = folder.name.rpartition("_")
        dx = [lab.get("icd10") for lab in labels
              if str(lab.get("qualifier", "")).strip().lower() == "diagnoses"]
        px = [lab.get("opcs4") for lab in labels
              if str(lab.get("qualifier", "")).strip().lower() == "procedures"]
        try:
            adm = datetime.strptime(date, "%Y%m%d").date()
        except ValueError:
            adm = None
        episodes[alnum_id(patient)] = _episode(dx, px, norm_id(patient), adm)
    return episodes


def annotated(episode):
    """Return True if an episode has at least one code."""
    return bool(episode["dx"] or episode["px"])


def shared(a, b):
    """Return keys annotated in both episode collections, in `a`'s order."""
    return [k for k in a if k in b and annotated(a[k]) and annotated(b[k])]
