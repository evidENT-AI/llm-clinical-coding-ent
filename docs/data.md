# Input formats

None of these files are distributed. This describes what the code expects.

## Episode bundles

One JSON file per episode, named `<hospital number>_<YYYYMMDD>.json`, built by
`extraction/build_episode_notes.py`. The pipeline uses only `notes`:

```json
{
  "episode_id": "SYN000001_20240115",
  "notes": [
    {"note_date": "2024-01-15", "note_class": "Note", "is_anchor": true, "text": "..."}
  ]
}
```

See `data/synthetic/episodes/` for a complete example.

## Clinician annotations

One Excel workbook per clinician, one row per annotation, with at least:

| Column | Content |
|---|---|
| `LocalPatientIdentifier` | hospital number |
| `AdmissionDate` | admission date |
| `ICD-10` | diagnosis code(s), space separated |
| `OPCS 4.11` | procedure code(s), space separated |
| `text`, `context` | annotated text and its sentence (used to anchor notes) |

## Clinical coding extract

The hospital coding extract (`.xlsb`), one row per coded episode, with
`LocalPatientIdentifier`, `AdmissionDate` (Excel serial), `DiagnosisCode`,
`DiagnosisCode5`, `DiagnosisCode6` and `ProcedureCode` to `ProcedureCode4`.
Episodes are matched on hospital number and admission date, allowing for a
day/month swap.

## Pilot files

Two CSV exports used for Table 1: the coding extract (`LocalPatientIdentifier`,
`AdmissionDate`, `DischargeDate` in US order, `DiagnosisCode`, `ProcedureCode`)
and the reference clinician's pilot annotations (`LocalPatientIdentifier`,
`AdmissionDate`, `DischargeDate` in UK order, `ICD-10`, `OPCS 4.11`). Episodes
are matched on hospital number, admission and discharge date.

## Model runs

`pipeline.run_batch` writes one folder per episode. The analyses read
`stage2_output.json` (or `stage1_output.json`), whose labels carry `qualifier`
(`diagnoses` or `procedures`), `snomed_ct`, `icd10` and `opcs4`.

## Clinician evaluations

One workbook per model, one row per model code or missed concept. The verdict
column is found automatically and holds free text such as:

| Verdict text | Category |
|---|---|
| `TP ...` | correct and central |
| `FP - additional not relevant` | valid but not central |
| any other `FP ...` | not applicable (hallucination) |
| `FN ...` mentioning a correct extraction, `incorrect code` or `no code` | coding error |
| any other `FN ...` | concept not extracted |
| `TN ...` | ruled-out finding correctly not coded |

`qualifier` and `primary_secondary` columns are read where present.
