# llm-clinical-coding-ent

Code for *Large Language Models Reach Human-Level Agreement in Coding Otolaryngology Clinical Notes* (Farache Trajano et al.). It compares four LLMs, three clinicians and the hospital's professional clinical coders on ICD-10 and OPCS-4 coding of inpatient ENT episodes at University College London Hospitals.

No patient data are included. The study notes, annotations and model outputs remain within the UCLH Trusted Research Environment under the study's information-governance approvals. A synthetic episode is provided so the pipeline can be run end to end.

## Layout

```
pipeline/     LLM coding pipeline
  config.py          models, decoding parameters, paths
  prompts.py         prompt templates, verbatim as used
  labeller.py        the per-episode pipeline
  parser.py          parsing and standardising model output
  snomed.py          SNOMED CT search (Snowstorm)
  classification.py  SNOMED CT to ICD-10 / OPCS-4 mapping
  run_batch.py       code a directory of episodes with one model
  cost_report.py     time, tokens and estimated cost per model
  serve_vllm.sh      serve the open models locally
extraction/   build episode note bundles from the OMOP snapshot
analysis/     agreement, hypothesis tests, evaluation metrics, power, figures
config/       example study file listing the analysis inputs
data/synthetic/  synthetic in-context example and test episode
docs/         setup and input formats
tests/        unit tests
```

## Pipeline

For each episode the model is given a pre-prompt and a worked example, then annotates an example episode and is shown its gold annotation. It then extracts the diagnoses and procedures from the episode's notes. Each extraction is grounded to a SNOMED CT concept by searching Snowstorm, with a bounded loop in which the model may choose a candidate or propose a better search term. Concepts are mapped to ICD-10 or OPCS-4 using the NHS complex maps (Stage 1). Finally the model, acting as a clinical coder, selects the principal and up to two secondary diagnoses and procedures for the admission from the Stage 1 candidates (Stage 2).

| Model | Key | Served as |
|---|---|---|
| Llama 3.1 8B Instruct | `LLAMA31_8B` | local, vLLM |
| Llama 4 Scout 17B-16E Instruct (4-bit) | `LLAMA4_SCOUT` | local, vLLM |
| GPT-4o (`gpt-4o-2024-11-20`) | `GPT4O` | Azure OpenAI in the TRE |
| GPT-5 (`gpt-5-2025-08-07`) | `GPT5` | Azure OpenAI in the TRE |

Temperature is 0 and the seed 42 for every model that accepts them. GPT-5 accepts neither and was run at reasoning effort "high".

In the study the in-context example was a de-identified episode from the cohort (excluded from evaluation), which cannot be released. The synthetic example in `data/synthetic/` stands in for it; point `EXAMPLE_NOTE_PATH` and `EXAMPLE_ANNOTATION_PATH` at your own example to use another.

## Installation

```
conda env create -f environment.yml
conda activate llm-clinical-coding-ent
python -m spacy download en_core_web_lg
cp .env.example .env
```

SNOMED CT grounding needs a Snowstorm server loaded with the UK Monolith release, and the ICD-10/OPCS-4 lookup is built from the same release. Both are licensed through NHS TRUD; see [docs/setup.md](docs/setup.md).

## Running the pipeline

With Snowstorm running and a model available:

```
python -m pipeline.run_batch --model GPT4O --notes-dir data/synthetic/episodes
python -m pipeline.cost_report --results results/runs/<run id>
```

Only send synthetic data to the public OpenAI API. Real notes were processed by local models or by Azure OpenAI deployments inside the TRE.

## Reproducing the analyses

Copy `config/study.example.toml` to `config/study.toml` and point it at the annotation workbooks, the coding extract, the model runs and the clinician's evaluations (formats in [docs/data.md](docs/data.md)).

| Paper | Command | Output |
|---|---|---|
| Table 1 | `python -m analysis.pilot` | `pilot_agreement.csv` |
| Table 2, Supp. A Tables S1-S3 | `python -m analysis.power apriori` | `power_apriori.csv` |
| Supp. A S1.4 | `python -m analysis.power design` | `power_design_*.csv` |
| Supp. A S1.6 | `python -m analysis.power posthoc` | `power_posthoc.csv` |
| Sections 3.2-3.5, Table 5 | `python -m analysis.hypotheses` | `agreement_*.csv`, `h1.csv`, `h2.csv` |
| Tables 6-7, Supp. B | `python -m analysis.adjudication` | `adjudication.csv` |
| Figures 1-3 | `python -m analysis.figures` | `figures/` |

All bootstrap intervals use 2,000 resamples of episodes with seed 42.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the paper and the archived
release of this code.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
