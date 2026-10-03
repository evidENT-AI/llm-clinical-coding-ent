#!/usr/bin/env python3
"""Prompt templates, verbatim as used in the study.

Order of use: pre-prompt, worked example, in-context feedback, episode
extraction, SNOMED CT selection and refinement, then admission scoping.
"""

import json

PREPROMPT = '''
You are an expert medical professional with extensive experience in clinical documentation analysis. Your task is to extract and analyze information from clinical notes systematically and accurately. For this task you extract ONLY diagnoses and procedures — ignore all other clinical information (symptoms, signs, treatments, socio-demographics, risk factors, test results).

Key Responsibilities:
1. Extract relevant clinical information into two structured categories only:
   - Diagnoses
   - Procedures

Standards for Analysis:
- Base all extractions on explicit evidence from the text
- Maintain clinical precision and accuracy
- Avoid inferring information not directly supported by the text
- Distinguish between confirmed and suspected findings
- Consider temporal relationships in the clinical narrative
- Preserve medical terminology as presented in the source text

Required Information for Each Extraction:
- Mandatory Fields:
  * Key Text: The specific phrase or term from the source text
  * Context: The complete relevant sentence or passage containing the key text
- Optional Fields (include when applicable):
  * Laterality (left, right, bilateral, NA)
  * Presence (confirmed, suspected, resolved, NA)
  * Primary/Secondary classification
  * Experiencer (patient, family member, NA)
  * Treatment stage (pre-treatment, current, post-treatment, NA)
  * SNOMED-CT code (if confident in the mapping)

Output Requirements:
- Maintain clinical accuracy and precision
- Present information in a structured, consistent format
- Use "NA" for any field where information is not applicable or cannot be determined from the text
- Preserve the original medical terminology

Remember:
- Only extract information explicitly present in the text
- Maintain medical accuracy and precision
- Follow a systematic approach to analysis
- Be prepared to justify each extraction with specific textual evidence
- Flag any uncertainties or ambiguities in the source text
'''

EXAMPLE_PROMPT = '''
I will now show you an example clinical note and its annotation to demonstrate the task. Your job will be to replicate this type of analysis on new clinical notes, extracting ONLY diagnoses and procedures.

Example Clinical Note (synthetic):
"ENT operation note. Indication for surgery: This 58-year-old gentleman has a longstanding right-sided chronic suppurative otitis media and a cholesteatoma of the right middle ear, confirmed on preoperative CT of the temporal bones. Procedure performed: Right modified radical mastoidectomy and tympanoplasty under general anaesthesia. Operative findings: Extensive attic cholesteatoma with erosion of the long process of the incus. Post-operative diagnosis: Cholesteatoma of the right middle ear with associated chronic otitis media."

Here is how this note should be analysed. Note that only diagnoses and procedures are extracted; symptoms, findings and other categories are ignored. Each extracted label has its associated metadata in JSON format:

{
    "labels": [
      {
        "text": "cholesteatoma",
        "context": "a cholesteatoma of the right middle ear, confirmed on preoperative CT of the temporal bones.",
        "qualifier": "Diagnoses",
        "laterality": "right",
        "presence": "confirmed",
        "primary_secondary": "primary",
        "experiencer": "Patient",
        "treatment_stage": "NA",
        "snomed_ct": "363668000"
      },
      {
        "text": "chronic suppurative otitis media",
        "context": "This 58-year-old gentleman has a longstanding right-sided chronic suppurative otitis media and a cholesteatoma of the right middle ear.",
        "qualifier": "Diagnoses",
        "laterality": "right",
        "presence": "confirmed",
        "primary_secondary": "secondary",
        "experiencer": "Patient",
        "treatment_stage": "NA",
        "snomed_ct": "38394007"
      },
      {
        "text": "modified radical mastoidectomy",
        "context": "Right modified radical mastoidectomy and tympanoplasty under general anaesthesia.",
        "qualifier": "Procedures",
        "laterality": "right",
        "presence": "confirmed",
        "primary_secondary": "primary",
        "experiencer": "Patient",
        "treatment_stage": "current",
        "snomed_ct": "29656003"
      },
      {
        "text": "tympanoplasty",
        "context": "Right modified radical mastoidectomy and tympanoplasty under general anaesthesia.",
        "qualifier": "Procedures",
        "laterality": "right",
        "presence": "confirmed",
        "primary_secondary": "secondary",
        "experiencer": "Patient",
        "treatment_stage": "current",
        "snomed_ct": "386556002"
      }
    ]
}

For each new clinical note, you should:
1. Extract ONLY labels that fall into these two categories:
   - Diagnoses
   - Procedures
   Ignore symptoms, signs, treatments (that are not procedures), socio-demographics, risk factors and test results.

2. For each label, provide:
   - The exact text from the note
   - The complete context (full sentence or relevant passage)
   - The qualifier (Diagnoses or Procedures)
   - All applicable metadata fields (laterality, presence, etc.)
   - Use "NA" for any field that is not applicable or cannot be determined

3. Format your response as a JSON object following the exact structure shown in the example above.

4. Ensure that:
   - Each label is an exact quote from the text
   - Context provides sufficient information to understand the label
   - All fields are completed (using "NA" when not applicable)
   - The classification is based solely on information present in the text

Are you ready to analyze a new clinical note following this format?
'''

_FEEDBACK = '''
Well done, that was a good first attempt. Here is what an expert consultant created for the same letter for reference. Consider the differences between the example, gold-standard annotation (below) and your own. Consider the different qualifiers that you may have missed out on and the information that was in the letter that your missed in your annotations.

{annotation}

DO NOT re-attempt this annotation. Rather, learning the deep structural components that are contributed to this gold-standard annotation JSON. We will now progress on to the next letter. This will be completely independent of the previous letter and so, you should learn the deep, structural patterns behind good annotations rather than specific items within the previous letter.

Are you ready?
'''

_EXTRACTION = '''
Analyze the following clinical letter and extract all relevant information. Your response must be a valid JSON object matching the following specification exactly.

Required Format:
{
  "labels": {
    "exact_text": { # replace exact_text with the label. In the examples, one was "15-year-old".
      "context": "string with complete sentence/passage",
      "qualifier": "MUST BE ONE OF: Diagnoses, Procedures",
      "laterality": "MUST BE ONE OF: left, right, bilateral, NA",
      "presence": "MUST BE ONE OF: confirmed, suspected, resolved, negated, NA",
      "primary_secondary": "MUST BE ONE OF: primary, secondary, NA",
      "experiencer": "MUST BE ONE OF: Patient, Family, NA",
      "treatment_stage": "MUST BE ONE OF: pre-treatment, current, post-treatment, NA",
      "snomed_ct": "string of numbers or NA"
    }
    "exact_text": { # the rest of the labels
      // ... same structure as above
    }
  }
}

Critical Requirements:
1. Response must be a single JSON object
2. Include all labels you identify
3. All field names must be exactly as shown above
4. Every label must have ALL fields specified (use "NA" if not applicable)
5. No additional fields or nested objects are allowed
6. All string values must be properly escaped
7. Fields must use ONLY the enumerated values where specified
8. Context must contain complete sentences, properly escaped
9. Do not include any explanatory text before or after the JSON

Extract ONLY instances of:
- Diagnoses
- Procedures
Ignore all other categories (symptoms, signs, treatments, socio-demographics, risk factors, test results).

IMPORTANT: The earlier worked example was for illustration of the FORMAT only. Extract only diagnoses and procedures that actually appear in the clinical notes provided below — do not carry over any conditions, procedures or codes from the example.

Clinical Note:
'''

_SCOPING = '''
You have extracted the candidate diagnoses and procedures below from a single hospital admission's notes. Now act as a hospital clinical coder and decide which of them are the codeable diagnoses and procedures FOR THIS ADMISSION.

Select, choosing ONLY from the candidate ids listed:
- The PRINCIPAL diagnosis: the main condition treated or investigated during this admission.
- Up to 2 SECONDARY diagnoses: ONLY conditions that DIRECTLY affected this admission or its management — e.g. a condition that altered the anaesthetic, the surgery, the peri-operative care, or the length/course of stay. Do NOT include incidental past medical history that had no bearing on this admission (a comorbidity merely being present in the record is not enough).
- The PRINCIPAL procedure: the main operation or intervention performed this admission.
- Up to 2 SECONDARY procedures: ONLY additional procedures that were actually part of this admission's management (not operative sub-steps of the principal procedure).

Critical rules:
1. Determine the admission's purpose from the notes themselves (e.g. the operation performed, the stated reason for attendance) — not from any single note being flagged.
2. Some admissions have ONLY diagnoses or ONLY procedures. If there is no codeable diagnosis, set the principal diagnosis to null and secondary to []; likewise for procedures. NEVER invent a diagnosis or procedure to balance the case.
3. If you DO select any diagnoses, designate the single most central condition as the principal — do not leave the principal null while listing secondary diagnoses. The same applies to procedures. Return a null principal with an empty secondary list only when there are genuinely no codeable items of that type.
4. Do NOT code operative sub-steps (e.g. flap raised, edge freshened, graft placed, closure), anaesthetic or airway details, cannulation, packing, medications, or follow-up arrangements — code the principal intervention itself, not its components or peri-operative details.
5. Choose each item once; if the same condition or procedure was extracted more than once, pick a single id for it.
6. Select ONLY from the candidate ids below. Do not add new items or invent ids.

Candidate extractions (id | category | text | code):
'''

_SCOPING_NOTES = '''

Full admission notes (for context):
'''

_SCOPING_FORMAT = '''

Respond with ONLY this JSON object and nothing else:
{
  "diagnoses": {"principal": <id or null>, "secondary": [<id>, ...up to 2]},
  "procedures": {"principal": <id or null>, "secondary": [<id>, ...up to 2]}
}
'''

SNOMED_SELECTION_PROMPT = '''
You are tasked with selecting the most appropriate SNOMED CT code for a clinical annotation.

Original Annotation:
Text: {text}
Context: {context}
Qualifier: {qualifier}

Retrieved SNOMED CT codes:
{snomed_options}

Instructions:
1. Review the original annotation and the retrieved SNOMED CT codes carefully
2. Consider both the term and context when selecting a code
3. Select the most appropriate code, OR ask to search again with a better term, OR indicate none are suitable

Choose EXACTLY ONE of the following responses:
1. If a suitable code is listed: return just the code (e.g. "123456").
2. If none of the listed codes fit but a DIFFERENT search might find the right concept: return "SEARCH: <a better, concise, standard term for this concept>". Use this when the retrieved options are close but not right, or clearly about a different concept.
3. If there is genuinely no SNOMED CT code for this concept: return "NaN".

Reply with ONLY the code, or "SEARCH: <term>", or "NaN" — no other text.'''

SNOMED_REFINEMENT_PROMPT = '''The term "{text}" returned no results when searching SNOMED CT. This could be because:
1. There is no SNOMED CT code for this concept
2. The wording is too specific or complex for the database search

Please consider "{text}" and it context:

"{context}"

Please suggest a simpler or more standard medical term to search for in SNOMED CT. Consider:
- Use standard medical terminology
- Break compound terms into their core concept
- Retains the appropriate clinical meaning given the context
- Ensure the suggested term maintains the clinical meaning

Provide only the suggested search term with no additional explanation.'''

# Recovery prompts carry trailing spaces as sent, so they are built line by line.
_CONTINUATION = "\n".join([
    "",
    "        Your previous response was truncated. You have already successfully "
    "extracted {count} annotations, with the last complete entry being "
    '"{last_key}".',
    "",
    "        Please continue the analysis from where you left off and provide the "
    "remaining annotations in the same JSON format. ",
    "        ",
    '        Start with any annotations that come after "{last_key}" and continue '
    "until you have extracted all remaining relevant clinical information.",
    "        ",
    "        Return ONLY the JSON structure with the new annotations:",
    "        ",
    "        {{",
    '          "labels": {{',
    '            "next_annotation_term": {{',
    '              "context": "...",',
    '              "qualifier": "...",',
    '              "laterality": "...",',
    '              "presence": "...",',
    '              "primary_secondary": "...",',
    '              "experiencer": "...", ',
    '              "treatment_stage": "...",',
    '              "snomed_ct": "..."',
    "            }}",
    "          }}",
    "        }}",
    "        ",
])

_JSON_RETRY = "\n".join([
    "",
    "        The previous response was not in valid JSON format. ",
    "        Please reformat the following content as a valid JSON object ",
    "        following the specified structure exactly:",
    "",
    "        {response}",
    "        ",
])


def feedback_prompt(annotation_path):
    """Return the in-context feedback prompt showing the gold annotation.

    Args:
        annotation_path: Path to the gold annotation JSON for the example.

    Returns:
        The prompt text.
    """
    with open(annotation_path, encoding="utf-8") as f:
        annotation = json.dumps(json.load(f), indent=2)
    return _FEEDBACK.format(annotation=annotation)


def extraction_prompt(episode_text):
    """Return the extraction prompt for one episode.

    Args:
        episode_text: The formatted notes for the admission.

    Returns:
        The prompt text.
    """
    return (_EXTRACTION + f"\n\n{episode_text}\n\n"
            "Respond only with the JSON object following the specified format.\n")


def scoping_prompt(candidates, episode_text):
    """Return the prompt that scopes candidates to the admission's codes.

    Args:
        candidates: One line per candidate, as `[id] category | text | code`.
        episode_text: The formatted notes for the admission.

    Returns:
        The prompt text.
    """
    return _SCOPING + candidates + _SCOPING_NOTES + episode_text + _SCOPING_FORMAT


def continuation_prompt(count, last_key):
    """Return the prompt asking the model to resume a truncated response.

    Args:
        count: Number of complete labels already received.
        last_key: Text of the last complete label.

    Returns:
        The prompt text.
    """
    return _CONTINUATION.format(count=count, last_key=last_key)


def json_retry_prompt(response):
    """Return the prompt asking the model to reformat invalid JSON.

    Args:
        response: The response that failed to parse.

    Returns:
        The prompt text.
    """
    return _JSON_RETRY.format(response=response)
