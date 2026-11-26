
# Beyond the Context Window: A Universal Personal Memory System for LLM Assistants

## UPMS: Structured Annotation and Human Verification

UPMS (Universal Personal Memory System) is a local-first external memory system for LLM assistants. Its shared-encoder multi-task Transformer performs local structured extraction to support memory construction, temporal memory management and query-aware retrieval. GPT-5 is used for offline supervision generation rather than routine memory construction.

This repository provides the structured annotation procedure, human verification guidelines, annotation schema, representative examples and annotation-quality results associated with UPMS. We use the [Multi-Session Chat (MSC) dataset](https://aclanthology.org/2022.acl-long.356/) (Xu et al., 2022) as the source dataset for extractor training and evaluation.

## Publication Status


The accompanying paper, **Beyond the Context Window: A Universal Personal Memory System for LLM Assistants**, has been submitted to the **International Conference on Database Systems for Advanced Applications (DASFAA 2027)** and is currently under review.

This repository currently provides supporting research materials associated with the structured annotation procedure and human verification study. The complete UPMS implementation and experimental resources are planned for public release upon acceptance of the paper.

## Table of Contents

- [1. Annotation Procedure](#1-annotation-procedure)
- [2. Human Verification Guidelines](#2-human-verification-guidelines)
- [3. Dataset and Sampling](#3-dataset-and-sampling)
- [4. Annotation Schema](#4-annotation-schema)
- [5. Representative Annotation Examples](#5-representative-annotation-examples)
- [6. Evaluation Metrics](#6-evaluation-metrics)
- [7. Human Verification Results](#7-human-verification-results)
- [8. GPT-5 Annotation Prompt](#8-gpt-5-annotation-prompt)
- [9. Dataset Reference](#9-dataset-reference)

## 1. Annotation Procedure

The reference annotation pipeline comprises three stages: automatic annotation, normalisation and preprocessing.

### 1.1. Automatic Annotation

GPT-5 generates structured annotations for conversational utterances using the following fields:

- `clean_text`: original conversational utterance.
- `is_question`: question detection.
- `temporal_info`: temporal information, including `datetime` and `recurrent_slots`.
- `intents`: applicable leaf-level intent categories.
- `reference_persons`: referenced persons and their relationships.
- `action_attribute`: verbatim action and attribute phrases.

These fields constitute an intermediate annotation representation from which supervision for the seven extraction tasks is derived.

Automatic annotation identifies conversational information grounded in the original utterance according to the predefined annotation schema. Relative temporal expressions are preserved rather than converted into absolute dates during this stage.

### 1.2. Normalisation

GPT-5-generated annotations are normalised to produce consistent task-specific representations. Normalisation includes:

- Canonicalising temporal expressions while retaining their original text spans.
- Mapping leaf-level intents to coarse intent categories.
- Standardising referenced-person categories and identifying eligible person-name spans.
- Separating action and attribute spans into their corresponding types.
- Filtering duplicate or ungrounded annotations.

Normalisation provides consistent representations for supervised learning while maintaining the distinction between semantic classification labels and text-grounded span annotations.

### 1.3. Preprocessing

Normalised annotations are converted into four sequence-level classification targets and three token-level extraction targets using typed Beginning-Inside-Outside (BIO) labels.

The sequence-level tasks identify questions, temporal information, applicable intent categories and referenced-person categories. The token-level tasks identify temporal expressions, action/attribute phrases and person-name spans.

Temporal detection is derived from temporal fields or grounded temporal spans. Token-level supervision is constructed using eligible spans grounded in the current utterance.

The resulting supervision is used to jointly train the shared-encoder multi-task Transformer for all seven extraction tasks.

### 1.4. Conversational Context

The reference annotation pipeline applies the following context-handling rules:

- GPT-5 interprets each utterance independently during automatic annotation.
- Person normalisation can use participant metadata and preceding references within the same conversational session.
- The extractor implementation supports preceding-turn context through a `[CUR]` separator.
- Token-level span annotations remain grounded in the current utterance.

Automatic annotation, contextual normalisation and extractor input construction serve distinct purposes. Automatic annotation identifies information expressed in individual utterances, while contextual normalisation supports consistent person attribution. Preceding-turn context provides additional linguistic information to the extractor without changing the current-utterance span grounding requirement.

## 2. Human Verification Guidelines

### 2.1. Purpose of Human Verification

The objective of human verification is to assess the quality of GPT-5-derived supervision across the seven extraction tasks. Evaluators assess the correctness and completeness of the normalised annotations according to the predefined annotation schema.

The verification study assesses the reference supervision used for extractor training and evaluation, rather than predictions generated by the trained multi-task Transformer.

### 2.2. Evaluator Background and Preparation

Three human evaluators, each with over five years of professional experience in data science and artificial intelligence, participated in the reported human verification study.

Before evaluating the sampled utterances, the evaluators completed a separate calibration exercise using the predefined annotation guidelines and validity criteria.

### 2.3. Verification Procedure

Each sampled utterance was independently evaluated by all three evaluators using the following procedure:

1. Examine the utterance and relevant conversational context before viewing the annotations to mitigate anchoring bias.
2. Review the human-readable, normalised GPT-5-derived annotations corresponding to all seven extraction tasks.
3. Assess the correctness and completeness of each task annotation according to the predefined annotation schema.
4. Record one binary validity judgement for each extraction task.

The evaluators assessed the classification labels, span types and span boundaries, including the derived temporal-detection label, rather than their numerical encodings.

Relevant conversational context supports the interpretation of contextual information and referenced-person attribution. The validity of token-level spans is assessed against the current utterance.

Evaluators did not correct, modify or replace any annotation values. Final validity decisions were determined by majority voting, with inter-evaluator agreement measured before voting.

### 2.4. Scoring Criteria

Each extraction task is evaluated using a binary scoring scale.

| Score | Interpretation |
|---|---|
| 1 (Valid) | The annotation is correct and complete according to the predefined annotation schema. |
| 0 (Invalid) | The annotation contains an incorrect, unsupported or missing eligible label or span, or an incorrect span type or boundary. |

The following criteria apply:

- Binary classification labels must be correct.
- Multi-label classification annotations must contain all applicable categories without unsupported categories.
- Span annotations must contain eligible spans with correct types and boundaries according to the annotation schema.
- Empty annotations are valid when no eligible label or span is missing.
- No partial credit is assigned.

Correctness and completeness are assessed within the representational scope of the corresponding extraction task.

One validity judgement is recorded per extraction task.

## 3. Dataset and Sampling

The Multi-Session Chat (MSC) dataset contains conversations between two speakers across multiple conversational sessions.

The extractor training and evaluation use the reported MSC training, validation and test partitions. For human verification, stratified random sampling was used to select 1,000 utterances across the three partitions.

| Dataset partition | Utterances | Human verification sample |
|---|---:|---:|
| Training | 236,987 | 500 |
| Validation | 31,456 | 250 |
| Test | 30,382 | 250 |
| **Total** | **298,825** | **1,000** |

The verification sample provides coverage across all three dataset partitions. Pooled inter-evaluator agreement describes the sampled utterances rather than a corpus-proportional sample.

## 4. Annotation Schema

### 4.1. Extraction Tasks

The shared-encoder multi-task Transformer jointly performs four sequence-level classification tasks and three token-level extraction tasks.

The following table defines the seven extraction tasks and their corresponding verification criteria.

| Extraction task | Task type | Annotation and verification criteria |
|---|---|---|
| Question detection | Binary classification | Identifies whether the utterance contains a question. A positive label applies when the utterance ends in `?` or is phrased as a question seeking information. |
| Temporal detection | Binary classification | Identifies temporal information from temporal fields or grounded temporal spans. Grammatical tense alone does not constitute an explicit temporal expression. |
| Intent classification | Multi-label classification | Identifies all applicable memory-relevant intent categories supported by the utterance. Missing or unsupported categories are invalid. |
| Referenced-person classification | Multi-label classification | Identifies person categories associated with memory-relevant information or fact-seeking questions. References must be attributed to the appropriate person. |
| Temporal-span extraction | Token-level extraction | Identifies eligible grounded temporal expressions (`DATE`) and recurring expressions (`RECUR`) with correct types and boundaries. |
| Action/attribute-span extraction | Token-level extraction | Identifies eligible action (`ACT`) and attribute (`ATTR`) phrases grounded in the utterance, following annotation normalisation. |
| Person-name-span extraction | Token-level extraction | Identifies eligible named-person and named-pet references (`PER`). Pronouns and generic relationship terms are excluded. |

Temporal detection and temporal-span extraction represent distinct prediction tasks. Temporal detection identifies the presence of temporal information, whereas temporal-span extraction identifies eligible expressions grounded in the current utterance.

When a temporal annotation cannot be reliably aligned with the original utterance, the ungrounded expression is excluded from token-level supervision. The temporal-detection label is derived separately from the temporal fields or grounded spans.

### 4.2. Intent Categories

The reference annotation schema groups leaf-level intent categories into broader categories for intent classification. The classifier inventory is determined from the processed dataset.

| Coarse category | Leaf categories | Description |
|---|---|---|
| `identity` | `identity_information`, `identity_relationship` | Personal identity and relationships. |
| `health` | `health_condition`, `health_treatment`, `health_other` | Health conditions, treatments and other health information. |
| `personal_fact` | `personal_fact_family`, `personal_fact_education`, `personal_fact_work`, `personal_fact_location`, `personal_fact_possession`, `personal_fact_financial`, `personal_fact_other` | Personal facts concerning family, education, work, location, possessions and finances. |
| `preference` | `preference_food`, `preference_hobby`, `preference_interest`, `preference_habit`, `preference_other` | Preferences, interests, likes, dislikes and habits. |
| `goal` | `goal_completed`, `goal_plan`, `goal_other` | Completed goals, future plans and intentions. |
| `experience` | `experience_event`, `experience_activity`, `experience_achievement`, `experience_other` | Personal events, activities and achievements. |
| `emotion` | `emotion_state`, `emotion_preference` | Emotional states and reactions. |
| `belief` | `belief_opinion`, `belief_other` | Opinions, beliefs and values. |
| `procedure` | `procedure_routine`, `procedure_instruction` | Routines, procedures and instructions. |
| `skill` | `skill_information` | Skills, abilities and proficiency. |
| `general` | `general_facts`, `general_other` | Other factual information, acknowledgements and general conversational content. |

Intent categories are not mutually exclusive. An utterance may receive multiple intent labels when its content satisfies multiple category definitions. Semantic overlap between categories is therefore accommodated through multi-label classification.

Specific leaf categories are preferred over their corresponding `_other` alternatives when applicable. Categories must be supported by the utterance rather than inferred from unsupported assumptions.

Pure information requests receive `general_other` rather than an intent category describing the information being requested.

### 4.3. Referenced-Person Categories

Referenced-person classification identifies the individuals associated with the information expressed or requested in an utterance.

Representative categories include:

| Category | Description |
|---|---|
| `me` | The speaker of the current utterance. |
| `interlocutor` | The other conversational participant. |
| `father` | The referenced father. |
| `mother` | The referenced mother. |
| `other_person` | Other eligible person references. |

The normalised category `self` is mapped to `me` for classifier supervision. Additional relationship and pet categories are supported by the reference schema.

Referenced-person classification and person-name-span extraction provide complementary representations. The former identifies normalised person categories, whereas the latter identifies eligible names explicitly occurring in the utterance.

Personal names are represented separately through person-name-span extraction rather than as individual classification categories. Generic relationship terms and pronouns may contribute to referenced-person classification without constituting person-name spans.

### 4.4. Typed BIO Labels

The three token-level extraction tasks use typed Beginning-Inside-Outside (BIO) labels.

| Extraction task | BIO labels |
|---|---|
| Temporal-span extraction | `O`, `B-DATE`, `I-DATE`, `B-RECUR`, `I-RECUR` |
| Action/attribute-span extraction | `O`, `B-ACT`, `I-ACT`, `B-ATTR`, `I-ATTR` |
| Person-name-span extraction | `O`, `B-PER`, `I-PER` |

- `B`: beginning of a span.
- `I`: continuation of a span of the same type.
- `O`: token outside an annotated span.

All extracted spans must be grounded in the current utterance.

The corresponding span types are defined as follows:

| Span type | Definition | Representative examples |
|---|---|---|
| `DATE` | Explicit dates, times, durations and relative temporal expressions. | `three years ago`, `next Friday`, `the morning` |
| `RECUR` | Explicit recurring temporal expressions. | `every Monday`, `weekly`, `every morning` |
| `ACT` | Eligible phrases describing actions, activities or events. | `taking a break`, `working in the fields` |
| `ATTR` | Eligible phrases describing properties, characteristics or requested information. | `comic books`, `birthday`, `blood type` |
| `PER` | Eligible personal names and named-pet references. | `Sarah`, `Adam` |

Action and attribute phrases are initially represented in the combined `action_attribute` field. During normalisation, eligible phrases are separated into `ACT` and `ATTR` span types before BIO label construction.

These typed representations distinguish conversational activities from attributes that may support structured indexing and query interpretation.

### 4.5. Temporal Annotation Scope

The intermediate temporal annotation schema distinguishes non-recurring temporal information from recurring temporal information through two fields:

- `datetime`: an explicit date, time, duration or relative temporal expression.
- `recurrent_slots`: an explicit recurring temporal expression.

Each field accommodates a single expression or a null value. An utterance containing both non-recurring and recurring temporal information can therefore contain values in both fields.

When an utterance contains multiple expressions of the same temporal type, the intermediate representation does not enumerate an unrestricted list of expressions. Temporal-span supervision is derived from the eligible grounded annotations available within this schema.

The temporal-detection task is distinct from the number of annotated temporal spans. It identifies the presence of temporal information rather than counting individual expressions.

Relative temporal expressions are preserved during annotation. Their interpretation against conversational timestamps is performed by the downstream temporal processing mechanism rather than by GPT-5 during offline supervision generation.

This annotation schema provides structured temporal signals for the memory system while preserving the original conversational utterance as supporting evidence.

## 5. Representative Annotation Examples

The following examples illustrate the human-readable supervision corresponding to the seven extraction tasks.

Examples 1 and 2 use utterances from the MSC dataset. Example 3 is an additional constructed utterance illustrating recurring temporal expressions and person-name spans.

The annotations are illustrative representations of the predefined schema rather than original GPT-5 outputs or recorded human judgements.

### 5.1. Example 1: Activity and Temporal Reference

```json
{
  "utterance": "I'm alright, just taking a break. Spent the morning working in the fields with dad.",
  "task_annotations": {
    "is_question": false,
    "has_temporal": true,
    "intents": [
      "emotion",
      "experience"
    ],
    "reference_persons": [
      "me",
      "father"
    ],
    "temporal_spans": [
      {
        "text": "the morning",
        "type": "DATE"
      }
    ],
    "action_attribute_spans": [
      {
        "text": "taking a break",
        "type": "ACT"
      },
      {
        "text": "working in the fields",
        "type": "ACT"
      }
    ],
    "person_name_spans": []
  }
}
```

The leaf-level intent categories `emotion_state` and `experience_activity` are mapped to `emotion` and `experience`, respectively.

The relationship term `dad` is normalised to `father` and is not retained as a person-name span.

The explicit temporal expression `the morning` is represented as a `DATE` span, while the conversational activities are represented as `ACT` spans.

### 5.2. Example 2: Information Request

```json
{
  "utterance": "What kind of comic books do you like?",
  "task_annotations": {
    "is_question": true,
    "has_temporal": false,
    "intents": [
      "general"
    ],
    "reference_persons": [
      "interlocutor"
    ],
    "temporal_spans": [],
    "action_attribute_spans": [
      {
        "text": "comic books",
        "type": "ATTR"
      }
    ],
    "person_name_spans": []
  }
}
```

The utterance is a pure information request and therefore receives the leaf-level intent category `general_other`, which is mapped to the coarse category `general`.

The referenced person is the interlocutor, and `comic books` is extracted as the requested attribute.

The utterance does not independently state a preference and therefore does not receive a preference intent label.

Source: [MSC dataset](https://huggingface.co/datasets/nayohan/multi_session_chat), dialogue ID 25, session ID 1 (zero-based).

### 5.3. Example 3: Recurring Activity and Named-Person Reference

The following constructed example focuses on recurring temporal expressions and person-name-span extraction.

```json
{
  "utterance": "My friend Sarah and I play tennis every Monday.",
  "illustrative_task_annotations": {
    "has_temporal": true,
    "temporal_spans": [
      {
        "text": "every Monday",
        "type": "RECUR"
      }
    ],
    "action_attribute_spans": [
      {
        "text": "play tennis",
        "type": "ACT"
      }
    ],
    "person_name_spans": [
      {
        "text": "Sarah",
        "type": "PER"
      }
    ]
  }
}
```

The recurring expression `every Monday` is identified as a `RECUR` span, while `play tennis` represents an action span.

The name `Sarah` is retained as a person-name span. The generic relationship term `friend` can contribute to referenced-person classification but does not constitute a person-name span.

This example illustrates the complementary nature of temporal, action and person-name extraction. It presents the relevant task annotations rather than a complete annotation record.

### 5.4. Examples of Valid and Invalid Annotations

The following illustrative examples demonstrate how the binary validity criteria apply to individual extraction tasks.

| Extraction task | Utterance | Valid (1) | Invalid (0) | Reason for invalidity |
|---|---|---|---|---|
| Question detection | What kind of comic books do you like? | `true` | `false` | The question is incorrectly classified. |
| Temporal detection | I moved here three years ago. | `true` | `false` | Explicit temporal information is missed. |
| Intent classification | I'm alright, just taking a break. Spent the morning working in the fields with dad. | `emotion`, `experience` | `emotion` | An applicable intent category is missing. |
| Referenced-person classification | What kind of comic books do you like? | `interlocutor` | `me` | The referenced person is incorrectly attributed. |
| Temporal-span extraction | I moved here three years ago. | `DATE: three years ago` | `DATE: years ago` | The temporal span boundary is incomplete. |
| Temporal-span extraction | I play tennis every Monday. | `RECUR: every Monday` | `DATE: every Monday` | The temporal span type is incorrect. |
| Action/attribute-span extraction | I'm alright, just taking a break. Spent the morning working in the fields with dad. | `ACT: taking a break`, `ACT: working in the fields` | `ACT: taking a break` | An applicable action span is missing. |
| Action/attribute-span extraction | What's your mom's blood type? | `ATTR: blood type` | `ACT: blood type` | The span is incorrectly classified as an action. |
| Person-name-span extraction | My father is visiting. | Empty | `PER: father` | A generic relationship term is incorrectly extracted as a person name. |
| Person-name-span extraction | My friend Sarah is visiting. | `PER: Sarah` | Empty | An eligible person-name span is missing. |

These examples illustrate that annotation validity depends on semantic correctness, completeness within the annotation schema, span grounding, span type and span boundaries.

## 6. Evaluation Metrics

### 6.1. Human Verification Metrics

Human verification measures the quality of GPT-5-derived supervision using the following metrics.

| Metric | Definition |
|---|---|
| Fleiss' κ | Chance-adjusted inter-evaluator agreement calculated from the original binary judgements of the three evaluators before majority voting. |
| Pairwise agreement (PA) | Percentage of agreeing evaluator pairs, averaged across the sampled utterances for each extraction task. |
| Task validity | Percentage of sampled utterances receiving a valid majority decision for the corresponding extraction task. |
| Joint validity | Percentage of sampled utterances receiving valid majority decisions for all seven extraction tasks. |

Fleiss' κ and pairwise agreement are calculated across the 1,000 sampled utterances for each task.

Task validity and joint validity are reported separately for the training, validation and test partitions.

Task validity assesses the quality of individual extraction-task annotations, whereas joint validity assesses whether all seven task annotations associated with an utterance satisfy the verification criteria.

### 6.2. Extractor Evaluation Metrics

The extractor is evaluated using held-out GPT-5-derived reference labels.

| Extraction task type | Evaluation metric |
|---|---|
| Binary classification | F1 |
| Multi-label classification | Micro-F1 |
| Token-level span extraction | Partial-overlap F1 |

The reference implementation computes partial-overlap F1 using lowercased, whitespace-normalised surface strings and a similarity threshold of 0.5.

When one span contains another, similarity is calculated using the ratio of their character lengths. Otherwise, character-set Jaccard similarity is used.

Predicted and reference spans are greedily matched one-to-one in descending similarity order within each utterance.

Matched and unmatched spans are aggregated to calculate micro precision, recall and F1.

The reported partial-overlap F1 evaluates surface-span recovery and does not require matching span types. This metric provides partial matching credit when predicted spans overlap sufficiently with their reference spans.

Human verification applies a distinct, stricter validity criterion that requires correctness and completeness of classification labels, span types and span boundaries.

Consequently, human verification assesses the quality of the reference supervision, whereas extractor evaluation measures predictive performance against held-out reference labels.

## 7. Human Verification Results

The following table presents the human verification results reported in the accompanying manuscript.

| Extraction task | Fleiss' κ | PA | Training (n=500) | Validation (n=250) | Test (n=250) |
|---|---:|---:|---:|---:|---:|
| Question detection | 0.90 | 99.6% | 98.2% | 98.0% | 97.6% |
| Temporal detection | 0.87 | 99.2% | 97.0% | 96.8% | 96.4% |
| Temporal-span extraction | 0.87 | 99.0% | 96.6% | 96.0% | 95.6% |
| Intent classification | 0.84 | 98.7% | 96.4% | 96.0% | 95.6% |
| Referenced persons | 0.85 | 99.1% | 97.0% | 96.8% | 96.4% |
| Action/attribute spans | 0.83 | 98.5% | 95.8% | 95.6% | 95.2% |
| Person-name spans | 0.90 | 99.5% | 98.0% | 97.6% | 97.2% |
| **All seven tasks valid** | N/A | N/A | **84.4%** | **83.2%** | **82.4%** |

Fleiss' κ ranges from 0.83 to 0.90, while pairwise agreement ranges from 98.5% to 99.6%.

Task validity ranges from 95.2% to 98.2% across the dataset partitions, with joint validity ranging from 82.4% to 84.4%.

Joint validity applies a stricter utterance-level criterion by requiring a valid majority decision for every extraction task.

The results provide evidence of consistent human judgements and high task-level validity of the GPT-5-derived supervision across the sampled utterances. Extractor F1 separately measures predictive agreement with held-out GPT-5-derived reference labels.

## 8. GPT-5 Annotation Prompt (Condensed Version)

For readability, we provide a condensed version of the
GPT-5 annotation prompt, retaining its core instructions,
annotation criteria and structured output format. The
complete prompt templates will be publicly released
alongside the UPMS implementation and experimental
resources upon acceptance of the paper.

### 8.1. Annotation Objective

Extract structured, memory-relevant information from
conversational utterances to support supervised multi-task
learning, temporal memory management, person associations,
structured indexing and query-aware retrieval.

Each utterance is annotated independently according to
the predefined annotation schema.

### 8.2. Annotation Fields

The prompt instructs GPT-5 to generate six intermediate
annotation fields for each utterance.

| Field | Data type | Annotation instruction |
|---|---|---|
| `clean_text` | String | Preserve the original conversational utterance without modification. |
| `is_question` | Boolean | Identify whether the utterance contains a question. |
| `temporal_info` | Object | Identify explicit temporal and recurring expressions using `datetime` and `recurrent_slots`, respectively. |
| `intents` | Array of strings | Assign applicable leaf-level intent categories from the predefined taxonomy in Section 4.2. |
| `reference_persons` | Array of strings | Identify persons associated with stated or requested information, including speaker references, relationships and explicit names. |
| `action_attribute` | Array of strings | Extract relevant action and attribute phrases grounded in the original utterance. |

### 8.3. Annotation Instructions

The annotation prompt incorporates the following principles:

- **Text preservation:** Retain the original utterance
  without paraphrasing or introducing additional information.

- **Temporal grounding:** Extract explicit temporal
  expressions according to the annotation schema in
  Section 4.5. Preserve relative expressions for
  downstream temporal resolution.

- **Multi-label intent classification:** Assign applicable
  leaf-level intent categories supported by the utterance.
  Prefer specific categories over general alternatives.

- **Person attribution:** Identify relevant person
  references using predefined relationship conventions
  and explicitly mentioned names.

- **Action and attribute grounding:** Extract meaningful
  phrases occurring verbatim in the utterance, including
  attributes requested by fact-seeking questions.

- **Question handling:** Distinguish fact-seeking questions
  from general conversational questions. Pure information
  requests receive the `general_other` intent category.

- **Annotation quality:** Maintain correctness,
  completeness within the annotation schema, textual
  grounding and consistency across utterances. Avoid
  unsupported inference and unnecessary duplication.

### 8.4. Structured Output Format

The prompt requires a valid JSON object containing an
array of annotation records in the original input order.
Each record contains exactly six intermediate annotation
fields.

The following example illustrates the expected output
structure.

**Input utterance:**

"I spent the morning working in the fields with dad."

**Illustrative output:**

```json
{
  "results": [
    {
      "clean_text": "I spent the morning working in the fields with dad.",
      "temporal_info": {
        "datetime": "the morning",
        "recurrent_slots": null
      },
      "is_question": false,
      "intents": [
        "experience_activity"
      ],
      "reference_persons": [
        "me",
        "father"
      ],
      "action_attribute": [
        "working in the fields"
      ]
    }
  ]
}
```

### 8.5. Supervision Construction

The intermediate GPT-5 annotations undergo normalisation
and preprocessing to construct the seven task-specific
supervision targets described in Section 4. These comprise
four sequence-level classification tasks and three
token-level extraction tasks.

GPT-5 is used exclusively for offline supervision
generation in this annotation pipeline. Routine structured
extraction within UPMS is performed locally by the trained
shared-encoder multi-task Transformer.

## 9. Dataset Reference

Xu, J., Szlam, A., and Weston, J. (2022). **Beyond Goldfish Memory: Long-Term Open-Domain Conversation.** Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5180–5197.

