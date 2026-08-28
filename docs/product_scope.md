# Product Scope

## One-line product definition

Multimodal Crawler is a governed public-science data pipeline for collecting, validating, preprocessing, curating, and
packaging multimodal web data into auditable dataset snapshots and training smoke/evaluation inputs.

## Ten-minute external-reader summary

This project is intended for data engineering and ML platform teams that need a controlled pipeline for public, approved
science sources. It may crawl allowlisted public hosts from NASA, NOAA, USGS, and similarly approved public-domain or
permissively licensed sources. It produces raw manifests, curated multimodal records, dataset snapshots, training
snapshots, baseline training/evaluation artifacts, and governance decisions. It refuses private networks, out-of-scope
domains, login-only content, paywalls, unknown-license material, personal data that cannot be redacted or quarantined,
aggressive crawling patterns, and any content that has not passed source-rules, lineage, quality, PII, dedupe, and
training-permission gates.

## Primary users

| User                        | Needs                                                    | Success signal                                                               |
|-----------------------------|----------------------------------------------------------|------------------------------------------------------------------------------|
| Data engineering team       | Safe, repeatable collection from approved public sources | Completed crawl manifest with no rules violations                           |
| ML platform team            | Reproducible curated/training snapshots                  | Versioned dataset snapshot and training smoke pass                           |
| Dataset governance reviewer | Explainable source, license, PII, and lineage decisions  | Auditable manifest and acceptance report                                     |
| Research engineer           | Fast local validation of multimodal sample plumbing      | Small `dev` run completes locally or against approved sources |
| Release engineer            | Clean source package without runtime/data artifacts      | Release artifact verifier passes                                             |

## Production-critical use cases

| Use case                          | Description                                                                                                          | Required maturity                      |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Governed public-source crawling   | Crawl only allowlisted public science sources under host, robots, budget, and source-rules constraints              | Stable                                 |
| Raw artifact lineage              | Persist every accepted payload with source URL, fetch metadata, content classification, hash, and manifest reference | Stable                                 |
| Curation into dataset snapshots   | Convert raw records into validated records for downstream ML processing                                              | Stable                                 |
| Training snapshot assembly        | Produce training-ready records only after source, license, PII, dedupe, quality, lineage, and quota gates pass       | Stable                                 |
| Baseline training smoke path      | Validate dataloader, task routing, checkpointing, metrics, and acceptance plumbing                                   | Stable as smoke; not a quality claim   |
| Multimodal media preprocessing    | Normalize image/document/audio/video inputs where optional backends are installed                                    | Beta                                   |
| Representative model evaluation | Evaluate model quality on labeled task datasets and evaluation thresholds                                             | Beta until evaluation coverage is complete |

## Non-goals

The project does not attempt to be:

- a general internet crawler;
- a search engine crawler;
- a scraper for arbitrary commercial, private, login-only, or paywalled sites;
- a browser automation framework;
- a personal-data collection system;
- a legal-rights determination engine independent of human-approved source rules;
- proof of a strong production-quality multimodal model;
- a substitute for task-specific evaluation;
- a real-time inference serving platform;
- a tool for bypassing robots.txt, rate limits, terms of use, authentication, or access controls.

## Allowed data scope

Allowed data is restricted to content that satisfies all of the following:

1. The source appears in `config/files/sources/source_registry.json` or an equivalent reviewed registry.
2. The source host and asset hosts are explicitly allowlisted for the active environment.
3. The URL passes scheme, host, DNS, private-IP, redirect, robots, rate-limit, and crawl-budget gates.
4. The source has a documented copyright/license basis.
5. The record passes PII, dedupe, quality, modality, lineage, and storage-integrity checks.
6. The record is separately marked as training-allowed before it can enter a training snapshot.

Initial approved source families are:

| Source family | Intended content                                                      | Default status                                                |
|---------------|-----------------------------------------------------------------------|---------------------------------------------------------------|
| NASA          | Public science publications, news, images, media pages, and documents | Approved with third-party exceptions handled by source rules |
| NOAA          | Public ocean, weather, climate, education, and library content        | Approved with third-party exceptions handled by source rules |
| USGS          | Public publications, science pages, media assets, and reports         | Approved with third-party exceptions handled by source rules |

## Explicitly disallowed data scope

The crawler must reject or quarantine:

- private, loopback, link-local, multicast, reserved, carrier-grade NAT, and cloud metadata network targets;
- non-HTTP(S) schemes unless a future rules explicitly allows them;
- URLs discovered outside active source scope;
- redirect chains that leave approved scope or resolve to blocked networks;
- login-required, authenticated, tokenized, session-only, or paywalled content;
- content whose license or training permission is unknown;
- third-party embedded assets that are not independently approved;
- personal data unless it is redacted or the record is quarantined before training;
- malformed or unsupported media payloads;
- payloads that exceed configured size, duration, depth, host, or run budgets;
- any content blocked by robots or active source rules.

## Training architecture boundary

[ADR-0002](adr/0002-scratch-only-model-and-preprocessing-boundary.md) is the
normative schema for the boundary between data extraction and model training.
External OCR, ASR, and diarization backends are permitted only in
`preprocessing/`. Their outputs are derived dataset observations with complete
provenance, not an online connection to training. External embeddings, hidden
states, teacher or distillation signals, remote calls, and external encoder
weights are outside product scope for the training path and release checkpoint.
The only top-level trainable model family is the repository-owned, randomly
initialized `MultimodalModel`, improved in place.

Production model cards use the
[model card template](../training/export/model_card_template.md) to record this
boundary and its release evidence.

## Product outputs

| Output                       | Path family                             | Consumer                                 |
|------------------------------|-----------------------------------------|------------------------------------------|
| Raw crawl objects            | `data/raw/runs/`                        | Curation pipeline                        |
| Raw crawl manifests          | `data/raw/manifests/`                   | Governance, debugging, acceptance checks |
| Curated snapshots            | `data/curated/`                         | Dataset assembly                         |
| Training snapshots           | `data/interim/training_sets/`           | Training smoke/evaluation                |
| Augmented training snapshots | `data/interim/augmented_training_sets/` | Optional training experiments            |
| Candidate model artifacts    | `artifacts/candidates/<campaign-id>/seed-<seed>/` | Acceptance and promotion gate |
| Promoted model releases      | `models/releases/` with `models/current.json` | Downstream consumers          |
| Workflow artifact registry   | `data/registry/workflow_artifacts/`     | Acceptance and lineage auditing          |
| Training artifacts           | `runtime/training/`, `runtime/logs/`    | Smoke/eval diagnostics                   |

Runtime outputs are not source release artifacts and must not be included in release packages.

## Output quality requirements

A run is successful only when all required manifest and workflow gates pass.

| Area                 | Minimum requirement for stable path                                                     |
|----------------------|-----------------------------------------------------------------------------------------|
| Crawl completion     | Final manifests exist and mark the run `completed` and `final`                          |
| Storage integrity    | Every manifest record points to an existing artifact with matching hash/size metadata   |
| Source governance    | Every record has an approved source rules decision                                     |
| Training eligibility | Training snapshots contain only records with `training_allowed=true`                    |
| PII handling         | PII records are redacted, rejected, or quarantined before training                      |
| Dedupe               | Duplicate and near-duplicate decisions are recorded                                     |
| Evaluation           | Baseline smoke/evaluation completes and writes metrics                                  |
| Reproducibility      | Environment, config fingerprint, source registry version, and snapshot IDs are recorded |

## Product maturity stance

| Component                          | Product status                  | Production claim                                                                     |
|------------------------------------|---------------------------------|--------------------------------------------------------------------------------------|
| URL governance and crawl admission | Stable target                   | Can enforce public allowlisted crawl scope when validation and acceptance gates are green |
| Raw/curated/training manifests     | Stable target                   | Can provide auditable lineage when finalization passes                               |
| Multimodal preprocessing           | Beta                            | Depends on installed optional media backends and per-modality validation             |
| Baseline `pipeline_smoke` training    | Stable smoke path               | Validates plumbing only; does not prove model quality                                |
| Model-quality evaluation         | Beta/experimental               | Requires labeled datasets, thresholds, and reproducible evaluation reports            |
| Release package                    | Must be clean before production | Runtime/data artifacts are forbidden in source releases                              |

## Definition of done for this scope document

This scope is complete when:

- every product-critical use case maps to an acceptance check or release gate;
- every non-goal has a corresponding rejection, quarantine, or documentation path;
- all approved sources have source-rules entries;
- source releases exclude runtime/data artifacts;
- a new contributor can answer within ten minutes: who this is for, what it may crawl, what it produces, and what it
  refuses.
