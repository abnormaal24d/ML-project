# MultimodalModel release model card template

Use this template for every candidate and production release. Replace all angle-
bracket placeholders; do not remove required sections when a value is unknown.
Record an explicit `not available` plus the blocking gate instead.

## Release identity

- Release ID: `<release-id>`
- Release stage: `<candidate|production>`
- Code fingerprint: `<sha256>`
- Configuration fingerprint: `<sha256>`
- Finalized dataset snapshot ID: `<snapshot-id>`
- Tokenizer fingerprint: `<sha256>`
- Random initialization seed/schema fingerprint: `<sha256>`
- Initial state fingerprint before optimizer step one: `<sha256>`
- Final checkpoint fingerprint: `<sha256>`

## Architecture boundary (required)

Normative decision: [ADR-0002: Scratch-only MultimodalModel and external models
strictly limited to preprocessing](../../docs/adr/0002-scratch-only-model-and-preprocessing-boundary.md).

- Top-level trainable class: `MultimodalModel` (the sole trainable model family).
- Initialization evidence: `<artifact or acceptance-report reference>`.
- Coherent checkpoint namespace evidence: `<artifact or acceptance-report reference>`.
- Offline training/evaluation evidence: `<artifact or acceptance-report reference>`.
- Boundary/dependency/bundle scan evidence: `<artifact or acceptance-report reference>`.
- External preprocessing backends used: `<name, version, revision, content hash>`.
- Derived observations and provenance: `<manifest/lineage reference>`.

External preprocessing output is dataset data, not an online model connection.
The checkpoint and effective training path must contain no external encoder
weights, embeddings, hidden states, teacher or distillation signals, remote model
calls, preprocessing caches, or external production judge.

## Intended use

`<supported users, tasks, modalities, and operating context>`

## Out-of-scope and disabled capabilities

`<unsupported tasks/modalities, disabled experimental routes, and prohibited uses>`

## Training data and provenance

- Source-rules evidence: `<reference>`
- Payload and lineage reconciliation: `<reference>`
- Preprocessing producer provenance: `<reference>`
- Split and leakage evidence: `<reference>`

## Evaluation and acceptance

- Representative evaluation datasets and thresholds: `<reference>`
- Slice metrics and uncertainty: `<reference>`
- Exact-resume evidence: `<reference>`
- Security/compliance/SBOM gates: `<reference>`
- Production acceptance decision: `<reference or not available>`

An external model or judge may not make the final production acceptance
decision.

## Limitations and risks

`<known failure modes, dataset limits, modality gaps, fairness/safety risks, and rollback trigger>`

## Release decision

- Decision: `<rejected|candidate|production_model>`
- Decision owner: `<role/name>`
- Decision timestamp: `<UTC timestamp>`
- Missing or waived gates: `<none, or explicit non-production blockers>`
