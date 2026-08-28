# Code Architecture 100/100 Score Gate

This score gate defines what **100/100 code architecture** means for the
multimodal crawler. The goal is not architectural purity for its own sake; the
goal is deterministic domain logic, safe adapter boundaries, and CI-enforced
regression prevention.

## 100/100 Definition

The architecture score is 100/100 only when all of the following are true:

1. Pure domain modules do not import concrete infrastructure packages.
2. Domain decisions, reproducibility-significant timestamps, and semantic
   identifiers receive explicit values; composition obtains those values from
   `Clock` and `IdGenerator`. Elapsed-time logic uses monotonic time, while
   cryptographic or operational entropy for temporary paths, locks, atomic
   publication, and collision avoidance remains infrastructure-owned.
   `shared/runtime_primitives.py` is the sole owner of those two application-wide
   contracts; packages do not duplicate or re-export them.
3. Filesystem, manifest, artifact, media, and network IO are reached through
   ports/adapters, not hidden inside domain logic.
4. Fetch/network/media backends live in infrastructure or composition layers.
5. Orchestration modules wire dependencies and run phases; they do not own
   business decisions.
6. Fragmentation is controlled: new tiny modules are either merged or justified.
7. CI runs all architecture guardrails on every pull request.

## Required Guardrail Commands

```bash
ruff check --no-cache .
ruff format --check .
mypy --config-file pyproject.toml .
python -m compileall -q -j 0 .
```

## Layer Rules

| Layer            | Allowed responsibility                            | Forbidden responsibility                           |
|------------------|---------------------------------------------------|----------------------------------------------------|
| Domain           | Pure decisions, parsing, rules, validation       | Network calls, media libraries, direct file writes |
| Storage/adapters | Filesystem, manifests, JSONL, atomic writes       | Product/business rules                            |
| Fetch adapters   | HTTP sessions, DNS, response bodies               | Source training eligibility decisions              |
| Media adapters   | Pillow/OpenCV/PyAV/OCR/ASR backends               | Dataset governance decisions                       |
| Orchestration    | Wire components, execute phases, handle lifecycle | Inline crawl business logic                        |
| Config           | Declarative settings and normalization            | Runtime side effects                               |

## Component ownership matrix

Each core responsibility has exactly one production owner package. Other
packages may consume its public schemas, but must not create a second
implementation root. Changes extend these owners in place; a top-level
`preprocessors/`, `encoders/`, `losses/`, `eval/`, `export/`, `v2/`, or similar
parallel tree is an architecture failure.

| Component       | Sole owner package              | Ownership boundary                                                                                |
|-----------------|---------------------------------|---------------------------------------------------------------------------------------------------|
| `preprocessors` | `preprocessing/`                | Extraction, normalization, OCR/ASR/diarization orchestration, and preprocessing provenance.       |
| `augmenters`    | `augmentation/`                 | Training-data augmentation rules and modality augmenters.                                      |
| `collators`     | `mmcrawler_datasets/collation/` | Raw-input batch assembly, padding, masks, and tokenizer-aware collator schemas.                 |
| `encoders`      | `multimodal/model/`             | Randomly initialized text, document, image, audio, and video encoders owned by `MultimodalModel`. |
| `fusion`        | `multimodal/model/`             | `GatedFusion`, token-level fusion, routing, and model-side fusion schemas.                      |
| `losses`        | `training/losses/`              | Optimization objectives and loss composition.                                                     |
| `runtime`       | `training/runtime/`             | Training lifecycle, offline enforcement, checkpoints, resume, and runtime state.                  |
| `evaluator`     | `evaluator/`                    | Evaluation loss, repository-owned metrics, leakage evidence, and reproducibility receipts.         |
| `export`        | `training/export/`              | Inference bundle and checkpoint-family export.                                                    |

The scratch/external-model constraint for these owners is normative in
[ADR-0002](../adr/0002-scratch-only-model-and-preprocessing-boundary.md).

## Current Applied Upgrade

This upgrade makes domain/infrastructure separation, semantic time and identity,
elapsed-time measurement, operational entropy, and filesystem boundaries
explicit architecture requirements.

It also removes the small-module fragmentation failure by keeping
the crawl-task construction and seed-kind resolution in
`crawler/crawl_tasks/crawl_task.py`.

## Merge Gate

A pull request may claim the architecture score improved only if the full
architecture gate above passes locally and in CI.
