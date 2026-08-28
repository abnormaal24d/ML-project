# Glossary

Canonical terms for the multimodal crawler. When a document and the code disagree,
the code is the source of truth; these terms are the single vocabulary every
document must use.

## scratch-initialized

Architecture rule (see [ADR-0002](adr/0002-scratch-only-model-and-preprocessing-boundary.md)):
the trainable `MultimodalModel` is project-owned and randomly initialized. It is
**not** loaded from, or fine-tuned against, external pretrained encoders,
embeddings, hidden states, teachers, or remote model calls. The same
scratch-initialized model family underlies every release stage; only run mode,
capacity, and evidence requirements change.

## backend

One of the registered `TrainingSettings.training_backend` values
(`config/multimodal/training_settings.py`):

- `pipeline_smoke` — the lightweight, plumbing-validation backend. Runs the
  project-owned, scratch-initialized encoders; validates the pipeline, the
  dataset/dataloader plumbing, and checkpoint save/load. It is **not** a
  model-quality claim.
- `dense_transformer` — the full-capacity dense training backend. Used by the
  `candidate` and `production_model` release stages.

A backend is `supported` when it is accepted by configuration and `implemented`
when runtime wiring exists for it. These sets are derived from
`SUPPORTED_TRAINING_BACKENDS` and `IMPLEMENTED_TRAINING_BACKENDS`.

## run mode

A `TrainingSettings.run_mode` value (`smoke`, `full`). Selects execution
capacity and evidence requirements for a run; independent of the backend family.
Default is `full`.

## release stage

A `TrainingSettings.release_stage` value — the maturity gate that wraps a run:
`pipeline_smoke`, `learning_candidate`, `candidate`, `production_model`. It is
independent from the runtime environment, which is exactly `dev`, `test`, or
`prod`. A normal `prod` workflow uses `candidate`; `production_model` is the
final promotion gate, not an environment name.

## training stage

A `TrainingSettings.training_stage` value — the phase within the training
schedule (for example `DATASET_FREEZE`, `MULTIMODAL_PRETRAIN`, `ACCEPTANCE`,
`PROMOTION`).

## source profile

A named seed-URL and governance profile selected from `config/files/sources/`.

## snapshot

A point-in-time, hash-verified artifact set:

- **raw snapshot** — persisted raw payloads and raw manifests (`data/raw/`).
- **curated snapshot** — governed, deduplicated records (`data/curated/`).
- **training snapshot** — rows cleared for training
  (`data/interim/training_sets/`).

## acceptance

The evidence decision for a `candidate` run. A candidate is accepted only when
the workflow, crawl, training, quality, and reproducibility phases meet the
full production policy with a `final=true` manifest and a matching config
fingerprint. Acceptance alone does not activate a production release.

## promotion

The separate transactional operation that revalidates an accepted candidate at
the final production gate and advances it to the active production release. It
is the only operation allowed to update the active-release pointer. A promoted
dataset always carries `final=true` and a config fingerprint that links it to
its environment and source-registry version.
