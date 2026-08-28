# Multimodal Crawler

Multimodal Crawler is a scoped data pipeline for collecting, preprocessing,
curating, augmenting, and training on page, document, image, audio, video, and
feed data. The repository emphasizes crawl governance, manifest-based lineage,
configurable preprocessing, and a baseline multimodal training loop that can
validate dataflow and checkpoint/evaluation plumbing.

## What This Is

- a crawler with URL-scope checks, private/local IP blocking, redirect
  validation, guarded DNS resolution, robots-aware rules, host pacing, and
  byte-limited fetching
- a snapshot pipeline with raw, curated, augmented, and training-set manifests
- a preprocessing stack for text cleaning, media normalization, PII checks,
  OCR/transcription hooks, language annotation, dedupe, and quarantine
- a task registry with explicit maturity metadata and disabled experimental
  generation defaults
- scratch-initialized training backends using project-owned encoders:
  `pipeline_smoke` for lightweight pipeline, dataset/dataloader, and
  checkpoint save/load validation; `dense_transformer` for full-capacity dense
  training in release stages `candidate` and `production_model`

## What This Is Not

- not a claim of a strong production-quality multimodal model
- not a replacement for evaluation on labeled, representative datasets
- not permission to train on collected data without source rules, license,
  safety, PII, quality, and lineage checks
- not a guarantee that every optional media backend is installed locally

## Architecture Schema

[ADR-0002: Scratch-only MultimodalModel and external models strictly limited
to preprocessing](docs/adr/0002-scratch-only-model-and-preprocessing-boundary.md)
is the normative architecture boundary. External OCR, ASR, and diarization
models may run only in `preprocessing/`; their extracted observations become
derived dataset data with complete provenance. They may not provide embeddings,
hidden states, teacher signals, or trainable modules to the effective training
path. Training and export contain one own, randomly initialized
`MultimodalModel` checkpoint family and run without model downloads or remote
model calls.

Every release model card must follow the
[model card template](training/export/model_card_template.md), which records this
boundary and the evidence used to verify it.

## Installation And Quality Checks

This release supports **Python 3.12 only**. The strict pin is intentional:
the current Torch/TorchAudio dependency family and its locked wheels have been
validated for Python 3.12, not Python 3.14. Keep Python 3.14 installed for
other projects, but create this project's environment explicitly with 3.12.
Python 3.14 support must not be declared until its dependency locks and a
3.12/3.14 CI matrix have both passed.

On Windows:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python scripts/check_python_version.py
python -m pip install --upgrade pip setuptools wheel
```

Install the full multimodal golden-path development environment (tooling +
ASR/OCR/PDF media runtimes):

```bash
python -m pip install -e ".[dev]" \
  --constraint requirements/constraints-py312-cpu.txt \
  --constraint requirements/constraints-py312-preprocessing.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Then start without config changes:

```bash
multimodal-crawler run --environment dev --fresh-run
```

System tools still required on PATH for PDF OCR: `pdfinfo`, `pdftoppm`
(Poppler), and optionally `tesseract` when `preprocessing.ocr.backend` is
`tesseract`.

For Docker or dedicated preprocessing images, the same media package set is
also available as `.[preprocessing-media]`.

### Preprocessing privacy

The privacy implementation lives at `preprocessing/privacy` and is scoped to
preprocessing only. Text, image, audio, document, and video pipelines share one
detector registry and deterministic fail-closed release policy. The project no
longer exposes a separate privacy CLI, HTTP handler, authorization workflow,
attestation service, audit store, review queue, retention worker, or
training-specific privacy application.

Install the project-owned random-initialized training runtime separately:

```bash
python -m pip install -e ".[training-scratch]" \
  --constraint requirements/constraints-py312-cpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Do not combine the media/OCR stack with a production training image; Docker
targets and CI enforce that separation.

Run the local static-quality and packaging checks:

```bash
python -m compileall -q -j 0 .
python scripts/check_python_version.py
ruff check --no-cache .
ruff format --check .
make typecheck
pytest -q
python -m pip wheel . --no-deps --no-build-isolation -w dist-check
```

## Pipeline Overview

1. Load typed settings for the selected environment.
2. Admit seed and discovered URLs through crawler governance.
3. Fetch and persist raw payloads with manifests.
4. Preprocess text, media, and metadata.
5. Build curated dataset snapshots.
6. Optionally augment eligible training rows.
7. Build training snapshots.
8. Train and evaluate the baseline multimodal model.
9. Write manifests, metrics, checkpoints, and acceptance decisions.

`orchestration/` wires phases together. Domain packages such as `crawler/`,
`mmcrawler_datasets/`, `augmentation/`, and `training/` do not depend on
orchestration.

## Configuration Environments

Runtime orchestration loads exactly one canonical `config.settings.root.Settings` tree through `orchestration.settings_loader` / `config.load`. The effective configuration is assembled in this order:

1. `config/profiles/{profile}.toml` (`dev`, `test`, or `prod`)
2. runtime-environment selection (`dev`, `test`, or `prod`), which binds the
   profile and source-registry selection but never encodes release maturity
3. `config/files/sources/source_registry.json` expansion
4. environment-variable overrides
5. explicit CLI overrides (including `--use-cuda`)
6. path resolution and strict Pydantic validation
7. cross-section, governance, and release-contract validation

Checked-in environments:

- `dev`: CPU-oriented development defaults with evaluation checks enabled
- `test`: isolated settings for the test suite
- `prod`: the full production policy, including mandatory evaluation, lineage,
  reproducibility, dataset-card, and model-card gates. A normal production run
  produces a `candidate`; it does not activate a production release.

Run entrypoints:

```bash
multimodal-crawler run --environment dev
python -m orchestration.main run --environment dev
```

### Production invocation

`prod` is the production deployment environment, not a replacement for the
development examples above. Every production control action and run requires
an explicit writable `--project-root` and four pinned Whisper inputs:
an absolute local `model_name` directory containing `model.bin`, its revision,
the lowercase 64-hex SHA-256 artifact hash, and the installed
`faster-whisper` backend version. Set them through the supported overrides,
for example:

```bash
export APP_OVERRIDE__preprocessing__transcription__model_name=/srv/models/whisper
export APP_OVERRIDE__preprocessing__transcription__model_revision=deployment-revision
export APP_OVERRIDE__preprocessing__transcription__model_artifact_hash=REPLACE_WITH_64_LOWERCASE_HEX_SHA256
export APP_OVERRIDE__preprocessing__transcription__backend_version=REPLACE_WITH_INSTALLED_FASTER_WHISPER_VERSION

multimodal-crawler control status --environment prod --project-root /app
multimodal-crawler run --environment prod --project-root /app \
  --checkpoint-headers \
  --checkpoint-blob-storage /secure/checkpoints \
  --staging-lock /secure/staging.lock
```

A production run additionally requires all three checkpoint/staging arguments
shown above. It applies the production policy and produces a release
`candidate`; it never changes the active production release. `--use-cuda` is
optional because the production profile already requests CUDA.

`run` is the single autonomous workflow command. It inspects the validated
workflow state and executes crawl, preprocessing, augmentation, training, or no
work as required until the workflow is complete or blocked. `control` is the
only other public command and provides pause, resume, stop, and status actions.
Both commands use the selected project root only as a writable workspace.
Packaged configuration is read separately and is never used as an artifact
destination. Promotion is a separate, transactional release operation after a
candidate has passed acceptance and reproducibility gates; only that operation
may update the active-release pointer.

## Artifact Layout

- raw crawl output: `data/raw/runs/`
- raw manifests: `data/raw/manifests/`
- curated snapshots: `data/curated/`
- training datasets: `data/interim/training_sets/`
- augmented datasets: `data/interim/augmented_training_sets/`
- candidate model artifacts: `artifacts/candidates/<campaign-id>/seed-<seed>/`
- promoted model releases: `models/releases/` with `models/current.json`
- workflow manifests: `data/registry/workflow_artifacts/`
- checkpoints and training logs: `runtime/training/`, `runtime/logs/`

Runtime artifacts, caches, wheels, and bytecode are ignored and are blocked by
the quality guardrail.

## Dataset and Manifest Schemas

Release `0.3.0` accepts only raw, curated, and training dataset schema `3.0`
and workflow-manifest schema `2.0`. Training source media is represented only
through canonical `objects[]`; old top-level media path, URL, MIME, and byte
fields are rejected. Missing release manifests are never reconstructed from
evaluation or training counters.

Older dataset schemas are rejected. Rebuild artifacts with the current
pipeline; the runtime and distribution contain no compatibility reader or
migration shim.

## Feature Maturity

Task definitions carry explicit metadata:

| Maturity       | Default rule             | Meaning                                                         |
|----------------|--------------------------|-----------------------------------------------------------------|
| `stable`       | may be enabled           | expected to have sample builders, metrics, and a usable backend |
| `beta`         | may be enabled carefully | useful but still needs review on new datasets                   |
| `experimental` | disabled by default      | incomplete backend, metrics, labels, or validation coverage     |
| `disabled`     | always disabled          | registered for planning, not active execution                   |

Examples:

| Task family                      | Typical status        | Default posture                        |
|----------------------------------|-----------------------|----------------------------------------|
| text pretraining/classification  | stable                | enabled                                |
| image-text and retrieval tasks   | stable/beta           | enabled where samples exist            |
| OCR, ASR, QA, captioning         | beta/experimental     | gated by backend, samples, and metrics |
| diarization and generation tasks | experimental/disabled | disabled by default                    |

Task maturity metadata must remain explicit, and experimental or disabled tasks
must not be enabled in the default configuration.

## Training Backend

All training backends share one own, randomly-initialized model family: a
project-owned, `scratch-initialized` `MultimodalModel` with no external encoder
weights (see [ADR-0002](docs/adr/0002-scratch-only-model-and-preprocessing-boundary.md)).
The active backend is `training_backend`; the only backends implemented in this
release are `pipeline_smoke` and `dense_transformer`. Release stage and run mode
are selected independently:

| Training backend | Run mode | Release stage | Purpose |
| ---------------- | -------- | ------------- | ------- |
| `pipeline_smoke` | `full` (default) | `pipeline_smoke` (dev default) | Lightweight pipeline, dataloader, and checkpoint save/load validation |
| `dense_transformer` | `full` | `candidate`, `production_model` | Full-capacity dense training |

The `pipeline_smoke` backend is suitable for:

- lightweight pipeline validation
- validating dataset and dataloader plumbing
- checking checkpoint save/load
- exercising evaluation and acceptance flow

It is not a model-quality claim. Results should be interpreted separately:

- `pipeline_success`: the workflow ran and artifacts were written
- `training_success`: optimization, checkpointing, and evaluation completed
- `quality_success`: task metrics meet configured acceptance thresholds

Evaluation can emit loss, classification metrics, retrieval metrics, text/task
metrics such as exact match or token F1, OCR/ASR error-style metrics, and
configured evaluation thresholds when labeled data is available.

## Safety And Governance

- URL filtering rejects invalid schemes, malformed hosts, blocked hosts, IP
  literals, private/local ranges, and out-of-scope discoveries.
- Redirect targets are validated before response bodies are accepted.
- Guarded network access checks DNS answers before socket use.
- Training snapshots re-check source rules, lineage, dedupe, quality, PII,
  language, caption, alignment, and quota decisions.
- Collection success is not training permission.


## Known Limitations

- Optional OCR, ASR, diarization, audio, and video preprocessing requires the
  `preprocessing-media` extra and native tools such as `ffmpeg`. Those learned
  backends are not installed in scratch training environments.
- Baseline training validates the pipeline; representative model quality still
  depends on labeled data, task-specific metrics, and evaluation thresholds.

## Release Process

Build distributable artifacts with the standard Python packaging toolchain:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Keep runtime artefacts, bytecode, logs, JSONL data, databases, and checkpoints
outside source distributions.
