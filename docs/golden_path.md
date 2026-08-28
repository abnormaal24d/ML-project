# End-to-End Golden Path

## Purpose

The golden path is the smallest complete workflow that proves the product works as scoped: approved seed URLs become
auditable raw records, curated records, training snapshots, baseline training/evaluation artifacts, and final acceptance
decisions.

This path is the product-level definition of success. A feature is not production-ready if it cannot preserve this path.

## Golden path statement

Given the `dev` environment and an approved small public-science source profile, the pipeline must:

1. load typed configuration and source registry;
2. admit only in-scope seed URLs;
3. apply URL, DNS/IP, redirect, robots, pacing, and byte-budget governance;
4. fetch approved raw payloads;
5. classify content modality;
6. persist raw artifacts and raw manifests;
7. preprocess text/media/metadata;
8. apply source-rules, PII, quality, dedupe, and lineage gates;
9. assemble curated and training snapshots;
10. run baseline training and evaluation;
11. write final manifests and acceptance decisions;
12. exit with completed/final status.

## Minimal validation environment

| Item               | Required value                                                           |
|--------------------|--------------------------------------------------------------------------|
| Python version     | `>=3.12,<3.13` (see `pyproject.toml`)                                     |
| Install profile    | `pip install -e ".[dev]"` — the single golden-path install; `dev` inlines the multimodal media stack (preprocessing-media set) and all tooling                                           |
| Environment        | `dev`                                                                   |
| Source profile     | `public_science_small`                                                    |
| Source families    | Small approved NASA, NOAA, or USGS subset                                |
| Network mode       | Approved public network access or operator-provided local payloads       |
| Training backend   | `pipeline_smoke` (default; smoke validation path)                        |
| Run mode           | `full` (default)                                                         |
| Release stage      | `pipeline_smoke` (dev default)                                           |
| Model init         | `scratch-initialized` (project-owned, random — [ADR-0002](adr/0002-scratch-only-model-and-preprocessing-boundary.md)) |
| Runtime size       | Small enough for a controlled validation run                             |
| Output destination | Temporary workspace, never checked into source release                   |

## Required command sequence

```bash
python --version
python -m pip install -e ".[dev]" \
  --constraint requirements/constraints-py312-cpu.txt \
  --constraint requirements/constraints-py312-preprocessing.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -c "import torch, torchaudio; from pyannote.audio import Pipeline; assert torch.__version__.split('+', 1)[0] == torchaudio.__version__.split('+', 1)[0]"
python -m compileall -q -j 0 .
ruff check --no-cache .
ruff format --check .
mypy --config-file pyproject.toml .
pytest -q
multimodal-crawler control validate-config --environment dev
multimodal-crawler run --environment dev
```

Run the workflow in an isolated project workspace and verify the resulting manifests and acceptance decisions.

### Startup preflight

Before `run`, the config root must contain the mandatory local files (see
[operations/startup_configuration.md](operations/startup_configuration.md)),
including `governance/processing_activities.json` (schema version `1.0.0`).
`multimodal-crawler control validate-config --environment dev` must exit `0`.
It uses the same settings, local-artifact, dependency, and backend validation
as `run`, without creating runtime state. A missing processing-activity config
fails fast with exit code `2` before any crawl loop starts.

### Expected terminal outcome

A passing golden run must terminate with `stop_trigger=frontier_drained` and the
terminal outcome `completed_ready` (see
[architecture/crawl_terminal_outcomes.md](architecture/crawl_terminal_outcomes.md)).
Any other terminal outcome means the golden path failed and must be diagnosed
with the runbook diagnosis matrix before re-running.

## Required final artifacts

A successful golden run must produce or validate these artifacts.

| Artifact                   | Required condition                                                        |
|----------------------------|---------------------------------------------------------------------------|
| Raw run manifest           | Exists, schema `3.0`, `status=completed`, `final=true`                   |
| Crawl manifest             | Exists, schema-valid, final URL and object counts match persisted records |
| Crawl state manifest       | Exists, `status=completed`, no stale `running` state                      |
| Raw object records         | JSONL exists, valid, count matches manifest                               |
| Error records              | JSONL exists, even if empty, and errors are classified                    |
| Curated snapshot manifest  | Exists and references only valid raw records                              |
| Training snapshot manifest | Exists, schema `3.0`, and contains only training-allowed records         |
| Acceptance decision        | Exists under workflow-manifest schema `2.0` and reports pass/fail        |
| Training metrics           | Exists for the baseline evaluation path                                          |
| Config fingerprint         | Exists and links run to environment and source-registry version           |

## Required manifest invariants

The acceptance checker must fail if any invariant is violated.

| Invariant           | Requirement                                                                   |
|---------------------|-------------------------------------------------------------------------------|
| Finalization        | No golden run may end with `status=running`                                   |
| Atomicity           | Final manifests are written only after records and counters validate          |
| Lineage             | Every curated/training record traces to a raw object and source URL           |
| Storage integrity   | Every referenced artifact exists and matches hash/size metadata               |
| Rules trace        | Every training record includes source-rules and training-permission decision |
| PII trace           | Every training record includes PII decision or proof of check                 |
| Dedupe trace        | Every training record includes dedupe decision or dedupe batch reference      |
| Quality trace       | Every training record includes quality decision                               |
| Split integrity     | Validation/test leakage checks pass                                           |
| Release cleanliness | Golden run artifacts are excluded from source release packages                |

## Pass/fail criteria

A golden run passes only when all product-critical stages pass.

```json
{
  "pipeline_success": true,
  "crawl_success": true,
  "curation_success": true,
  "training_snapshot_success": true,
  "training_success": true,
  "quality_success": true,
  "workflow_acceptance_success": true
}
```

A run must fail if:

- any final manifest is missing;
- any required manifest remains `running`;
- `final=true` is missing from final artifacts;
- object counts disagree across manifests and JSONL records;
- source, PII, license, quality, dedupe, or lineage gates are missing for training rows;
- artifacts exist outside the configured runtime/data workspace;
- a source release package includes runtime outputs.

## Manual reviewer checklist

A reviewer should be able to answer these questions from the artifacts alone:

1. Which environment and config produced the run?
2. Which source profile and source-registry version were used?
3. Which seed URLs were admitted and which were rejected?
4. Which records were fetched, curated, rejected, quarantined, or trained on?
5. Why was each training row allowed?
6. Were robots, redirect, DNS/IP, rate-limit, PII, dedupe, quality, and lineage gates applied?
7. Did baseline training/evaluation complete?
8. Did the run close transactionally as `completed` and `final`?
9. Are all runtime artifacts excluded from release packaging?

## Minimal production smoke test

Before a feature may touch prod, the operator must execute the golden-path
validation in `dev` and additionally verify:

- `multimodal-crawler control status --environment prod --project-root /app` reports healthy after the production deployment pins from the [runbook](operations/runbook.md#production-deployment-inputs) are supplied;
- `python -m pytest -q tests/config tests/datachecker` passes (workflow/config acceptance);
- if training changed: `python -m pytest -q tests/training/acceptance tests/training`;
- `python -m bandit -q -ll -r augmentation config crawler datachecker logger mmcrawler_datasets multimodal orchestration preprocessing training`.

## Promotion and rollback verification

- Promotion is transactional: final manifests and acceptance decisions are
  written only after records and counters validate; a promoted dataset always
  has `final=true` and a matching config fingerprint.
- Rollback: restore the previous dataset pointer/manifest tag, then validate
  the rollback in staging before applying to prod. See
  [operations/runbook.md](operations/runbook.md) for the full procedure.

## Definition of done

The golden path is production-ready when:

- the golden-path artifact assertions pass on a fresh run;
- the same run is reproducible in a clean temporary workspace;
- a controlled validation run exercises the full path reproducibly;
- live/dev acceptance can be run manually against approved sources;
- every product-critical use case maps to at least one golden-path artifact or invariant.
