# Operations Runbook

Status: Operational
Source of truth for: start/stop/pause/resume, exit codes, diagnosis, and
recovery procedures for the autonomous workflow.

Related:
- [startup_configuration.md](startup_configuration.md) — config loading and typed startup errors
- [crawl_terminal_outcomes.md](../architecture/crawl_terminal_outcomes.md) — terminal outcomes and operator actions
- [incident-response.md](incident-response.md) — incident response procedure

## Startup

Prerequisites:

- Python `>=3.12,<3.13` (see `pyproject.toml`).
- Install profile: `pip install -e ".[dev]"` with the pinned constraint files:
  `--constraint requirements/constraints-py312-cpu.txt --constraint requirements/constraints-py312-preprocessing.txt` plus the PyTorch CPU wheel index `--extra-index-url https://download.pytorch.org/whl/cpu`.
- Native dependencies required by the preprocessing profile (image/audio codecs) must be present; the GPU profile additionally needs CUDA matching the lock files in `requirements/`.
- Config root: the operator config directory must contain the mandatory local files listed in [startup_configuration.md](startup_configuration.md), including `governance/processing_activities.json`.
- The project root must be writable by the run user. The config root is a
  readable, read-only artifact tree; checkpoint, state, dataset, and promotion
  writes belong under the project root.

Preflight:

```powershell
python --version
python -m pip check
multimodal-crawler control validate-config --environment dev
```

Start or continue the autonomous workflow:

```powershell
multimodal-crawler run --environment dev
```

## Production deployment inputs

Every `prod` command loads the one production profile, including read-only
control actions. Set an explicit writable `--project-root` and these
deployment-pinned overrides before using one:

- `APP_OVERRIDE__preprocessing__transcription__model_name`: absolute local
  model directory containing `model.bin`.
- `APP_OVERRIDE__preprocessing__transcription__model_revision`.
- `APP_OVERRIDE__preprocessing__transcription__model_artifact_hash`: lowercase
  64-hex SHA-256 of the local model artifact.
- `APP_OVERRIDE__preprocessing__transcription__backend_version`: installed
  `faster-whisper` version.

For a production `run`, also pass all three mandatory release-storage flags:
`--checkpoint-headers`, `--checkpoint-blob-storage DIR`, and
`--staging-lock PATH`. Missing pins, a missing `model.bin`, a hash mismatch,
or a missing flag is a deliberate fail-closed startup error. `--use-cuda` is
optional because the production profile already requests CUDA.

A successful `prod` run applies the full production policy and creates a
`candidate` with acceptance and reproducibility evidence. It does not replace
the active production release or its `current.json` pointer. Promotion is a
separate, operator-controlled transactional release operation: it revalidates
the accepted candidate against the final production gate before atomically
updating that pointer.

## Start / Stop / Pause / Resume

- Pause: `multimodal-crawler control pause --environment dev`.
- Resume: `multimodal-crawler control resume --environment dev`, then restart `run` with the same project root.
- Stop: `multimodal-crawler control stop --environment dev` or SIGTERM for graceful shutdown.
- Status: `multimodal-crawler control status --environment dev`.

A run that was interrupted or paused resumes from the last strict checkpoint
(worker task counters and retry budgets are serialized via `to_payload()`).

## Exit codes

Only codes actually implemented in the codebase (`orchestration/main.py`,
`orchestration/bootstrap/application.py`).

| Code | Constant                        | Meaning                                            | When it happens                                             |
| ---: | ------------------------------- | -------------------------------------------------- | ----------------------------------------------------------- |
| `0`  | `SUCCESS_EXIT_CODE`             | Success                                            | Workflow completed, control succeeded, or configuration is valid |
| `1`  | `FAILURE_EXIT_CODE`             | Runtime or unexpected validation failure           | Unrecoverable exception, or run outcome not completed       |
| `2`  | `STARTUP_CONFIGURATION_EXIT_CODE` or `EXIT_PARTIAL_DOWNSTREAM_INVALID` | Startup/configuration error or incomplete workflow | Invalid settings, missing artifact, unavailable required backend, or a controlled/incomplete downstream workflow outcome |
| `130`| `INTERRUPTED_EXIT_CODE`         | Keyboard interrupt                                 | `Ctrl+C` during the run                                     |
| `131`| `CANCELLED_EXIT_CODE`           | Cancelled                                          | `asyncio.CancelledError` during the run                     |

Exit code `2` alone does not distinguish a startup failure from an incomplete
workflow: inspect the `Startup configuration error:` block and the persisted
workflow/terminal outcome. Exit code `0` alone is not proof the dataset is
promotion-ready; the terminal outcome (see
[crawl_terminal_outcomes.md](../architecture/crawl_terminal_outcomes.md))
decides promotion eligibility.

## Diagnosis matrix

| Symptom / outcome                                  | Evidence to collect                                                                                        | Root cause checks                                        | Resolution                                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `processing_activity_config_missing`               | `Startup configuration error:` block on stderr, exit code `2`                                              | `governance/processing_activities.json` missing/invalid; `governance.processing_activities_file` setting points elsewhere | Restore the mandatory config file at the config root; verify setting name; re-run  |
| `incomplete_transient_infrastructure`              | `root_seeds_transient_failed>0`, `retry_exhausted_total>0`, DNS/network/robots errors in logs              | DNS, network, robots fetch, 5xx backends                 | Restore DNS/network/robots access; re-run (this is the only auto-retry outcome)    |
| `completed_no_eligible_content`                    | `object_records_total=0`, `eligible_records_total=0`, `root_seeds_total>=1`                                | Extraction/admission gates, source profile               | Investigate extraction and source selection; adjust source profile; re-run         |
| `completed_governance_blocked`                     | `root_seeds_governance_blocked == root_seeds_total`, `root_seeds_succeeded=0`                              | Robots rules, source registry status, URL admission      | Review governance rules; invalidate robots cache if stale; re-run                  |
| `completed_below_readiness_threshold`              | `unmet_requirements=[...]`, `quality_score`, `modality_counts`                                             | Readiness gate thresholds vs. achieved metrics           | Adjust `crawl_output_gate` thresholds or source profiles; re-run                   |
| `incomplete_dependency_failure`                    | `required_dependency_failures>0`, or startup error with exit code `2`                                      | Missing config, native dependency, unavailable backend   | Fix deployment/dependency; re-run                                                 |

## Recovery procedures

### Pause / resume

1. `multimodal-crawler control pause --environment dev`.
2. Wait for in-flight tasks to settle (host queue drains).
3. Investigate and remediate.
4. `multimodal-crawler control resume --environment dev`, then restart `run`.

### Safe retry

Only re-run immediately when the outcome is `incomplete_transient_infrastructure`.
For all other non-completed outcomes, remediate the root cause first; do not
blindly retry a `completed_*` outcome.

### Checkpoint restore

State is restored from strict checkpoint payloads (retry budgets via
`TaskRetryState.to_payload()`). To restore:

1. Stop the run.
2. Restore the state snapshot from the last consistent checkpoint (same run id).
3. Restart `run` with the same project root; the scheduler and worker counters resume.

### Dataset promotion rollback

1. Identify the previous promoted dataset tag.
2. Restore the dataset pointer/manifest to the previous tag.
3. Validate the rollback in staging before applying to prod.

### Corrupt pointer / manifest

- Use `datachecker` repair on the affected dataset run id, or re-crawl the run id if repair is not possible.

### Missing processing-activity config

1. Restore `governance/processing_activities.json` at the config root (schema version `1.0.0`, see [startup_configuration.md](startup_configuration.md)).
2. Re-run; the run fails fast with exit code `2` rather than starting a crawl.

### Robots host cooldown

- A host that failed robots fetching is subject to a cooldown; new tasks for that host are deferred (`reason="host_not_ready"`) and do not consume the task retry budget.
- To force a re-check: invalidate the robots cache for the host, then resume the run.

## Validation after intervention

Always re-run:

- `multimodal-crawler control status --environment prod --project-root /app`
  with the production deployment inputs above
- `python -m pytest -q tests/config tests/datachecker` for workflow/config acceptance
- `python -m pytest -q tests/training/acceptance tests/training` if training changed
- `python -m bandit -q -ll -r augmentation config crawler datachecker logger mmcrawler_datasets multimodal orchestration preprocessing training`

New operators: execute the golden-path validation ([golden_path.md](../golden_path.md))
and one stability run before touching prod.
