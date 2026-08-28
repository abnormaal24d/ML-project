# Configuration

Status: Operational + Normative
Source of truth for: how the configuration schema is produced, which setting
groups exist, how environments and release stages are separated, and how schema
changes are reviewed. The normative configuration rules are captured in
[adr/0004-configuration-architecture.md](adr/0004-configuration-architecture.md)
(one canonical name per setting, single load path, strict models,
discriminated-union backends, policy/tuning separation, side-effect-free
`validate-config`).

Related:
- [adr/0004-configuration-architecture.md](adr/0004-configuration-architecture.md) — normative rules
- [operations/startup_configuration.md](operations/startup_configuration.md) — config loading, mandatory files, typed errors
- [configuration_schema.json](configuration_schema.json) — the generated schema
- [runbook.md](operations/runbook.md) — operator procedures

## The generated schema

`configuration_schema.json` is a generated reference derived from the Pydantic
settings models in `config/`. It is **generated** (classification: Generated):
do not edit it by hand.

| Property | Value |
| -------- | ----- |
| Source | Pydantic settings models in `config/` (`SettingsModel` subclasses, e.g. `config/settings/root.py`) |
| Generation command | `python -c "import json; from config.settings.root import Settings; print(json.dumps(Settings.model_json_schema(), indent=2))"` (preserve the generated-file `$comment` when writing the checked-in artifact). |
| Verification command | `python -m pytest -q tests/training/test_training_runtime_blockers.py` — asserts the checked-in schema matches the live Pydantic models (e.g. `TrainingSettings` properties) |
| Re-generation | When a settings model changes, regenerate the schema from the models and update the verification test before committing |

Because the schema is verified against the live models by tests, any manual edit
that disagrees with the models fails verification.

## Important setting groups

The schema is organized as `$defs` entries per settings model. The main groups:

| Group | Model | Purpose |
| ----- | ----- | ------- |
| Application | `config/settings/app.py` | Identity, runtime environment, release-requirement selection |
| Paths | `config/settings/paths.py` | Resolved writable workspace, data, cache, and output roots |
| Sources | `config/settings/sources.py` | Expanded source registry, scopes, governance, and active seeds |
| Collection | `config/settings/collection.py` | Fetching, pacing, discovery, governance, processors, and HTTP rules |
| Preprocessing | `config/settings/preprocessing.py` | Text/media preprocessing and backend settings |
| Multimodal | `config/settings/multimodal.py` | Canonical model architecture and generation settings |
| Training | `config/settings/training.py` | Canonical training, task, release-stage, and optimization settings |
| Augmentation | `config/settings/augmentation.py` | Typed augmentation policy per modality |
| Classification | `config/settings/classification.py` | Content-kind and detector settings |

### Augmentation contract

Augmentation has one typed configuration contract. `augmentation.enabled`
switches the complete phase on or off; `augmentation.text.enabled` controls
text-field variants. Image, audio, and video transforms use their own
`enabled` fields. Document media transforms additionally require
`augmentation.document.mode = "document_media"`; the default
`"text_field_only"` mode changes document text fields without transforming
the document binary. Production restrictions are enforced after all TOML,
environment, and CLI layers have been merged by the cross-sectional release
validator, so later overrides cannot silently enable unreleased media
transforms.

## Environment selection and release stages

Configuration is layered (see ADR-0004 rule 9 — exactly one documented order):

1. Canonical profile: `config/profiles/{profile}.toml` (`dev`, `test`, or `prod`).
2. Runtime-environment selection: exactly `dev`, `test`, or `prod`. It binds
   the profile and source-registry selection; it is not a release stage.
3. Source-registry expansion from `config/files/sources/source_registry.json`.
4. Environment-variable overrides.
5. CLI overrides.
6. Filesystem path resolution.
7. Strict Pydantic and config-owned cross-section validation.
8. Orchestration-owned governance, task-registry, release-contract, and
   runtime/backend readiness validation.

Merge semantics (ADR-0004 rule 10): mappings deep-merge; scalar values and
lists/tuples replace; lists are never implicitly appended.

The active environment is selected with `--environment` (e.g.
`multimodal-crawler run --environment dev`). Environment variables are read
only by the canonical configuration facades
(`config/environment/runtime_environment.py` and
`config/environment/source_selection.py`); product modules must not read
runtime environment variables directly (ADR-0004 rule 4).

`prod` always applies the one full production policy from `prod.toml` and
begins a normal workflow at release stage `candidate`. Candidate acceptance
uses the production task and runtime thresholds. A candidate must complete its
acceptance and reproducibility evidence before the separate transactional
promotion operation may perform the final production gate and atomically update
the active production release. `run` never performs that promotion implicitly.

`config_root` always supplies runtime artifacts below `config/files`. A custom
root may additionally supply `config/profiles/{profile}.toml`; when that exact
profile is absent, the loader uses the packaged canonical profile. This lets
operators mount artifacts independently without copying the profile catalog.

## Required vs optional files

- **Required at config root**: `governance/processing_activities.json`
  (setting `governance.processing_activities_file`). Its absence fails startup
  with exit code `2` — see
  [startup_configuration.md](operations/startup_configuration.md).
- **Profiles**: `config/profiles/test.toml`, `dev.toml`, and `prod.toml` are the canonical runtime profiles.
- **Governance/source artifacts**: `config/files/governance/processing_activities.json` and
  `config/files/sources/source_registry.json`. Release contract data is now defined within
  `prod.toml` under `[release]` and is derived at runtime, eliminating the separate
  `config/files/releases/*.toml` configuration layer.
- **Removed legacy paths**: `config.loader`, `config.settings_tree`, and files
  under `config/files/defaults/`, `config/files/environments/`, and
  `config/files/multimodal_profiles/` are not supported configuration APIs.

## Example configuration

See `config/profiles/test.toml`, `config/profiles/dev.toml`, and
`config/profiles/prod.toml` for the canonical runtime profiles.

### Document native-text migration

`collection.processors.document.text_preview` has been renamed to
`collection.processors.document.native_text`. Its `max_characters` limit
controls the bounded native text retained as `document_text`, rather than a
persisted short preview. The former `pdf_header_scan_bytes`,
`text_reader_encodings`, `remove_duplicate_lines`, `normalize_whitespace`, and
`reject_binary_without_metadata` settings have been removed because they had no
runtime effect. There are no compatibility aliases: old keys fail strict
configuration validation and deliberately produce a configuration/fingerprint
migration.

## Schema change review

Any change to a settings model changes the generated schema. Review checklist:

- Regenerate the schema from the models (keep in sync with the verification test).
- Check defaults remain compatible with existing environment files (strict
  `extra="forbid"` and `validate_default=True` reject unknown/loose values).
- Confirm no mandatory file requirement changed (processing-activity registry).
- Run `python -m pytest -q tests/config` for workflow/config acceptance.
