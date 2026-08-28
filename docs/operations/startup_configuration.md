# Startup Configuration

Status: Operational + Normative
Source of truth for: how configuration artifacts are loaded at startup, which
artifacts are mandatory, how failures are classified, and what operators may
safely log or share.

Related:
- [runbook.md](runbook.md) — exit codes and recovery
- [crawler/governance/processing_activity.py](../../crawler/governance/processing_activity.py) — registry loading and `ProcessingActivityConfigError`
- [orchestration/errors.py](../../orchestration/errors.py) — typed bootstrap errors
- [orchestration/main.py](../../orchestration/main.py) — entrypoint handling of startup errors

## Configuration roots

Two distinct roots exist and must not be conflated:

| Root         | Content                                              | Writable |
| ------------ | ---------------------------------------------------- | -------- |
| Config root  | Governance and operator configuration artifacts (e.g. `governance/processing_activities.json`) | No (read-only at runtime) |
| Project root | Checkpoint, state, dataset and promotion trees        | Yes      |

Config artifacts belong to the read-only config root. The workflow never writes
configuration; a write attempt indicates an operator error.

Canonical profile TOMLs are part of the installed application. An explicitly
selected config root is an artifact overlay and therefore only needs to provide
`config/files`; it is not required to copy `config/profiles`. When the overlay
does provide `config/profiles`, those explicit profiles are honored. Otherwise
profile loading falls back to the packaged canonical profiles. Source
registries, governance files, and release requirements are always resolved
from the selected artifact root.

## Mandatory local files

| Relative path                             | Setting                               | Purpose                                  |
| ----------------------------------------- | ------------------------------------- | ---------------------------------------- |
| `governance/processing_activities.json`   | `governance.processing_activities_file` | Processing-activity/DPIA registry (schema version `1.0.0`) |

Additional mandatory files (e.g. source registry, robots cache) are declared in
the configuration schema ([configuration_schema.json](../configuration_schema.json))
and are validated by the same typed-error path.

## Typed startup errors

Startup configuration failures are never logged as raw tracebacks. Each is
classified into a typed error carrying safe structured context.

| Error type                          | Kind                           | Meaning                                            |
| ----------------------------------- | ------------------------------ | -------------------------------------------------- |
| `ProcessingActivityConfigError`     | `processing_activity_registry` | A processing-activity artifact is unusable         |
| `SettingsLoadError`                 | `settings_load_error`          | Settings sources could not be loaded                |
| `SettingsValidationError`           | `settings_validation_error`    | Settings violate a runtime-domain invariant         |
| `BackendConfigurationError`         | `backend_configuration_error`  | A selected backend/runtime mode cannot serve the workflow |
| `LoggingConfigurationError`         | `logging_configuration_error`  | Runtime logging could not be configured safely      |
| `ApplicationWiringError`            | `application_wiring_error`     | A runtime dependency graph is incomplete/inconsistent |
| `RuntimeServicesBuildError`         | `runtime_services_build_error` | Crawler runtime services could not be constructed  |

The entrypoint prints the error to stderr in a fixed format and exits with code
`2` (`STARTUP_CONFIGURATION_EXIT_CODE`), e.g.:

```
Startup configuration error:
  component: processing_activity_registry
  setting:   governance.processing_activities_file
  file:      processing_activities.json
  issue:     processing_activity_config_missing
```

## Safe fields vs. never-logged fields

Safe to include in logs and error output (structured, no local topology):

| Field       | Example                                   |
| ----------- | ----------------------------------------- |
| `component` | `processing_activity_registry`            |
| `setting`   | `governance.processing_activities_file`   |
| `basename`  | `processing_activities.json`              |
| `issue`     | `processing_activity_config_missing`, `processing_activity_registry_invalid` |
| `required`  | `governance/processing_activities.json` (relative path only) |

Never log (absolute local topology and credentials):

- Absolute local paths (e.g. `C:\app\config\...`, `/app/config/...`)
- Usernames and home directories
- Tenant directory layout and mount layout
- Credentials, tokens, and connection strings

Rationale: an operator must know exactly which setting and which basename is
wrong without leaking the deployment topology or secrets.

## Startup preflight

Before starting a run, verify (see [runbook.md](runbook.md)):

1. Python `>=3.12,<3.13` and the install profile are correct.
2. The config root contains all mandatory files, including `governance/processing_activities.json`.
3. The config root is read-only and the project root is writable.
4. `multimodal-crawler control validate-config --environment dev` exits `0`.

The command performs the same settings, local-artifact, dependency, and
backend readiness checks as `run`, but creates no workflow state. A validation
failure exits `2`; an unexpected internal failure exits `1`. A missing
processing-activity config therefore fails before any crawl loop starts.
