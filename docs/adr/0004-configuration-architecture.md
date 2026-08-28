# ADR-0004: Domain-owned configuration architecture

## Status

Accepted.

## Context

Configuration has grown across the codebase and must remain auditable. Rules
must be settled about: ownership of every setting, the single load path,
structure of the settings tree, merge semantics, environment-variable access,
and the boundary between policy and tuning.

The orchestration runtime uses the canonical `config.settings.root.Settings`
tree, loaded through `config.load` and completed with domain validation at the
`orchestration.settings_loader` boundary. The former `config.loader` and
`config.settings_tree` paths have been removed and are not compatibility APIs.

## Decision

The rules below are normative for all settings code.

1. **One canonical name per setting.** Every setting has exactly one canonical
   dotted key; no aliases, no duplicate fields, no `os.getenv()` in product
   modules, no broad `Settings` injection through the project.
2. **One owning domain subconfig per setting.** The root config only aggregates
   domain subconfigs, validates global cross-domain invariants, and never
   builds services, opens files, performs network I/O, or supports old field
   names.
3. **Strict base model everywhere.** All settings models use
   `extra="forbid"`, `frozen=True`, `validate_default=True`,
   `str_strip_whitespace=True`. Never `extra="ignore"`; no
   `populate_by_name=True` when aliases are not intended. Unknown keys are
   startup errors.
4. **Central loader only.** Runtime configuration I/O and overrides are owned by
   `config/load.py`, `config/overrides.py`, and `orchestration/settings_loader.py`.
   Product-domain modules (crawler, training, datasets, augmentation) do not
   read runtime environment variables directly.
5. **Root settings stop before workflow/runtime.** The canonical root
   `Settings` may be loaded and validated at the configuration/bootstrap
   boundary and consumed by bootstrap or composition builders. It must not be
   retained in returned runtime containers or object graphs, and workflow or
   product-domain services must not depend on the root object. Runtime
   consumers receive the smallest relevant typed subconfig, derived immutable
   value, policy, service, factory, or capability rather than the root object.
6. **No sibling cross-imports.** Leaf settings modules import only
   `config/common/`, the standard library, and Pydantic. Parents import
   children. Direction: common → leaf → domain parent → root. Cross-domain
   validation lives in the root settings, not in leaf modules.
7. **Backends use discriminated unions.** Backend choice is modeled as a
   discriminated union keyed on a `backend` literal (OCR, speech, video
   decoding, object storage, checkpoint storage, logging backend, metrics
   exporter, tokenizer, model backend, queue/backend). No fat optional-field
   classes; no conflicting backend combinations.
8. **Separate defaults from product decisions.** Safe technical defaults may be
   in code (`connect_timeout_seconds: float = 10.0`). Governance, security, and
   release decisions (robots enablement, processing-activity file, promotion
   strictness, personal-data allowance, unknown-result action) must be explicit
   configuration with no hidden permissive defaults.
9. **Explicit layered config.** Runtime configuration is assembled in one
   documented order: canonical profile TOML → runtime-environment semantics →
   source-registry expansion → environment-variable overrides → CLI overrides →
   path resolution → strict Pydantic and cross-section validation.
10. **Explicit merge semantics.** Mappings deep-merge; scalar values and
    lists/tuples replace (never implicitly append). Append behavior, when
    needed, is an explicit configuration concept, not magic merge code.
11. **Secrets never in repository TOML.** Secrets come from environment
    variables, secrets managers, or mounted secret files, modeled as
    `SecretStr`. Never log `settings.model_dump()` without explicit
    redaction.
12. **Paths are explicitly configurable.** Config artifacts are declared as
    settings (`file: Path`), resolved by the central loader relative to the
    config root, and normalized to absolute paths after validation. No
    hardcoded path concatenation in composition.
13. **Policy and tuning are separated** within larger domains
    (`SchedulerPolicySettings` vs `SchedulerTuningSettings`). Policy controls
    meaning and correctness; tuning controls capacity and performance.
14. **Output fingerprints use explicit projections.** Content fingerprints
    contain only settings that change output content — never output
    directories, log files, worker counts, cache paths, metrics exporters,
    snapshot filenames.
15. **Configurability follows ownership.** Timeouts, limits, backends,
    retries, paths, quality thresholds, governance decisions, resource
    budgets, release policy, and logging level are configurable. Schema field
    names, cryptographic algorithms, architectural invariants, output types,
    mandatory security checks, transaction steps, and lineage obligations are
    not.
16. **A side-effect-free `validate-config` command exists.** The CLI provides
    `multimodal-crawler control validate-config --environment <env>` that loads
    all layers, validates all subconfigs, checks mandatory local artifacts and
    required backends, and executes or mutates nothing. Failures print the
    dotted key, the problem, and the required artifact.

## Consequences

- Configuration stays auditable: one owner, one name, one load path per
  setting; typo'd or unknown keys fail at startup.
- Workflow and runtime services remain decoupled from the root settings tree.
  Bootstrap/composition builders may consume canonical `Settings` and project
  it into exact typed subconfigs, immutable derived values, policies,
  factories, and services.
- Environment-variable access becomes centralized, so redaction and secret
  handling are enforceable in one place.
- Discriminated unions keep backend configuration self-documenting and make
  invalid combinations unrepresentable.
- The `validate-config` command gives operators a preflight that mirrors
  startup failure behavior without running a crawl.
- Follow-up work (migration plan, test strategy, schema generation) is tracked
  in [../configuration.md](../configuration.md) and the roadmap.

## Acceptance criteria

- Architecture tests prohibit legacy settings imports from orchestration and
  keep the canonical root settings object inside the orchestration/configuration boundary.
- `orchestration/workflow/**` does not import `config.settings.root.Settings`.
- Product-domain packages do not depend on root `Settings`.
- Bootstrap/composition may consume root `Settings` for construction.
- Returned runtime containers and workflow objects do not retain root
  `Settings`.
- Runtime consumers receive exact typed subconfigs, policies, derived values,
  factories, or capabilities.
- Runtime environment access is restricted to the canonical configuration boundary.
- All backend choice settings use discriminated unions.
- `validate-config` runs against every environment without side effects.
- No alias or backward-compatibility setting fields remain.
