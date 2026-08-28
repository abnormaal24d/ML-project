# Documentation Index

This README is the entry point for the `docs/` tree. It classifies every document so
that a developer can immediately tell what is the source of truth, what is
operational guidance, what is generated, and what is historical.

## Start here

| If you want to...                    | Read                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Understand the product scope         | [product_scope.md](product_scope.md)                                                             |
| Run the smallest complete workflow   | [golden_path.md](golden_path.md)                                                                 |
| Operate the system in production     | [operations/runbook.md](operations/runbook.md)                                                   |
| Diagnose startup/configuration errors | [operations/startup_configuration.md](operations/startup_configuration.md)                       |
| Understand crawl terminal outcomes   | [architecture/crawl_terminal_outcomes.md](architecture/crawl_terminal_outcomes.md)               |
| Understand scheduler and retry rules | [architecture/scheduler_retry_semantics.md](architecture/scheduler_retry_semantics.md)           |
| Review security posture              | [security/threat-model.md](security/threat-model.md)                                             |

## Document classification

Every document in this tree belongs to exactly one class. Generated files must never
be edited by hand; historical files are snapshots and never represent current truth.

| Type               | Meaning                                                                    | Examples                                                                      |
| ------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Normative          | Source of truth: rules that implementations must satisfy                   | `adr/*`, `product_scope.md`, `source_governance.md`, `architecture/dependency_inversion.md` |
| Operational        | Runbooks, incident response, golden path, operator procedures              | `operations/runbook.md`, `operations/incident-response.md`, `golden_path.md`   |
| Architecture ref   | Ownership and contract documents that describe current design              | `architecture/infrastructure_boundaries.md`, `architecture/text_pipeline_ownership.md` |
| Generated          | Machine-produced references; never edit by hand                            | `configuration_schema.json`                                                    |
| Historical/planning| Roadmaps, dated snapshots, closed blockers; not current truth              | `roadmaps/*`, `archive/*`                                                                        |

## Normative documents

| Document                                             | Normative content                                                                 |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| [ADR-0001: package boundaries](adr/0001-package-boundaries.md) | Composition-root definition, architectural invariants, dependency direction rules |
| [ADR-0002: scratch-only model and preprocessing boundary](adr/0002-scratch-only-model-and-preprocessing-boundary.md) | Where model and preprocessing artifacts may live                                   |
| [ADR-0003: local multimodal privacy inspection](adr/0003-local-multimodal-privacy-inspection.md) | Privacy inspection must be local                                                    |
| [ADR-0004: domain-owned configuration architecture](adr/0004-configuration-architecture.md) | Configuration ownership, single load path, merge semantics, policy/tuning separation |
| [product_scope.md](product_scope.md)                 | What the product does and does not do                                              |
| [source_governance.md](source_governance.md)         | Source approval, collection and training gates, robots decision matrix             |
| [misuse_boundaries.md](misuse_boundaries.md)         | Accepted and rejected uses of the system                                           |
| [architecture/dependency_inversion.md](architecture/dependency_inversion.md) | Dependency-inversion rules derived from ADR-0001                                  |
| [architecture/crawl_terminal_outcomes.md](architecture/crawl_terminal_outcomes.md) | Terminal crawl outcomes, retry policy, operator actions                            |
| [architecture/scheduler_retry_semantics.md](architecture/scheduler_retry_semantics.md) | Fetch-attempt, timed-defer, retry-deadline and exhaustion semantics                |

## Architecture decisions

Architecture decision records live in [adr/](adr/). New decisions that change rules
must be an ADR; a decision is not final until it is recorded there.

- [0001: package boundaries](adr/0001-package-boundaries.md)
- [0002: scratch-only model and preprocessing boundary](adr/0002-scratch-only-model-and-preprocessing-boundary.md)
- [0003: local multimodal privacy inspection](adr/0003-local-multimodal-privacy-inspection.md)
- [0004: domain-owned configuration architecture](adr/0004-configuration-architecture.md)

## Operations

| Document                                              | Content                                                          |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| [operations/runbook.md](operations/runbook.md)        | Start/stop/pause/resume, exit codes, diagnosis matrix, recovery  |
| [operations/startup_configuration.md](operations/startup_configuration.md) | Config loading, typed errors, safe fields, required files        |
| [operations/incident-response.md](operations/incident-response.md) | Incident response procedure                                      |
| [golden_path.md](golden_path.md)                      | End-to-end validation workflow                                   |

## Security and governance

| Document                                        | Content                                                        |
| ----------------------------------------------- | -------------------------------------------------------------- |
| [security/threat-model.md](security/threat-model.md) | Threat model and mitigations                                |
| [security/privacy-release-boundary.md](security/privacy-release-boundary.md) | Privacy artifacts release boundary                       |
| [source_governance.md](source_governance.md)     | Source lifecycle, collection/training gates, robots matrix     |
| [model_card_template.md](../training/export/model_card_template.md) | Canonical model card template for release evidence             |
| [task_maturity_matrix.md](task_maturity_matrix.md) | Task readiness classification                                  |

## Generated references

| File                                             | Source                                                        | Command                     |
| ------------------------------------------------ | ------------------------------------------------------------- | --------------------------- |
| [configuration_schema.json](configuration_schema.json) | Pydantic settings models in `config/`                 | Verified by `tests/training/test_training_runtime_blockers.py` |

`configuration_schema.json` is generated. Do not edit it manually. See
[configuration.md](configuration.md) for generation, groups, and review.

## Historical roadmaps

Roadmaps and dated snapshots are never current truth. They are kept for auditability
and moved to `archive/` when superseded.

| Document | Status |
| -------- | ------ |
| [roadmaps/dependency_injection_phase_9.md](roadmaps/dependency_injection_phase_9.md) | Planning roadmap with PR breakdown |
| [archive/fase-9-dependency-injection-2026-07-04.md](archive/fase-9-dependency-injection-2026-07-04.md) | Original monolithic planning document (historical) |
| [archive/dependency_violations_2026-08-01.md](archive/dependency_violations_2026-08-01.md) | Dated dependency-violation snapshot (historical) |
