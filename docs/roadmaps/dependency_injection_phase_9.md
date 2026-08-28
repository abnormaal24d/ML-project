# Phase 9 — Dependency Injection Execution Plan

Status: Roadmap / Planning (derived from the archived planning document
[../archive/fase-9-dependency-injection-2026-07-04.md](../archive/fase-9-dependency-injection-2026-07-04.md))

Related:
- [architecture/dependency_inversion.md](../architecture/dependency_inversion.md) — normative rules and phase-wide Definition of Done
- [architecture/infrastructure_boundaries.md](../architecture/infrastructure_boundaries.md) — current architecture
- [archive/dependency_violations_2026-08-01.md](../archive/dependency_violations_2026-08-01.md) — dated snapshot with exact locations
- [ADR-0001](../adr/0001-package-boundaries.md)

## Scope

**In scope**

- Fetch execution, sessions, retry, preflight, response parsing, redirects,
  payload assembly (`crawler/fetching/network/**`,
  `orchestration/composition/runtime/build_fetch_services.py`)
- Storage writers, manifests, snapshots, fetch results (clock + ID + atomic
  FS) in `crawler/storage/datasets/writing/`,
  `mmcrawler_datasets/snapshots/`, `datachecker/artifacts/`,
  `crawler/fetching/results/`
- Media decoders, frame sampling, audio/video transcode, image ops (adapters)
  in `preprocessing/media/`, `augmentation/video/`, `augmentation/image/`
- Source registry loading (`crawler/governance/discovery/`, `config/`)
- Scheduler state / checkpoints / recovery stores (`crawler/scheduling/`,
  `crawler/worker/`)
- Dependency probing and optional backends
  (`orchestration/composition/runtime/optional_dependency_validator.py`)
- Composition-root splitting and builder hygiene (`orchestration/composition/**`)
- Import-boundary tests, schema checks, deterministic isolated validation checks

**Out of scope**

- Crawl scheduling algorithms and queue strategy
- Full dataset persistence / collation logic (only the writers/manifests)
- Robots rules details and governance decisions (only the preflight abstraction)
- Metrics collection implementation
- Training loop and model-training internals
- Full end-to-end crawler runs as a replacement for isolated validation checks

## Migration strategy (mandatory)

Never combine interface design, implementation replacement, and business-logic
rewrites in one PR — that is the largest regression source.

Per refactor item (062.x, 063.x, ...), in this order:

1. **Introduce the interface** (e.g. `RetryExecutor`, `FileSystem`, `Clock`)
   in the appropriate package.
2. **Implement the adapter** wrapping the old logic (e.g. `FetchRetryExecutor`
   as implementation of `RetryExecutor`, `SystemClock`).
3. **Update the composition root** in `orchestration/composition/runtime/`
   builders to construct and inject the interface + implementation.
4. **Migrate callers** in domain code to the interface; keep old direct calls
   temporarily supported via the adapter if needed.
5. **Migrate tests** to fakes/mocks of the interface; add import-boundary and
   schema checks; update the automated import-boundary check.
6. **Remove old constructors** and direct `aiohttp`/`datetime`/`Path` calls;
   update `build_*` builders.

Order is always: interface → adapter + composition → migration of calls →
checks → cleanup.

## Rollback strategy

For every large refactor (especially 062, 065, 066):

- Migrate all call sites within the same change; do not add a parallel
  constructor or compatibility route.
- Remove old implementations and direct infra calls only once all call sites
  are migrated, import-boundary tests and relevant isolated validation checks
  are green, and at least one canary/integration test run succeeds.
- On blocking regression: revert the migration PR and keep the old constructors
  until the next iteration. Never "all or nothing".

## PR breakdown

### 062 — Fetch services

| PR    | Content |
| ----- | ------- |
| 062.1 | Inventory of fetch dependencies; update snapshot sections 1 and 4; map `FetchServices` dataclass and `build_fetch_services()` exactly |
| 062.2 | `HttpClientSessionProvider` interface; session creation/lifecycle only via provider from composition; remove aiohttp from public signatures of `Fetcher`, `FetchProfileExecutor`; add mock provider |
| 062.3 | `RetryExecutor` interface; centralize retry + backoff + rate-limit hints in `FetchRetryExecutor`; no `sleep()` outside it; `Fetcher` and `HeadPreflightService` receive retry behavior via interface; deterministic retry checks |
| 062.4 | Response handling: `ResponseBodyReader`, `FetchResponseStatusHandler`, `FetchResponseValidator`, `FetchedPayloadAssembler`, redirect handling, `ResponseStreamWriter`; move aiohttp usage into internal adapters within `crawler/fetching/network/body/` and `response/` |
| 062.5 | Cleanup: remove old constructors and direct aiohttp imports in domain layers; update call sites; clean/split `build_fetch_services()` (see 068); boundary checks green; snapshot updated |

### 063 — Media processing

| PR    | Content |
| ----- | ------- |
| 063.1 | Define interfaces + first adapters (video) |
| 063.2 | Video frame sampling + keyframe + metadata |
| 063.3 | Audio decode + transcription |
| 063.4 | Image / PIL + document paths + raw_inputs |
| 063.5 | Filesystem + `TempDirectoryStrategy` + cleanup |
| 063.6 | Augmentation adapters + checks |

### 064 — Source registry

| PR    | Content |
| ----- | ------- |
| 064.1 | Model + `RegistryReader` interface |
| 064.2 | Reader implementation + migration of `crawler/governance/discovery/*` |
| 064.3 | Injection in composition roots + remove old constructors |

### 065 — Storage (incl. FetchResult & manifests)

| PR    | Content |
| ----- | ------- |
| 065.1 | `Clock` + `IdGenerator` interfaces + `SystemClock` / `FixedClock` + `SequenceIdGenerator`; replace calls in `result.py` and the top-5 storage files from the snapshot |
| 065.2 | `AtomicWriter` (write-to-tmp + atomic rename) |
| 065.3 | `PathLayout` / extend `DatasetPathLayout` + large-scale replacements in writers and run layout |
| 065.4 | Manifest/snapshot determinism + checks + cleanup |

### 066 — Scheduler state

| PR    | Content |
| ----- | ------- |
| 066.1 | `SchedulerStateStore` / `CheckpointStore` interfaces |
| 066.2 | In-memory + file adapters + migration |
| 066.3 | Injection + cleanup in `build_scheduler_services.py` + worker |

### 067 — Dependency detection

| PR    | Content |
| ----- | ------- |
| 067.1 | Probe service + composition |
| 067.2 | Replace `try/except` imports |
| 067.3 | Cleanup |

### 068 — Composition roots

| PR    | Content |
| ----- | ------- |
| 068.1 | Guidelines and review criteria for module size |
| 068.2 | Split fetch builders (builds on 062.5) |
| 068.3 | Split scheduler + runtime + dataset services |
| 068.4 | Split media / processing builders |
| 068.5 | Extract validation and dir creation |

## Risks

| Area                | Risk       |
| ------------------- | ---------- |
| 062 Fetch           | High       |
| 063 Media           | Medium     |
| 064 Registry        | Medium     |
| 065 Storage         | High       |
| 066 Scheduler       | High       |
| 067 Dependency probes | Low      |
| 068 Composition roots | Medium   |

Mitigations: canary runs, recorded sessions, schema checks on interfaces,
golden fetch responses, deterministic fixed-clock validation.

## Next steps

1. Use the snapshot (`dependency_violations_2026-08-01.md`) as the source for
   exact locations, files, and lines.
2. Start with 062.1 + 065.1 (highest validation-stability gain + biggest risk).
   Follow the migration strategy strictly.
3. After every sub-PR, update the snapshot sections, the DoD checklists, and the
   central dependency graph.
4. Add import-boundary checks and builder-size guards where still missing.
5. After each PR run the automated import-boundary check and the relevant
   isolated validation checks (fetch, storage, media).
