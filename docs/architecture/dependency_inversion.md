# Dependency Inversion Rules

Status: Normative
Source of truth for: the dependency-inversion rules all product code must
satisfy. Derived from ADR-0001 (package boundaries and composition roots).

Related:
- [ADR-0001](../adr/0001-package-boundaries.md) — composition-root definition and architectural invariants
- [infrastructure_boundaries.md](infrastructure_boundaries.md) — current architecture and violation inventory
- [roadmaps/dependency_injection_phase_9.md](../roadmaps/dependency_injection_phase_9.md) — execution plan

## Core rule

Infrastructure (network, clock, filesystem, media decoders, identifiers) is
created in exactly one place: the composition roots under
`orchestration/composition/`. Domain services in `crawler/`, `preprocessing/`,
`datasets/`, `datachecker/`, `augmentation/`, and `training/` depend on explicit
interfaces and never construct concrete infrastructure themselves.

`shared/runtime_primitives.py` is the single owner of the application-wide
`Clock` and `IdGenerator` contracts. Their concrete implementations remain in
`orchestration/composition/adapters/`; package-specific ports remain owned by
their consumers.

Concretely, no domain service may:

- create HTTP clients, sessions, or responses, or accept them in signatures;
- call `Path(...).open()`, `.mkdir()`, `tempfile`, or perform direct file
  mutations;
- obtain wall-clock timestamps or semantic identifiers inside domain/evidence
  construction instead of receiving their values explicitly;
- import `cv2`, `av`, `PIL`, or invoke ffmpeg/subprocess outside explicit
  adapters;
- hold hidden singleton state, module-level caches, or globals for retry,
  sessions, or dependencies;
- make infrastructure (network, clock, FS, media decoders, IDs) non-replaceable
  by test implementations (mocks, fakes, in-memory).

## Allowed and forbidden dependency directions

| Direction                        | Allowed |
| -------------------------------- | ------- |
| `orchestration` → `crawler`      | Yes     |
| `orchestration` → `preprocessing`| Yes     |
| `orchestration` → `datasets`     | Yes     |
| `crawler` → `orchestration`      | No      |
| `training` → `crawler.runtime`   | No      |
| `preprocessing/media` → `crawler.runtime`, `orchestration` | No |
| `datasets` → `orchestration`     | No      |
| Domain/evidence code → wall-clock time or semantic-ID generation | No; composition passes explicit values |
| Deadline/timeout code → monotonic elapsed time | Yes; inject a monotonic callable when deterministic timeout tests matter |
| Filesystem/security adapters → `tempfile`, `secrets.token_hex()` for temporary paths, locks, atomic publication, or collision avoidance | Yes; this entropy must not be made deterministic |

## Infrastructure implementations

Infrastructure components that live in the correct layer may use concrete
libraries: `HttpClientSessionProvider`
(`crawler/fetching/network/session.py`), `FetchRetryExecutor`
(`crawler/fetching/execution/attempt.py`), `ResponseBodyReader`
(`crawler/fetching/network/body/reader.py`), and media adapters
(`preprocessing/media/adapters/`). The privacy artifact workspace is likewise a
filesystem/security boundary and may own cryptographic temporary-path entropy.
Their existence is not a violation; leaking
their concrete types into domain signatures or domain runtime is.

The dependency direction is the problem, not the presence of backoff code
inside an executor.

## Phase-wide Definition of Done

A PR or phase is done only when all items hold:

- No direct `aiohttp.ClientSession` / `ClientResponse` outside
  `crawler/fetching/network/session.py`, its providers, and
  `orchestration/composition/runtime/build_fetch_services.py`.
- No hidden wall-clock time or semantic-ID generation in domain decisions or
  persisted reproducibility evidence. Composition obtains those values from
  `Clock` / `IdGenerator` and passes the resulting values explicitly.
- Timeout and duration measurement uses monotonic time, never wall-clock time.
- Security and operational entropy used only for temporary paths, locks,
  atomic publication, and collision avoidance remains owned by infrastructure
  adapters through `tempfile` or `secrets`; it is not routed through a
  deterministic ID generator.
- No direct `Path(...).open/mkdir`, `tempfile`, `os.open` in domain code
  (except inside `AtomicWriter` / `PathLayout` adapters).
- No `import cv2`, `import av`, `from PIL`, direct ffmpeg calls outside
  `preprocessing/media/adapters/`, `augmentation/*/adapters/`.
- All fetch validation checks in `crawler/fetching/` run without sockets.
- All storage/snapshot/manifest validation checks are deterministic.
- Media validation checks run without OpenCV/PyAV/FFmpeg/PIL (via fakes).
- Composition builders in `orchestration/composition/runtime/build_*_services.py`
  are split and explicit.
- ADR-0001 package boundaries are not violated (import-boundary checks green).
- Builder size and fragmentation checks green.
- Migration strategy followed (interface → adapter → composition → migration →
  checks → cleanup).
