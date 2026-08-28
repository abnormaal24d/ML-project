# Infrastructure Boundaries

Status: Architecture reference
Describes the current architecture and the known dependency-inversion
violations. Updated as the phase-9 execution plan progresses; the dated
snapshot is archived at [../archive/dependency_violations_2026-08-01.md](../archive/dependency_violations_2026-08-01.md).

Related:
- [dependency_inversion.md](dependency_inversion.md) — normative rules
- [roadmaps/dependency_injection_phase_9.md](../roadmaps/dependency_injection_phase_9.md) — execution plan

## Composition overview

```
orchestration
        │
        ▼
build_fetch_services()   [orchestration/composition/runtime/build_fetch_services.py]
        │
        ▼
FetchServiceGraph (FetchServices dataclass)

        │
        ├── HttpClientSessionProvider          [crawler/fetching/network/session.py]
        ├── FetchRetryExecutor                 [crawler/fetching/execution/attempt.py]
        ├── FetchProfileExecutor               [crawler/fetching/execution/attempt.py]
        ├── ResponseBodyReader                 [crawler/fetching/network/body/reader.py]
        ├── FetchResponseStatusHandler         [crawler/fetching/response/status_rules.py]
        ├── FetchResponseValidator             [crawler/fetching/response/validator.py]
        ├── HeadPreflightService               [crawler/fetching/network/preflight/executor.py]
        ├── RetryManager                       [crawler/governance/retry/retry_manager.py]
        ├── RateLimiter
        ├── RedirectValidator
        └── ...
```

## Allowed infrastructure implementations

These components may use concrete infrastructure because they live in the
infrastructure layer (`crawler/fetching/network/` or equivalent):

- `HttpClientSessionProvider` (`crawler/fetching/network/session.py`)
- `ResponseBodyReader` (`crawler/fetching/network/body/reader.py`)
- `FetchRetryExecutor` (`crawler/fetching/execution/attempt.py`)
- `FetchProfileExecutor`, `HeadPreflightService` (provided their public API
  does not leak concrete aiohttp types)

These belong to the infrastructure layer and are exposed to higher layers only
through interfaces.

## Known unwanted infrastructure leaks

The real violations to be removed (tracked per area; see the archive snapshot
for exact line numbers):

- `FetchResult` (`crawler/fetching/results/result.py`) → `datetime.now()`
- `HostProfilePreferenceStore` → `datetime.now()`
- Domain services (`Fetcher`, `HeadPreflightService`, response assemblers) →
  direct `aiohttp.ClientSession` / `ClientResponse` in signatures or runtime
- Response assemblers and body readers leaking `ClientResponse` to callers
  outside the network layer
- Storage writers, manifests, snapshots, fetch results → direct
  `Path().open/mkdir`, `tempfile`, `uuid.uuid4()`
- `crawler/extraction/payloads/video_payload_extractor.py`,
  `preprocessing/privacy/inspection/local_visual_analysis.py`,
  `augmentation/image/*`, `augmentation/document/document_page_augmenter.py` →
  direct `cv2`, `PIL` (outside adapters)
- `RetryManager` called directly from domain fetch logic instead of through the
  `RetryExecutor` interface

For reviewers: not everything that touches aiohttp or datetime is a leak. It is
a leak only when it appears in domain code or public domain-service APIs, or
when the infrastructure layer depends on higher layers.

## Per-area current situation

### Fetch services (area 062)

The `crawler/fetching/network/` package already contains the primary network
abstractions; the composition root is `build_fetch_services()` which builds the
`FetchServices` dataclass with 15+ components. Remaining violations concentrate
in `crawler/fetching/results/result.py`, `crawler/fetching/network/`,
`crawler/fetching/fetcher.py` (direct dependency on `RetryManager`), and
`crawler/fetching/profiles/host_preferences.py`.

### Media processing (area 063)

Direct or lazy native-library imports outside adapters exist in
`crawler/extraction/payloads/video_payload_extractor.py`,
`preprocessing/privacy/inspection/local_visual_analysis.py`,
`preprocessing/media/adapters/audio_decode.py`,
`crawler/analysis/enrichment/speech/speech_transcriber.py`,
`augmentation/video/*`, `augmentation/image/*`, and
`crawler/curation/snapshots/alignment_rows.py`. Direct `Path`/`tempfile`
usage exists in `crawler/analysis/enrichment/video/video_frame_sampler.py`,
`augmentation/cache.py`, and the `crawler/storage/datasets/writing/` group.

### Storage (area 065)

`datetime.now(timezone.utc)` in `crawler/fetching/results/result.py`,
`crawler/storage/datasets/writing/dataset_writer.py`,
`crawler/storage/datasets/records/failed_task_record_assembler.py`,
`mmcrawler_datasets/snapshots/*`, `training/*`, `augmentation/cache.py`,
`datachecker/artifacts/manifest_file_writer.py`; `uuid.uuid4()` in
`raw_payload_writer.py`; direct `Path`/`.mkdir()`/`.open()` in
`raw_payload_writer.py`, `dataset_manifest_writer.py`,
`dataset_path_layout.py`.

### Source registry (area 064)

Direct `Path`, `open()`, `toml.load` in `crawler/governance/discovery/`,
`crawler/curation/`, `config/source_catalog/registry_settings.py`,
`orchestration/composition/governance/`.

### Scheduler state (area 066)

Checkpoint stores, dead-letter queues, and hidden caches in
`crawler/scheduling/`, `crawler/worker/`,
`orchestration/composition/runtime/build_scheduler_services.py`,
`crawler/worker_scaling/`.

### Dependency detection (area 067)

`orchestration/composition/runtime/optional_dependency_validator.py` plus
scattered `try: import cv2` / `import av` in `crawler/extraction/payloads/*`,
`preprocessing/privacy/inspection/*`, `augmentation/image/*` (only `av` remains
in `preprocessing/media/adapters/pyav_media.py`).

### Composition roots (area 068)

`build_fetch_services.py`, `build_task_processor.py`,
`build_scheduler_services.py`, `build_crawler_runtime_services.py`,
`build_dataset_services.py`, `build_worker_services.py` are large and contain
validation and `mkdir` logic; they should only wire and create objects.
