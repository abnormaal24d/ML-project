# Strict timed-media contract

## Status

Implemented. This document is normative for persisted curated audio and video
records.

## Contract ownership

The only persisted audio/video wire contract lives in:

```text
mmcrawler_datasets/curated/timed_media.py
```

Crawler producers, JSONL writers, snapshot readers, and training consumers must
not define independent audio/video field allowlists. The canonical Pydantic
models are the single source of truth for:

- the complete wire keyset;
- field types and nullability;
- nested transcript, privacy, and asset-lineage structures;
- governance invariants;
- JSON Schema generation; and
- contract digests stored in the snapshot manifest.

The contract package is neutral. It must not import `crawler`, `preprocessing`,
`training`, or `orchestration`.

## Persisted flow

```text
preprocessed media
  -> typed CuratedAudioRecord / CuratedVideoRecord
  -> typed CuratedDatasetWriter
  -> JSONL plus manifest contract digests
  -> fail-closed snapshot reader
  -> training-owned projection
  -> training sample assembly
```

The writer accepts canonical record instances only. Raw mappings are rejected.
Every field in the audio and video JSON Schema is required. Nullable values are
represented by an explicit `null`; a missing key is a contract violation.
Unknown fields and implicit scalar coercions are forbidden.

## Versioning and digest binding

Timed-media records currently use curated schema version `3.0` because the
crawler already published this fieldset under schema 3.0. The repair therefore
formalizes the existing producer contract rather than relabelling all curated
entities with an unrelated global major version.

Exact model identity is bound separately through:

- `curated_audio_contract_sha256`; and
- `curated_video_contract_sha256`.

Both digests are generated from the canonical JSON Schemas and written into the
snapshot manifest. The reader rejects a missing or mismatched digest before it
reads entity rows. Existing snapshots without these digests must be regenerated
or explicitly migrated; they are not partially accepted.

## Failure semantics

Training snapshot consumption is transactional and fail-closed:

- malformed JSON rejects the snapshot;
- an incorrect schema version rejects the snapshot;
- a contract-digest mismatch rejects the snapshot;
- one invalid entity row rejects the snapshot; and
- no partial record tuple is returned.

Diagnostic tooling may implement a separate tolerant inspection mode, but the
training reader must remain fail-closed.

## Semantic invariants

The canonical models enforce at least the following invariants:

- `language` and `transcript_language` agree;
- `trainable` and `curated_media_status` agree;
- trainable media has explicit training permission, a licence, and valid privacy
  clearance;
- metadata-only media includes a rejection reason;
- `asset_context.safety_status` agrees with the top-level safety status;
- transcript ranges are finite, non-negative, and ordered;
- media paths are project-relative on POSIX and Windows path semantics;
- numeric values reject strings, booleans, NaN, and infinity where applicable;
- `context_score` is only populated from a real alignment signal and is never a
  duplicate alias for `quality_score`; and
- persisted video keyframes remain prohibited until each keyframe has
  object-level, byte-bound privacy clearance.

## Training projection

Training consumes the persisted contract through:

```text
mmcrawler_datasets/curated/training_projection.py
```

The projection contains only fields needed by training assembly and converts
persisted privacy evidence to the existing training-domain clearance type. This
keeps the persisted wire schema independent from consumer-specific models.

## Extending the contract

A timed-media field change is complete only when all applicable steps are made
in one change:

1. Add or change the field in the canonical model.
2. Populate it in the typed producer without coercion.
3. Add it to the training projection only when training needs it.
4. Update round-trip and malformed-input tests.
5. Run the producer-to-training end-to-end contract test.
6. Regenerate snapshots because the manifest contract digest will change.

Do not add a second allowlist, decoder dataclass, or free-form producer row.
