# Privacy release boundary

## Invariant

A training-eligible object must be bound to the exact output bytes or released
text values that were inspected by the current preprocessing run. Generic
payload metadata cannot declare a privacy check complete.

## Active multimodal flow

All modalities use one `DetectorRegistry` assembled by orchestration and shared
with `PiiDetector`:

| Modality | Local content construction | Inspector |
|---|---|---|
| Image | exact bytes, local decode, injected OCR, local OpenCV observations | `inspect_image` |
| Audio | exact bytes and local decode; semantic backends must be locally available | `inspect_audio` |
| Video | exact bytes, exhaustive bounded frame decode, injected frame OCR and local visual observations | `inspect_video` |
| Document | exact PDF bytes/page extraction or exact normalized document text | `inspect_document` |

The inspector returns `InspectionResult`. The release boundary independently
recomputes the file SHA-256 and accepts the result only when:

1. the inspection subject digest matches the exact current bytes;
2. every modality-required evidence field is checked;
3. temporal/page/frame coverage is complete;
4. no detector failure exists;
5. policy-specific findings are rejected or remediated;
6. any derivative is freshly generated and locally rescanned;
7. the final `ApprovedObject` digest equals the released bytes.

## Untrusted inputs

The following values are treated only as ordinary input metadata and never as
proof of inspection:

- `privacy_analysis`;
- `privacy_residual_analysis`;
- detector names or completion flags in enrichment dictionaries;
- caller-selected sanitized or normalized media paths.

The metadata flattener excludes `privacy_*` keys from released privacy fields.

## Missing backends

The boundary is fail-closed. For example, disabled image OCR or unavailable
audio semantic analysis leaves required evidence unchecked and sends the item
to quarantine. It does not infer completion from existing transcript/OCR text
or enrichment flags.

## Remediation

Image masks and metadata-clean derivatives use deterministic application-owned
sibling paths. Existing destination files are replaced only after a fresh
successful temporary write. The derivative bytes are inspected again; stale or
caller-provided files cannot satisfy residual verification.

## Regression requirements

Release-blocking tests verify that:

- a `totally-untrusted` / `banana` payload cannot approve media;
- subject digest mismatch fails closed;
- identity documents are rejected from local findings;
- local face findings create a new derivative and residual scan;
- attacker-selected destination paths are ignored;
- incomplete audio coverage and unauthorized identifiable voice are rejected;
- production call sites invoke every modality inspector;
- no active preprocessing source reads `privacy_analysis` as evidence.
