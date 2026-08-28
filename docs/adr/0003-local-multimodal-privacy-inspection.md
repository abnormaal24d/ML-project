# ADR-0003: Local multimodal privacy inspection

## Status

Accepted.

## Context

A previous media boundary accepted a caller-supplied `privacy_analysis`
dictionary as evidence that OCR, visual, audio, or video checks had completed.
The dictionary was structurally validated and byte-bound, but its producer and
execution were not trusted. A well-formed self-declaration could therefore
influence training clearance.

## Decision

Image, audio, video, and document preprocessing use **Option A**: inspection is
executed in the preprocessing process through the shared `DetectorRegistry`.

- `ImagePreprocessor` builds `ImageContent` from exact local bytes and calls
  `inspect_image`.
- `AudioPreprocessor` builds `AudioContent` from exact local bytes and calls
  `inspect_audio`.
- `VideoPreprocessor` builds `VideoContent` from exact local bytes and calls
  `inspect_video`.
- `TextInputPreparer` builds `DocumentContent` and calls `inspect_document` for
  document inputs.
- The same registry instance is shared with `PiiDetector`.
- `privacy_analysis`, `privacy_residual_analysis`, and caller-selected
  sanitization paths have no authority at the release boundary.
- Inspection completion, coverage, detector failures, and the subject digest
  are derived locally.
- Missing local OCR, ASR, voice, tracking, decode, or other required backends
  produce incomplete evidence and quarantine; they are never converted into
  successful checks.
- Image remediations are written to an application-derived sibling path and
  are rescanned from the new bytes before clearance.

Concrete media/OCR implementations are injected at composition time through
preprocessing-owned protocols. This preserves package boundaries while keeping
backend selection outside the privacy policy.

## Consequences

The pipeline can no longer be approved by constructing a metadata dictionary.
Deployments must provide every backend needed by the modality they intend to
release. A modality with incomplete local capability remains fail-closed. This
is intentionally preferable to accepting unverified upstream assertions.

An external signed-attestation service is not part of this architecture. If it
is introduced later, it requires a separate ADR and may not silently restore
raw dictionary evidence.
