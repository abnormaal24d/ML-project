# Preprocessing privacy

This package contains only privacy logic that is executed by the text, image,
audio, document, or video preprocessing pipelines.

## Scope

The retained flow is deliberately small:

```text
content detection
    -> deterministic privacy classification
    -> text or media sanitization when supported
    -> residual inspection
    -> exact-output clearance or fail-closed rejection
```

The package does not provide a separate privacy application, HTTP or CLI API,
audit repository, authorization service, attestation chain, retention worker,
human-review workflow, monitoring subsystem, reprocessing scheduler, or
training-specific evaluator.

## Retained responsibilities

- deterministic text and structured-identifier detection;
- credential and secret detection;
- contextual sensitive-information detection;
- visual-region detector adapters;
- field-level release inspection;
- exact input/output digest binding through `PrivacyClearance`;
- image-region masking used by image preprocessing;
- fail-closed handling of incomplete inspection and failed remediation.

All preprocessors share the same `PiiDetector` and detector registry. Findings
that can be safely redacted are remediated and inspected again. Secrets,
restricted identifiers, sensitive categories, incomplete coverage, and
unverified media derivatives are rejected or quarantined by the preprocessing
result.
