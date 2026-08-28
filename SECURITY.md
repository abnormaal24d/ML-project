# Security Rules

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

- Use the repository's **Security → Advisories → Report a vulnerability**
  workflow. GitHub Private Vulnerability Reporting is the canonical reporting
  channel; do not send secrets or exploit details through a public issue.
- A production release is blocked unless repository administrators have
  enabled Private Vulnerability Reporting and verified the reporting workflow.
- Include the affected component or URL, reproduction steps or a minimal PoC
  (without live exploitation), impact, affected versions, and a suggested fix
  when available.
- The repository administrators own intake. The security triage role owns
  severity and disclosure decisions; the affected package owner owns the fix
  and regression verification.
- Response targets are initial triage within five business days and within 48
  hours for suspected high or critical severity. We coordinate disclosure and
  reporter credit through the private advisory.

This project processes untrusted web content (HTML, archives, media) and therefore treats SSRF, path traversal, unsafe
deserialization, decompression bombs, DNS rebinding and supply-chain risks as P0.

If the private reporting button is unavailable, do not publish the report. Ask
the repository owner to enable Private Vulnerability Reporting; the project is
not eligible for production release while that channel is unavailable.

## Privacy release boundary

The privacy release invariant, multimodal evidence schema, remediation
requirements, and fail-closed training checks are documented in
`docs/security/privacy-release-boundary.md`.
