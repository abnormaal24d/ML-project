# Threat Model

Based on misuse boundaries (see docs/misuse_boundaries.md).

## Assets

- Source allow-lists and governance state
- Raw/curated/training datasets and manifests (lineage, licenses, PII flags)
- Model checkpoints and evaluation reports
- Crawler runtime (workers, storage, secrets)

## Actors & Trust Boundaries

- External internet hosts (untrusted)
- Approved source operators (partially trusted)
- Internal operators / CI (trusted)
- Malicious submitter of crafted content (untrusted)

## High Risk Threats (with controls)

1. SSRF / DNS rebinding / redirect to private metadata (mitigated by URL governance + security checks + fail-closed DNS)
2. Archive traversal / zip bombs (crawler archive ingestion is disabled; dataset tar members are read without filesystem extraction)
3. Unsafe pickle / tensor load (weights_only=True + hash verification + no untrusted checkpoints)
4. Data poisoning via compromised approved source (source revocation + lineage + drift detection)
5. PII leakage into training (PII detector + quarantine + dataset card review)
6. Credential / secret exposure (no secrets in TOML/logs/CLI; vault only)
7. Resource exhaustion (per-payload byte/pixel/time limits + container limits)
8. Supply chain (SBOM + pip-audit + image scan + reproducible builds)

Every threat has: owner, code control(s), verification evidence, monitor/alert, and incident runbook reference.

Security controls are verified through reviewable runtime evidence and automated rules checks.
