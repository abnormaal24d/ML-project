# Security and Data Incident Response

## Triggers

- PII or license violation detected in published snapshot/release
- Security scanner flags critical in deps or image
- Source takedown / revocation notice
- Detected poisoning / anomalous drift
- Credential leak suspected

## Immediate Actions (first 30 min)

1. Pause the selected deployment: `multimodal-crawler control pause --environment prod --project-root /app` with the production deployment pins already configured for that service.
2. Identify blast radius using lineage: query manifests by source_id / time window.
3. Quarantine affected: move snapshots to quarantine/ (never auto-promote).
4. Rotate any involved secrets.
5. Block source in source_registry (set status=revoked).

## Takedown / Revocation

- Update source_registry.json (status=revoked, reviewer, date)
- Rebuild manifests + snapshots excluding revoked material (or full rebuild)
- File impact report (list of affected releases, models, datasets)
- If public release: follow disclosure + recall process.

## Postmortem

Within 5 business days: root cause, contributing config/code gaps, validation checks added, owners.

## Tabletop

At least quarterly. Record date + participants + scenario.
