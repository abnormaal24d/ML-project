# Source governance rules

## Rules purpose

This rules defines which sources the crawler may collect, which records may be curated, and which records may enter
training snapshots. Collection permission and training permission are separate gates.

A source being reachable on the public web is not sufficient. Every source must be approved in a source registry,
crawled politely, checked for legal and ethical constraints, and revalidated before training use.

## Core rule

A record may enter a training snapshot only if all of the following are true:

1. Source family is approved for the active environment.
2. URL, final URL, redirects, and resolved network targets are within allowed scope.
3. Robots and host-pacing decisions allow collection.
4. License/copyright basis is known and compatible with the intended use.
5. Third-party content exceptions are rejected or separately approved.
6. PII check passes, redacts, or quarantines the record.
7. Lineage, dedupe, quality, modality, and storage-integrity checks pass.
8. Source rules explicitly marks the domain or record as `allow_training=true`.

## Source lifecycle

| Status                  | Meaning                                                                  | May crawl? | May train? |
|-------------------------|--------------------------------------------------------------------------|-----------:|-----------:|
| `proposed`              | Candidate source, not reviewed                                           |         No |         No |
| `approved_collect_only` | Collection allowed for evaluation or indexing, but training not approved |        Yes |         No |
| `approved_training`     | Collection and training allowed under rules gates                       |        Yes |        Yes |
| `paused`                | Temporarily disabled due to robots, terms, quality, abuse, or incident   |         No |         No |
| `revoked`               | Permanently removed or disallowed                                        |         No |         No |

## Approved initial source families

| Source | Purpose                                                               | Current intended status                   | Training notes                                                                                  |
|--------|-----------------------------------------------------------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| NASA   | Public science publications, news, documents, images, and media pages | `approved_training` for allowlisted hosts | Treat third-party and embedded platform assets as exceptions unless separately approved         |
| NOAA   | Public ocean, weather, climate, education, and library content        | `approved_training` for allowlisted hosts | Treat third-party credits and externally hosted assets as exceptions unless separately approved |
| USGS   | Public reports, publications, science pages, and media assets         | `approved_training` for allowlisted hosts | Treat third-party images, maps, and embedded media as exceptions unless separately approved     |

The source of truth is `config/files/sources/source_registry.json`. This document defines the governance schema that
registry entries must satisfy.

## Minimum source-registry fields

Every approved source family should include:

```json
{
  "source_id": "nasa",
  "description": "NASA public science publications, news, and media assets",
  "owner": "data-engineering",
  "status": "approved_training",
  "last_verified_at": "2026-07-01",
  "review_expires_at": "2026-10-01",
  "allowed_hosts": ["www.nasa.gov", "science.nasa.gov"],
  "allowed_asset_hosts": ["assets.science.nasa.gov"],
  "seed_urls": ["https://www.nasa.gov/"],
  "robots_rules_expected": "allowed",
  "copyright_basis": "us-federal-public-domain-with-third-party-exceptions",
  "default_training_allowed": false,
  "governance": [
    {
      "domain": "www.nasa.gov",
      "license": "us-federal-public-domain-with-third-party-exceptions",
      "terms_source": "source-registry",
      "allow_training": true,
      "allow_boilerplate_image_caption": true
    }
  ],
  "disallowed_patterns": ["/login", "/account", "?session="],
  "pii_risk": "low",
  "retention_days": 90,
  "contact_or_rules_url": "https://example.invalid/rules"
}
```

## Collection gates

A URL may be fetched only after it passes all collection gates.

| Gate               | Required behavior                                                               |
|--------------------|---------------------------------------------------------------------------------|
| Scheme gate        | Allow only `http` and `https` unless explicitly approved                        |
| Host gate          | Host must be in source registry or approved asset-host list                     |
| DNS/IP gate        | All resolved IPs must be public and non-reserved                                |
| Redirect gate      | Every redirect hop must rerun all URL, host, DNS/IP, and rules gates           |
| Robots gate        | Robots rules must allow the active user-agent for the requested path           |
| Rate-limit gate    | Per-host and global budgets must permit the request                             |
| Depth/budget gate  | Source depth, page, byte, media, and runtime budgets must not be exceeded       |
| Content-type gate  | Payload type must be allowed for the source and active modality settings        |
| Size/duration gate | Payload must fit configured byte, page-count, media-duration, and memory limits |

## Robots decision matrix

The robots gate runs before every fetch. Decisions come from the robots checker
(`RobotsCheckResult` with decision, confidence, and error classification) and
map to exactly one scheduler outcome.

| Robots/network signal                        | Decision / outcome                                            | Retry                                            |
| -------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| `Allow` (`AUTHORITATIVE_ALLOW`)              | Fetch proceeds                                                 | —                                                |
| `Disallow` (`AUTHORITATIVE_DENY`)            | Permanent governance block; never fetched                     | No                                               |
| 404/410 (no robots file)                     | Treated as allow; fetch                                       | —                                                |
| 429 with `Retry-After`                       | Timed defer honoring `retry_after_seconds` (capped by `max_retry_after_seconds`) | Yes (timed)                      |
| 5xx / DNS error / timeout                    | Timed defer, fail-closed: the host is treated as unavailable  | Yes (timed)                                      |
| Unsafe redirect during robots fetch          | Permanent security block                                      | No                                               |
| Retry deadline exceeded                      | Retry exhausted; task abandoned for this run                  | No                                               |

Notes:

- `robots.txt` is checked pre-fetch; `X-Robots-Tag` headers and `<meta name="robots">`
  are enforced post-fetch on the payload. Both must allow for training use.
- Robots fetches are subject to the same SSRF and redirect-safety rules as
  content fetches (DNS/IP and redirect gates apply).
- Transient robots errors (timeout, 5xx, DNS) are never treated as "no content":
  they produce a timed defer, and if the retry budget is exhausted they surface
  as `incomplete_transient_infrastructure`, not as a governance block.

## Training gates

A collected record may enter training only after it passes all training gates.

| Gate                        | Required behavior                                                               |
|-----------------------------|---------------------------------------------------------------------------------|
| Source permission           | `allow_training=true` for the final domain or record rules                     |
| License compatibility       | License basis is known and compatible with intended training use                |
| Third-party exception check | Embedded/credited third-party assets are rejected unless independently approved |
| PII check                   | PII absent, redacted, or quarantined before snapshot assembly                   |
| Dedupe check                | Duplicate and near-duplicate rules applied                                     |
| Quality check               | Minimum text/media/metadata quality threshold passed                            |
| Lineage check               | Raw object, fetch metadata, hash, source ID, and curation path are complete     |
| Modality check              | Required fields and media artifacts exist and validate                          |
| Split/leakage check         | Train/validation/test split avoids duplicate leakage                            |
| Quota check                 | Source and task quotas are respected to avoid source dominance                  |

## Disallowed content and handling

| Content type                                          | Default action                                                 |
|-------------------------------------------------------|----------------------------------------------------------------|
| Login-only or authenticated pages                     | Reject before fetch or quarantine if discovered after fetch    |
| Paywalled content                                     | Reject                                                         |
| Unknown-license content                               | Collect only if approved for diagnostics; never train          |
| Third-party images or embedded video without approval | Reject or mark `training_allowed=false`                        |
| Personal data                                         | Redact if safe; otherwise quarantine and exclude from training |
| Sensitive personal data                               | Quarantine and exclude                                         |
| Malware, executable payloads, or active content       | Reject                                                         |
| Private network or metadata-service target            | Reject before connection                                       |
| Aggressive crawl target or host under error pressure  | Pause/cooldown                                                 |

## Manifest requirements

Every accepted raw record should carry enough rules information to audit the decision.

```json
{
  "source_id": "nasa",
  "source_rules_version": "1.0",
  "collection_allowed": true,
  "training_allowed": true,
  "license_checked": true,
  "license": "us-federal-public-domain-with-third-party-exceptions",
  "robots_checked": true,
  "robots_decision": "allowed",
  "pii_checked": true,
  "pii_action": "passed",
  "dedupe_checked": true,
  "quality_checked": true,
  "lineage_complete": true,
  "final_url": "https://www.nasa.gov/...",
  "resolved_host": "www.nasa.gov"
}
```

## Review and expiration

- Source rules must be reviewed before initial approval.
- Approved source families should expire on a fixed cadence, for example every 90 days.
- Any change in terms, robots behavior, ownership, host list, or license posture should pause the source until reviewed.
- Revoked or paused sources must not contribute new records to curated or training snapshots.
- Existing snapshots should preserve historical rules metadata for auditability.

## Acceptance criteria

This source rules is production-ready when:

- every active source has status, owner, review timestamp, allowed hosts, license basis, and training permission fields;
- collection and training permissions are represented separately;
- training snapshot assembly refuses records without explicit training permission;
- manifests contain source-rules decision metadata;
- source changes are testable through config validation or a dedicated registry validator;
- a reviewer can trace every training record back to source, URL, license, PII decision, and approval basis.
