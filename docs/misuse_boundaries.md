# Misuse Boundaries

## Purpose

This document states what the crawler must not do. These boundaries are product requirements, not optional safety notes.

The crawler is designed for governed public-source data collection, not unrestricted scraping or network exploration.

## Absolute prohibitions

The crawler must never intentionally:

- access private, loopback, link-local, multicast, reserved, carrier-grade NAT, or cloud metadata network targets;
- bypass authentication, paywalls, session checks, CAPTCHAs, rate limits, robots.txt, or access controls;
- crawl arbitrary user-supplied domains without source-registry approval;
- collect login-only, private, personal, medical, financial, or confidential records for training;
- use unknown-license material in training snapshots;
- hide or rotate identity to evade host restrictions;
- continue aggressive crawling when a host returns errors, throttling, or block signals;
- execute downloaded active content;
- include runtime data, logs, secrets, caches, or fetched media in source release artifacts.

## Network misuse boundaries

| Boundary                | Required enforcement                                                          |
|-------------------------|-------------------------------------------------------------------------------|
| Private networks        | Reject before connection and after every DNS/redirect step                    |
| Cloud metadata services | Explicitly block metadata IPs and hostnames                                   |
| DNS rebinding           | Resolve and validate at connection time; reject if any resolved IP is blocked |
| Redirect abuse          | Re-run all gates for every redirect target                                    |
| Scheme abuse            | Allow only approved schemes, normally `http` and `https`                      |
| Host spoofing           | Normalize IDNA/punycode and reject userinfo tricks or malformed hosts         |
| Proxy abuse             | Do not inherit unsafe proxy settings without explicit config approval         |
| Port abuse              | Reject disallowed ports unless source rules explicitly permits them          |

## Web-rules misuse boundaries

| Boundary                   | Required enforcement                                                            |
|----------------------------|---------------------------------------------------------------------------------|
| Robots.txt                 | Respect robots decisions for the configured user-agent                          |
| Crawl-delay and throttling | Respect crawl-delay when available and use conservative host pacing             |
| Retry-After                | Respect `Retry-After`, `429`, and `503` signals                                 |
| Error pressure             | Cool down or pause hosts with repeated errors                                   |
| Crawl budget               | Enforce per-host and per-run limits on pages, bytes, depth, duration, and media |
| Identity                   | Use a clear user-agent suitable for governed data collection                    |

## Data misuse boundaries

| Boundary                        | Required enforcement                                |
|---------------------------------|-----------------------------------------------------|
| Unknown license                 | Exclude from training                               |
| Third-party embedded content    | Exclude unless independently approved               |
| Personal data                   | Redact or quarantine; never train by default        |
| Sensitive personal data         | Quarantine and exclude                              |
| Biometric identity tasks        | Disabled unless explicit governance approval exists |
| Paywalled/login-only content    | Reject                                              |
| Malformed or dangerous payloads | Reject or quarantine                                |
| Duplicate/leaky samples         | Exclude from affected splits or snapshots           |

## ML misuse boundaries

The baseline training path must not be presented as proof of model quality. It validates dataflow, routing,
checkpointing, and evaluation plumbing.

The project must not claim production model quality until:

- target tasks are defined;
- labeled evaluation datasets exist;
- metric thresholds are configured;
- train/validation/test leakage checks pass;
- model and dataset cards are written;
- evaluation reports are reproducible;
- failure modes and misuse risks are documented.

## Release misuse boundaries

Source releases must not contain:

- `data/` runtime datasets;
- `runtime/` outputs;
- terminal logs;
- `.env` secrets;
- `.pyc` or `__pycache__`;
- `.sqlite3` runtime databases;
- crawl JSONL output;
- downloaded PDFs, MP3s, MP4s, images, checkpoints, or model weights;
- temporary or backup files.

A release artifact verifier should fail if any forbidden artifact is present.

## User-supplied source rules

If a user wants to add a new source, the default answer is not “crawl it.” The source must first become a registry entry
with:

- owner;
- intended source status;
- allowed hosts and asset hosts;
- seed URLs;
- robots expectation;
- copyright/license basis;
- collection and training permission;
- disallowed patterns;
- PII risk;
- review and expiration dates.

Until then, the source is `proposed` and may not be crawled in production.

## Incident response triggers

The system should pause or fail closed when any of the following occur:

- a URL resolves to a blocked network;
- a redirect leaves approved scope;
- robots denies the active user-agent;
- license/training permission is unknown;
- PII is detected and cannot be safely redacted;
- host error rate or throttling exceeds thresholds;
- manifest finalization fails;
- release artifact verifier detects runtime/data files;
- source rules expires or is revoked.

## Acceptance criteria

Misuse boundaries are production-ready when:

- every absolute prohibition maps to code enforcement or a failing acceptance check;
- SSRF, redirect, DNS, robots, rate-limit, PII, license, and release-artifact checks exist;
- training snapshot assembly refuses records that lack rules proof;
- logs/manifests explain rejection and quarantine decisions;
- defaults fail closed rather than permissive.
