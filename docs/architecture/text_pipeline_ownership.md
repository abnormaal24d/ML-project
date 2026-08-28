# Text pipeline ownership

The project separates crawl-time evidence, preprocessing release checks, and
snapshot-wide dataset decisions.

## Crawler

The crawler owns URL normalization, HTML/page extraction, extraction sidecars,
raw payload hashes, crawl admission, URL scheduling deduplication, and initial
language classification. The raw record persists the language value together
with its confidence, source, and detector contract version.

## Preprocessing

Preprocessing consumes crawler-owned extracted text and does not reparse HTML or
renormalize URLs. It owns Unicode/whitespace normalization, final privacy and
PII release inspection, language verification or fallback, training-oriented
quality scoring, the exact key of the final released text, and construction of
`PreprocessedDocument`.

Preprocessing deliberately does not drop duplicate URLs or duplicate text and
does not assign near-duplicate clusters. This keeps the stage reentrant and
prevents local batches from making snapshot-wide selection decisions.

## Curation and dataset assembly

Curation selects the best valid candidate per normalized URL, removes exact
released-text duplicates, assigns deterministic near-duplicate clusters across
the snapshot, applies per-domain limits, and then builds curated records.
Chunking and split assignment are dataset-assembly responsibilities. Split keys
prefer the near-duplicate cluster, then the exact content key, so related text
cannot leak across train, validation, and test splits.
