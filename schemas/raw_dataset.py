"""Shared raw dataset record wire contract.

Owns the persisted schema that both the crawler producer and the
datachecker consumer agree on.  The full ``DatasetRecord`` in
``crawler.storage.datasets.records`` remains the producer's internal
representation; this module defines the minimal subset that the
independent validator needs to parse and verify raw JSONL payloads.
"""

from __future__ import annotations

from pydantic import BaseModel


class RawDatasetRecord(BaseModel):
    """Minimal validated shape for one raw dataset JSONL row.

    Only the fields that the datachecker inventory reader needs for
    schema validation, deduplication, and modality evidence matching
    are declared here.  The producer ``DatasetRecord`` is a superset.
    Extra fields are allowed to accommodate producer extensions.
    """

    model_config = {"extra": "allow"}

    schema_version: str
    run_id: str
    fetch_record_id: str
    stable_url_id: str
    modality: str
    mime_type: str | None = None
    content_sha256: str
    byte_size: int
    storage_relative_path: str


__all__ = ["RawDatasetRecord"]
