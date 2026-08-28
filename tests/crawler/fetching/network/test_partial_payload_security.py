from __future__ import annotations

import os
from pathlib import Path

import pytest

from crawler.fetching.network.body.partial_store import (
    PartialPayloadSecurityError,
    PartialPayloadStorage,
)


def _preserved_partial(storage: PartialPayloadStorage) -> tuple[Path, str]:
    descriptor, payload = storage.create_temp_file()
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"partial")
        handle.flush()
        os.fsync(handle.fileno())
    partial = storage.finalize_incomplete_payload(
        path=payload,
        reason="stream_interrupted",
        url="https://example.test/object?secret=never-store-this",
        status_code=200,
        content_length=20,
        observed_bytes=7,
        chunk_count=1,
        max_bytes=20,
        etag='"immutable"',
    )
    assert partial is not None
    metadata = storage.read_metadata(
        metadata_path=partial.with_name(f"{partial.name}.json")
    )
    token = metadata.get("owner_token")
    assert isinstance(token, str)
    return partial, token


def test_partial_sidecar_is_owned_and_does_not_store_url_secrets(
    tmp_path: Path,
) -> None:
    storage = PartialPayloadStorage(
        temporary_directory=tmp_path / "owned",
        preserve_partial_files=True,
    )
    partial, token = _preserved_partial(storage)
    sidecar = partial.with_name(f"{partial.name}.json")

    assert "secret=never-store-this" not in sidecar.read_text(encoding="utf-8")
    with pytest.raises(PartialPayloadSecurityError, match="owner token"):
        storage.discard_partial(path=partial, owner_token="wrong")
    assert partial.is_file()

    storage.discard_partial(path=partial, owner_token=token)
    assert not partial.exists()
    assert not sidecar.exists()


def test_resume_rejects_metadata_and_file_length_mismatch(
    tmp_path: Path,
) -> None:
    storage = PartialPayloadStorage(
        temporary_directory=tmp_path,
        preserve_partial_files=True,
    )
    partial, token = _preserved_partial(storage)
    with partial.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(PartialPayloadSecurityError, match="length"):
        storage.validate_resume(path=partial, owner_token=token)


def test_cleanup_cannot_escape_the_temporary_directory(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    storage = PartialPayloadStorage(temporary_directory=owned)
    external = tmp_path / "crawler_body_external"
    external.write_bytes(b"must-remain")

    with pytest.raises(PartialPayloadSecurityError, match="escapes"):
        storage.delete(path=external)

    assert external.read_bytes() == b"must-remain"
