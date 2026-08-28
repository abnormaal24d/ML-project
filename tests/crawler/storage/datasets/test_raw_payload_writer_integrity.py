"""Integrity checks for content-addressed raw payload reuse."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from config.settings.datasets import (
    DatasetPathSettings,
    RawDatasetWriterSettings,
)
from crawler.classification.media_kind import MediaKind
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from crawler.storage.datasets.writing import raw_payload_writer
from crawler.storage.datasets.writing.raw_payload_writer import (
    RawPayloadWriter,
)


def _writer(*, run_directory: Path) -> RawPayloadWriter:
    return RawPayloadWriter(
        settings=RawDatasetWriterSettings(),
        dataset_paths=DatasetPathSettings(),
        run_directory=run_directory,
    )


def _result(*, payload_path: Path, body: bytes) -> FetchResult:
    digest = hashlib.sha256(body).hexdigest()
    return FetchResult(
        url="https://example.test/object.bin",
        final_url="https://example.test/object.bin",
        status_code=200,
        headers={},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="application/octet-stream",
        mime_type="application/octet-stream",
        encoding=None,
        language=None,
        kind=MediaKind.DOCUMENT,
        payload=FetchedPayload(
            temp_path=payload_path,
            byte_size=len(body),
            sha256_hex=digest,
            sniff_bytes=body[:64],
            chunk_count=1,
        ),
        body_sha256=digest,
    )


def _prepared_write(
    *,
    tmp_path: Path,
) -> tuple[RawPayloadWriter, FetchResult, Path, Path, bytes]:
    body = b"correct content-addressed payload"
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    source = tmp_path / "incoming.bin"
    source.write_bytes(body)
    writer = _writer(run_directory=run_directory)
    result = _result(payload_path=source, body=body)
    relative_path, _digest, _size, inline_body = writer.prepare(
        result=result,
        kind="document",
        modality="document",
    )
    assert inline_body is None
    return (
        writer,
        result,
        relative_path,
        writer.absolute_path(relative_path=relative_path),
        body,
    )


@pytest.mark.parametrize(
    "corrupt_body",
    (b"wrong", b"x" * len(b"correct content-addressed payload")),
)
def test_corrupt_existing_object_is_replaced_not_reused(
    tmp_path: Path,
    corrupt_body: bytes,
) -> None:
    writer, result, relative_path, object_path, body = _prepared_write(
        tmp_path=tmp_path,
    )
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(corrupt_body)

    writer.persist_prepared(
        relative_path=relative_path,
        result=result,
        inline_body=None,
    )

    assert object_path.read_bytes() == body
    assert result.payload is not None
    assert not result.payload.temp_path.exists()


def test_verified_existing_object_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, result, relative_path, object_path, body = _prepared_write(
        tmp_path=tmp_path,
    )
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(body)

    def _unexpected_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("verified duplicate should not be copied")

    monkeypatch.setattr(
        raw_payload_writer,
        "_write_temp_payload_durable",
        _unexpected_write,
    )

    writer.persist_prepared(
        relative_path=relative_path,
        result=result,
        inline_body=None,
    )

    assert object_path.read_bytes() == body
    assert result.payload is not None
    assert not result.payload.temp_path.exists()


def test_non_regular_existing_object_is_not_discarded_as_a_duplicate(
    tmp_path: Path,
) -> None:
    writer, result, relative_path, object_path, _body = _prepared_write(
        tmp_path=tmp_path,
    )
    object_path.mkdir(parents=True)

    with pytest.raises(OSError, match="not a replaceable regular file"):
        writer.persist_prepared(
            relative_path=relative_path,
            result=result,
            inline_body=None,
        )

    assert result.payload is not None
    assert result.payload.temp_path.exists()


def test_temporary_sibling_uses_a_short_name_for_hashed_destinations(
    tmp_path: Path,
) -> None:
    """Do not repeat content hashes in a same-directory temporary filename."""

    destination = tmp_path / ("a" * 64 + ".html")
    temporary = raw_payload_writer._create_temporary_sibling(destination)

    try:
        assert temporary.parent == destination.parent
        assert temporary.name.startswith(".tmp-")
        assert temporary.name.endswith(".tmp")
        assert destination.name not in temporary.name
    finally:
        temporary.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "relative_path",
    (Path("."), Path(".."), Path(r"C:\\escape")),
)
def test_absolute_path_rejects_noncanonical_relative_paths(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    writer = _writer(run_directory=run_directory)

    with pytest.raises(ValueError, match="relative_path"):
        writer.absolute_path(relative_path=relative_path)
