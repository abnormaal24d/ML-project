"""Workflow manifest schema checks at the DataChecker boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datachecker.data_checker import DataChecker


@dataclass(frozen=True)
class _Manifest:
    value: str

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> _Manifest:
        return cls(value=str(payload["value"]))


def _read_manifest(path: Path) -> tuple[_Manifest | None, str | None]:
    return DataChecker._read_manifest(
        path=path,
        parser=_Manifest.from_payload,
        checkpoint=lambda _stage: None,
        stage="workflow_manifest_schema_test",
    )


def test_workflow_manifest_reader_rejects_unsupported_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"manifest_schema_version": "1.0", "value": "x"}),
        encoding="utf-8",
    )

    manifest, error = _read_manifest(path)

    assert manifest is None
    assert error == "ValueError"


def test_workflow_manifest_reader_accepts_current_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"manifest_schema_version": "2.0", "value": "x"}),
        encoding="utf-8",
    )

    manifest, error = _read_manifest(path)

    assert error is None
    assert manifest == _Manifest(value="x")
