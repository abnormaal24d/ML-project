"""Regression tests for the privacy artifact transaction boundary."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from preprocessing.privacy.artifacts import (
    DerivationReceipt,
    PrivacyArtifactWorkspace,
    PublishedPrivacyArtifact,
    build_receipt,
    canonical_sha256,
    file_sha256,
    verify_published_artifact,
    write_exclusive_bytes,
)

_CREATED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


def _publish(tmp_path: Path) -> tuple[PublishedPrivacyArtifact, Path]:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")
    workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="sanitized",
        run_id="publish-run",
    )
    temporary = workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(temporary, b"sanitized bytes")
    output_digest = file_sha256(temporary)
    receipt = build_receipt(
        workspace=workspace,
        source_path=source,
        source_sha256=file_sha256(source),
        transform_input_sha256=file_sha256(source),
        output_path=temporary,
        output_sha256=output_digest,
        source_mime_type="application/octet-stream",
        output_mime_type="application/octet-stream",
        transform_id="test-transform",
        transform_version="1",
        transform_artifact_path=Path(__file__),
        configuration={"mode": "test"},
        residual_inspection_sha256=canonical_sha256({"clean": True}),
        created_at=_CREATED_AT,
    )
    artifact = workspace.publish(
        temporary_path=temporary,
        receipt=receipt,
        final_name="source.sanitized.bin",
    )
    workspace.cleanup()
    return artifact, source


def test_workspaces_are_unique_and_private(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")

    first = PrivacyArtifactWorkspace(
        source_path=source,
        stage="clean",
        run_id="first-run",
    )
    second = PrivacyArtifactWorkspace(
        source_path=source,
        stage="clean",
        run_id="second-run",
    )
    try:
        assert first.directory != second.directory
        assert first.run_id != second.run_id
        if os.name == "nt":
            assert first.directory.is_dir()
            assert second.directory.is_dir()
        else:
            assert first.directory.stat().st_mode & 0o777 == 0o700
            assert second.directory.stat().st_mode & 0o777 == 0o700
    finally:
        first.cleanup()
        second.cleanup()


def test_derivation_digest_excludes_run_and_timestamp_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")

    def build(*, run_id: str, created_at: datetime) -> DerivationReceipt:
        workspace = PrivacyArtifactWorkspace(
            source_path=source,
            stage="stable-derivation",
            run_id=run_id,
        )
        try:
            temporary = workspace.new_bytes_temp(suffix=".bin")
            write_exclusive_bytes(temporary, b"same output")
            return build_receipt(
                workspace=workspace,
                source_path=source,
                source_sha256=workspace.source_snapshot.sha256,
                transform_input_sha256=workspace.source_snapshot.sha256,
                output_path=temporary,
                output_sha256=file_sha256(temporary),
                source_mime_type="application/octet-stream",
                output_mime_type="application/octet-stream",
                transform_id="stable-transform",
                transform_version="1",
                transform_artifact_path=Path(__file__),
                configuration={"mode": "stable"},
                residual_inspection_sha256=canonical_sha256({"clean": True}),
                created_at=created_at,
            )
        finally:
            workspace.cleanup()

    first = build(run_id="run-one", created_at=_CREATED_AT)
    second = build(
        run_id="run-two",
        created_at=datetime(2026, 8, 27, 10, 30, tzinfo=UTC),
    )

    assert first.receipt_digest != second.receipt_digest
    assert first.derivation_digest == second.derivation_digest
    assert first.derivation_dict() == second.derivation_dict()

    child_output = hashlib.sha256(b"same child output").hexdigest()
    first_child = replace(
        first,
        run_id="child-one",
        transform_input_sha256=first.output_sha256,
        output_sha256=child_output,
        output_size=len(b"same child output"),
        transform_id="stable-child-transform",
        parent_receipt_sha256=first.receipt_digest,
        parent_derivation_sha256=first.derivation_digest,
        created_at=_CREATED_AT.isoformat(),
    )
    second_child = replace(
        first_child,
        run_id="child-two",
        parent_receipt_sha256=second.receipt_digest,
        parent_derivation_sha256=second.derivation_digest,
        created_at=datetime(2026, 8, 27, 10, 30, tzinfo=UTC).isoformat(),
    )

    assert first_child.receipt_digest != second_child.receipt_digest
    assert first_child.derivation_digest == second_child.derivation_digest


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    link = tmp_path / "source-link.bin"
    try:
        link.symlink_to(source)
    except OSError as exc:
        if exc.winerror == 1314:
            pytest.skip(f"file symlinks are unavailable: {exc}")
        raise

    with pytest.raises(ValueError, match="symbolic link"):
        PrivacyArtifactWorkspace(
            source_path=link,
            stage="clean",
            run_id="symlink-run",
        )


def test_publication_is_no_clobber(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="clean",
        run_id="no-clobber-run",
    )
    temporary = workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(temporary, b"output")
    final = workspace.directory / "source.clean.bin"
    final.write_bytes(b"attacker bytes")
    digest = file_sha256(temporary)
    receipt = build_receipt(
        workspace=workspace,
        source_path=source,
        source_sha256=file_sha256(source),
        transform_input_sha256=file_sha256(source),
        output_path=temporary,
        output_sha256=digest,
        source_mime_type=None,
        output_mime_type=None,
        transform_id="test",
        transform_version="1",
        transform_artifact_path=Path(__file__),
        configuration={},
        residual_inspection_sha256=canonical_sha256({"clean": True}),
        created_at=_CREATED_AT,
    )
    try:
        with pytest.raises(FileExistsError):
            workspace.publish(
                temporary_path=temporary,
                receipt=receipt,
                final_name=final.name,
            )
        assert final.read_bytes() == b"attacker bytes"
    finally:
        workspace.cleanup()


def test_receipt_tampering_is_detected(tmp_path: Path) -> None:
    artifact, source = _publish(tmp_path)
    artifact.receipt_path.chmod(0o600)
    artifact.receipt_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="receipt digest mismatch"):
        verify_published_artifact(
            artifact,
            expected_source_sha256=file_sha256(source),
        )


def test_output_replacement_after_publication_is_detected(
    tmp_path: Path,
) -> None:
    artifact, source = _publish(tmp_path)
    artifact.path.chmod(0o600)
    artifact.path.write_bytes(b"replacement")

    with pytest.raises(ValueError, match="output mismatch"):
        verify_published_artifact(
            artifact,
            expected_source_sha256=file_sha256(source),
        )


def test_receipt_binds_exact_source_and_output(tmp_path: Path) -> None:
    artifact, source = _publish(tmp_path)

    receipt = verify_published_artifact(
        artifact,
        expected_source_sha256=file_sha256(source),
    )

    assert receipt.source_sha256 == hashlib.sha256(b"source bytes").hexdigest()
    assert (
        receipt.output_sha256 == hashlib.sha256(b"sanitized bytes").hexdigest()
    )
    assert receipt.output_sha256 == artifact.sha256
    assert receipt.run_id == artifact.run_id


def test_workspace_uses_private_source_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original bytes")
    workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="clean",
        run_id="source-snapshot-run",
    )
    try:
        source.write_bytes(b"replacement bytes")

        assert workspace.source_path.read_bytes() == b"original bytes"
        assert (
            workspace.source_snapshot.sha256
            == hashlib.sha256(b"original bytes").hexdigest()
        )

        temporary = workspace.new_bytes_temp(suffix=".bin")
        write_exclusive_bytes(temporary, b"output")
        with pytest.raises(ValueError, match="source changed"):
            build_receipt(
                workspace=workspace,
                source_path=source,
                source_sha256=workspace.source_snapshot.sha256,
                transform_input_sha256=workspace.source_snapshot.sha256,
                output_path=temporary,
                output_sha256=file_sha256(temporary),
                source_mime_type=None,
                output_mime_type=None,
                transform_id="test",
                transform_version="1",
                transform_artifact_path=Path(__file__),
                configuration={},
                residual_inspection_sha256=canonical_sha256({"clean": True}),
                created_at=_CREATED_AT,
            )
    finally:
        workspace.cleanup()


def test_root_receipt_rejects_unrelated_transform_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="clean",
        run_id="root-input-run",
    )
    temporary = workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(temporary, b"output")
    try:
        with pytest.raises(ValueError, match="root transform input"):
            build_receipt(
                workspace=workspace,
                source_path=source,
                source_sha256=workspace.source_snapshot.sha256,
                transform_input_sha256=hashlib.sha256(b"other").hexdigest(),
                output_path=temporary,
                output_sha256=file_sha256(temporary),
                source_mime_type=None,
                output_mime_type=None,
                transform_id="test",
                transform_version="1",
                transform_artifact_path=Path(__file__),
                configuration={},
                residual_inspection_sha256=canonical_sha256({"clean": True}),
                created_at=_CREATED_AT,
            )
    finally:
        workspace.cleanup()


def test_child_receipt_is_bound_to_parent_output(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")

    parent_workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="metadata-clean",
        run_id="parent-run",
    )
    parent_temp = parent_workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(parent_temp, b"metadata-clean")
    parent_receipt = build_receipt(
        workspace=parent_workspace,
        source_path=source,
        source_sha256=parent_workspace.source_snapshot.sha256,
        transform_input_sha256=parent_workspace.source_snapshot.sha256,
        output_path=parent_temp,
        output_sha256=file_sha256(parent_temp),
        source_mime_type=None,
        output_mime_type=None,
        transform_id="metadata-clean",
        transform_version="1",
        transform_artifact_path=Path(__file__),
        configuration={},
        residual_inspection_sha256=canonical_sha256({"clean": True}),
        created_at=_CREATED_AT,
    )
    parent = parent_workspace.publish(
        temporary_path=parent_temp,
        receipt=parent_receipt,
        final_name="source.metadata-clean.bin",
    )

    child_workspace = PrivacyArtifactWorkspace(
        source_path=source,
        stage="privacy-sanitized",
        run_id="child-run",
    )
    child_temp = child_workspace.new_bytes_temp(suffix=".bin")
    write_exclusive_bytes(child_temp, b"masked")
    child_receipt = build_receipt(
        workspace=child_workspace,
        source_path=source,
        source_sha256=child_workspace.source_snapshot.sha256,
        transform_input_sha256=parent.sha256,
        output_path=child_temp,
        output_sha256=file_sha256(child_temp),
        source_mime_type=None,
        output_mime_type=None,
        transform_id="mask",
        transform_version="1",
        transform_artifact_path=Path(__file__),
        configuration={},
        residual_inspection_sha256=canonical_sha256({"clean": True}),
        created_at=_CREATED_AT,
        parent_artifact=parent,
    )
    child = child_workspace.publish(
        temporary_path=child_temp,
        receipt=child_receipt,
        final_name="source.privacy-sanitized.bin",
    )

    verified = verify_published_artifact(
        child,
        expected_source_sha256=file_sha256(source),
        parent_artifact=parent,
    )
    assert verified.parent_receipt_sha256 == parent.receipt_sha256
    assert verified.parent_derivation_sha256 == parent.derivation_sha256
    assert verified.transform_input_sha256 == parent.sha256

    with pytest.raises(ValueError, match="unexpected privacy receipt parent"):
        verify_published_artifact(
            child,
            expected_source_sha256=file_sha256(source),
        )


def test_active_policy_is_verified_not_just_receipt_self_hash(
    tmp_path: Path,
) -> None:
    import json

    artifact, source = _publish(tmp_path)
    payload = json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
    payload["policy_version"] = "attacker-policy"
    tampered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    artifact.receipt_path.chmod(0o600)
    artifact.receipt_path.write_bytes(tampered)
    forged = PublishedPrivacyArtifact(
        path=artifact.path,
        sha256=artifact.sha256,
        receipt_path=artifact.receipt_path,
        receipt_sha256=hashlib.sha256(tampered).hexdigest(),
        derivation_sha256=artifact.derivation_sha256,
        run_id=artifact.run_id,
    )

    with pytest.raises(ValueError, match="policy mismatch"):
        verify_published_artifact(
            forged,
            expected_source_sha256=file_sha256(source),
        )


def test_successful_publication_removes_raw_snapshot_and_locks_artifacts(
    tmp_path: Path,
) -> None:
    artifact, _source = _publish(tmp_path)

    names = {path.name for path in artifact.path.parent.iterdir()}
    assert names == {artifact.path.name, artifact.receipt_path.name}
    if os.name == "nt":
        assert not artifact.path.stat().st_mode & stat.S_IWRITE
        assert not artifact.receipt_path.stat().st_mode & stat.S_IWRITE
    else:
        assert artifact.path.stat().st_mode & 0o777 == 0o400
        assert artifact.receipt_path.stat().st_mode & 0o777 == 0o400
        assert artifact.path.parent.stat().st_mode & 0o777 == 0o500
