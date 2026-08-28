"""Secure lifecycle for locally remediated privacy artifacts.

Artifacts are created inside a cryptographically random, mode-0700 workspace,
validated before publication, and published with no-clobber semantics.  Every
published derivative has a canonical derivation receipt that binds the exact
source, transformation input, output, policy, configuration, and residual
inspection evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Mapping, Protocol

from preprocessing.provenance import hash_file

_POLICY_DESCRIPTOR: dict[str, object] = {
    "id": "local-media-privacy-policy",
    "version": "1.0.0",
    "requirements": (
        "exact-source-digest",
        "approved-transform",
        "residual-local-inspection",
        "no-clobber-publication",
        "derivation-receipt",
    ),
}
_BINARY_MODE = getattr(os, "O_BINARY", 0)
POLICY_SHA256 = ""


def canonical_sha256(value: object) -> str:
    """Hash a value using the repository's canonical compact JSON form."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


POLICY_SHA256 = canonical_sha256(_POLICY_DESCRIPTOR)


class PrivacyArtifactVerificationError(ValueError):
    """Raised when a published artifact's binding cannot be trusted."""


class PrivacyArtifactOutputMismatch(PrivacyArtifactVerificationError):
    """Raised when an artifact's bytes differ from the expected output."""


class _HashWriter(Protocol):
    def update(self, data: bytes) -> object: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class DerivationReceipt:
    """Audit receipt plus reproducible identity for one derivative."""

    schema_version: str
    run_id: str
    source_sha256: str
    transform_input_sha256: str
    output_sha256: str
    source_size: int
    output_size: int
    source_mime_type: str | None
    output_mime_type: str | None
    transform_id: str
    transform_version: str
    transform_artifact_sha256: str
    configuration_sha256: str
    policy_id: str
    policy_version: str
    policy_sha256: str
    residual_inspection_sha256: str
    parent_receipt_sha256: str | None
    created_at: str
    parent_derivation_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "privacy-derivation.v1",
            "privacy-derivation.v2",
        }:
            raise ValueError("unsupported privacy receipt schema")
        for name in (
            "run_id",
            "transform_id",
            "transform_version",
            "policy_id",
            "policy_version",
            "created_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"privacy receipt {name} must be non-empty")
        for name in (
            "source_sha256",
            "transform_input_sha256",
            "output_sha256",
            "transform_artifact_sha256",
            "configuration_sha256",
            "policy_sha256",
            "residual_inspection_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        if self.parent_receipt_sha256 is not None:
            _require_sha256(
                self.parent_receipt_sha256,
                field_name="parent_receipt_sha256",
            )
        if self.parent_derivation_sha256 is not None:
            _require_sha256(
                self.parent_derivation_sha256,
                field_name="parent_derivation_sha256",
            )
        if self.source_size < 0 or self.output_size < 0:
            raise ValueError("privacy receipt sizes must be non-negative")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.schema_version == "privacy-derivation.v1":
            payload.pop("parent_derivation_sha256")
        return payload

    def derivation_dict(self) -> dict[str, object]:
        """Return only reproducibility-significant provenance fields."""

        return {
            "schema_version": "privacy-derivation-identity.v1",
            "source_sha256": self.source_sha256,
            "transform_input_sha256": self.transform_input_sha256,
            "output_sha256": self.output_sha256,
            "source_size": self.source_size,
            "output_size": self.output_size,
            "source_mime_type": self.source_mime_type,
            "output_mime_type": self.output_mime_type,
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "transform_artifact_sha256": self.transform_artifact_sha256,
            "configuration_sha256": self.configuration_sha256,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "residual_inspection_sha256": self.residual_inspection_sha256,
            "parent_derivation_sha256": self.parent_derivation_sha256,
        }

    @property
    def derivation_digest(self) -> str:
        """Stable identity independent of execution time and run identity."""

        return canonical_sha256(self.derivation_dict())

    @property
    def receipt_digest(self) -> str:
        """Tamper-evident identity of this specific audit execution."""

        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Private immutable copy of the exact source bytes used by a run."""

    original_path: Path
    path: Path
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class PublishedPrivacyArtifact:
    """Published output plus the immutable receipt that authenticates it."""

    path: Path
    sha256: str
    receipt_path: Path
    receipt_sha256: str
    derivation_sha256: str
    run_id: str


class PrivacyArtifactWorkspace:
    """Private per-operation workspace rooted beside a trusted source."""

    def __init__(self, *, source_path: Path, stage: str, run_id: str) -> None:
        trusted = trusted_regular_file(source_path)
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("privacy run_id must be non-empty")
        self.original_source_path = trusted
        self.stage = _safe_stage(stage)
        parent = trusted.parent.resolve(strict=True)
        directory = tempfile.mkdtemp(
            prefix=f".privacy-{self.stage}-",
            dir=parent,
        )
        self.directory = Path(directory)
        os.chmod(self.directory, 0o700)
        self.run_id = normalized_run_id
        self._published = False
        try:
            self.source_snapshot = self._snapshot_source(trusted)
        except Exception:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise
        self.source_path = self.source_snapshot.path

    def _snapshot_source(self, source: Path) -> SourceSnapshot:
        """Capture source bytes once through a no-follow file descriptor."""

        source_flags = os.O_RDONLY | _BINARY_MODE
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_fd = os.open(source, source_flags)
        snapshot_path = self.directory / (
            f".source-{secrets.token_hex(16)}{source.suffix}"
        )
        digest = hashlib.sha256()
        size = 0
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError("privacy source must be a regular file")
            destination_flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_MODE
            )
            if hasattr(os, "O_NOFOLLOW"):
                destination_flags |= os.O_NOFOLLOW
            destination_fd = os.open(
                snapshot_path,
                destination_flags,
                0o600,
            )
            try:
                with os.fdopen(
                    source_fd,
                    "rb",
                    closefd=False,
                ) as source_handle:
                    size = _copy_and_hash(
                        source_handle,
                        destination_fd,
                        digest,
                    )
                os.fsync(destination_fd)
            except Exception:
                snapshot_path.unlink(missing_ok=True)
                raise
            finally:
                os.close(destination_fd)
        finally:
            os.close(source_fd)
        snapshot = assert_regular_file(snapshot_path)
        if snapshot.stat().st_size != size:
            snapshot.unlink(missing_ok=True)
            raise OSError("privacy source snapshot size mismatch")
        return SourceSnapshot(
            original_path=source,
            path=snapshot,
            sha256=digest.hexdigest(),
            size=size,
            device=source_stat.st_dev,
            inode=source_stat.st_ino,
        )

    def new_bytes_temp(self, *, suffix: str) -> Path:
        """Create an exclusive regular tempfile for in-process byte writes."""

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_MODE
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        path = self.directory / f".{secrets.token_hex(16)}{suffix}"
        descriptor = os.open(path, flags, 0o600)
        os.close(descriptor)
        assert_regular_file(path)
        return path

    def new_external_output_path(self, *, suffix: str) -> Path:
        """Reserve an unpredictable absent path for tools such as FFmpeg.

        The parent directory is atomically created with mode 0700, so no other
        user can pre-place or replace the output name.  The external program is
        required to create the file itself.
        """

        path = self.directory / f".{secrets.token_hex(16)}{suffix}"
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        return path

    def publish(
        self,
        *,
        temporary_path: Path,
        receipt: DerivationReceipt,
        final_name: str,
    ) -> PublishedPrivacyArtifact:
        """Publish validated bytes and receipt atomically without clobbering."""

        temporary = assert_regular_file(temporary_path)
        temporary.resolve(strict=True).relative_to(
            self.directory.resolve(strict=True)
        )
        _fsync_file(temporary)
        actual_digest = file_sha256(temporary)
        if actual_digest != receipt.output_sha256:
            raise ValueError("published output digest does not match receipt")
        if temporary.stat().st_size != receipt.output_size:
            raise ValueError("published output size does not match receipt")

        final_path = self.directory / Path(final_name).name
        receipt_path = final_path.with_name(f"{final_path.name}.receipt.json")
        receipt_bytes = json.dumps(
            receipt.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt.receipt_digest:
            raise ValueError("receipt canonicalization mismatch")

        receipt_temp = self.new_bytes_temp(suffix=".receipt.tmp")
        try:
            _write_exact_bytes(receipt_temp, receipt_bytes)
            _link_no_clobber(temporary, final_path)
            try:
                published_output = assert_regular_file(final_path)
                if file_sha256(published_output) != actual_digest:
                    raise ValueError("published output changed during commit")
                _link_no_clobber(receipt_temp, receipt_path)
                published_receipt = assert_regular_file(receipt_path)
                if (
                    hashlib.sha256(published_receipt.read_bytes()).hexdigest()
                    != receipt.receipt_digest
                ):
                    raise ValueError("published receipt changed during commit")
            except Exception:
                receipt_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
                raise
            temporary.unlink()
            receipt_temp.unlink()
            self.source_path.unlink()
            _fsync_directory(self.directory)
            os.chmod(final_path, 0o400)
            os.chmod(receipt_path, 0o400)
            os.chmod(self.directory, 0o500)
        except Exception:
            receipt_temp.unlink(missing_ok=True)
            raise

        self._published = True
        return PublishedPrivacyArtifact(
            path=final_path,
            sha256=actual_digest,
            receipt_path=receipt_path,
            receipt_sha256=receipt.receipt_digest,
            derivation_sha256=receipt.derivation_digest,
            run_id=receipt.run_id,
        )

    def cleanup(self) -> None:
        if not self._published:
            shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> PrivacyArtifactWorkspace:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.cleanup()


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("privacy receipt created_at must include a timezone")
    return value.astimezone(UTC).isoformat()


def build_receipt(
    *,
    workspace: PrivacyArtifactWorkspace,
    source_path: Path,
    source_sha256: str,
    transform_input_sha256: str,
    output_path: Path,
    output_sha256: str,
    source_mime_type: str | None,
    output_mime_type: str | None,
    transform_id: str,
    transform_version: str,
    transform_artifact_path: Path,
    configuration: Mapping[str, object],
    residual_inspection_sha256: str,
    created_at: datetime,
    parent_artifact: PublishedPrivacyArtifact | None = None,
) -> DerivationReceipt:
    """Build a complete receipt after all exact bytes have been validated."""

    trusted_source = trusted_regular_file(source_path)
    snapshot = assert_regular_file(workspace.source_path)
    output = assert_regular_file(output_path)
    if workspace.source_snapshot.sha256 != source_sha256:
        raise ValueError("source digest does not match private snapshot")
    if file_sha256(snapshot) != source_sha256:
        raise ValueError("private source snapshot changed")
    if file_sha256(trusted_source) != source_sha256:
        raise ValueError("source changed during privacy remediation")
    parent_receipt_sha256: str | None = None
    parent_derivation_sha256: str | None = None
    if parent_artifact is None:
        if transform_input_sha256 != source_sha256:
            raise ValueError("root transform input must equal source bytes")
    else:
        parent_receipt = verify_published_artifact(
            parent_artifact,
            expected_source_sha256=source_sha256,
        )
        if transform_input_sha256 != parent_artifact.sha256:
            raise ValueError(
                "child transform input does not match parent output"
            )
        if parent_receipt.output_sha256 != transform_input_sha256:
            raise ValueError(
                "parent receipt output does not match child input"
            )
        parent_receipt_sha256 = parent_artifact.receipt_sha256
        parent_derivation_sha256 = parent_receipt.derivation_digest
    if file_sha256(output) != output_sha256:
        raise ValueError("output changed after residual inspection")
    return DerivationReceipt(
        schema_version="privacy-derivation.v2",
        run_id=workspace.run_id,
        source_sha256=source_sha256,
        transform_input_sha256=transform_input_sha256,
        output_sha256=output_sha256,
        source_size=workspace.source_snapshot.size,
        output_size=output.stat().st_size,
        source_mime_type=source_mime_type,
        output_mime_type=output_mime_type,
        transform_id=transform_id,
        transform_version=transform_version,
        transform_artifact_sha256=file_sha256(
            trusted_regular_file(transform_artifact_path)
        ),
        configuration_sha256=canonical_sha256(dict(configuration)),
        policy_id=str(_POLICY_DESCRIPTOR["id"]),
        policy_version=str(_POLICY_DESCRIPTOR["version"]),
        policy_sha256=POLICY_SHA256,
        residual_inspection_sha256=residual_inspection_sha256,
        parent_receipt_sha256=parent_receipt_sha256,
        created_at=_utc_timestamp(created_at),
        parent_derivation_sha256=parent_derivation_sha256,
    )


def verify_published_artifact(
    artifact: PublishedPrivacyArtifact,
    *,
    expected_source_sha256: str,
    parent_artifact: PublishedPrivacyArtifact | None = None,
) -> DerivationReceipt:
    """Re-verify output bytes and the canonical on-disk receipt."""

    output = assert_regular_file(artifact.path)
    receipt_path = assert_regular_file(artifact.receipt_path)
    if output.parent.resolve(strict=True) != receipt_path.parent.resolve(
        strict=True
    ):
        raise ValueError(
            "privacy receipt must be co-located with its artifact"
        )
    receipt_payload = receipt_path.read_bytes()
    if hashlib.sha256(receipt_payload).hexdigest() != artifact.receipt_sha256:
        raise ValueError("privacy receipt digest mismatch")
    raw = json.loads(receipt_payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("privacy receipt must be an object")
    schema_version = raw.get("schema_version")
    expected_keys = {field.name for field in fields(DerivationReceipt)}
    if schema_version == "privacy-derivation.v1":
        expected_keys.remove("parent_derivation_sha256")
    if set(raw) != expected_keys:
        raise ValueError("privacy receipt schema is invalid")
    receipt = DerivationReceipt(**raw)
    if receipt.receipt_digest != artifact.receipt_sha256:
        raise ValueError("privacy receipt canonical digest mismatch")
    if receipt.run_id != artifact.run_id:
        raise ValueError("privacy receipt run mismatch")
    if receipt.source_sha256 != expected_source_sha256:
        raise ValueError("privacy receipt source mismatch")
    if (
        receipt.policy_id != _POLICY_DESCRIPTOR["id"]
        or receipt.policy_version != _POLICY_DESCRIPTOR["version"]
        or receipt.policy_sha256 != POLICY_SHA256
    ):
        raise ValueError("privacy receipt policy mismatch")
    if parent_artifact is None:
        if receipt.parent_receipt_sha256 is not None:
            raise ValueError("unexpected privacy receipt parent")
        if receipt.parent_derivation_sha256 is not None:
            raise ValueError("unexpected privacy derivation parent")
        if receipt.transform_input_sha256 != receipt.source_sha256:
            raise ValueError("root receipt input mismatch")
    else:
        parent_receipt = verify_published_artifact(
            parent_artifact,
            expected_source_sha256=expected_source_sha256,
        )
        if receipt.parent_receipt_sha256 != parent_artifact.receipt_sha256:
            raise ValueError("privacy receipt chain mismatch")
        if receipt.schema_version == "privacy-derivation.v1":
            receipt = replace(
                receipt,
                parent_derivation_sha256=parent_receipt.derivation_digest,
            )
        elif (
            receipt.parent_derivation_sha256
            != parent_receipt.derivation_digest
        ):
            raise ValueError("privacy derivation chain mismatch")
        if receipt.transform_input_sha256 != parent_artifact.sha256:
            raise ValueError("privacy receipt parent output mismatch")
        if parent_receipt.output_sha256 != receipt.transform_input_sha256:
            raise ValueError("privacy receipt parent digest mismatch")
    actual_output = file_sha256(output)
    if (
        actual_output != artifact.sha256
        or actual_output != receipt.output_sha256
    ):
        raise ValueError("privacy artifact output mismatch")
    if output.stat().st_size != receipt.output_size:
        raise ValueError("privacy artifact size mismatch")
    if artifact.derivation_sha256 != receipt.derivation_digest:
        raise ValueError("privacy derivation digest mismatch")
    return receipt


def verify_artifact_binding(
    artifact: PublishedPrivacyArtifact,
    *,
    expected_source_sha256: str | None,
    expected_output_sha256: str | None,
    parent_artifact: PublishedPrivacyArtifact | None = None,
) -> DerivationReceipt:
    """Verify artifact bytes, receipt, source binding, and expected output.

    Raises ``PrivacyArtifactOutputMismatch`` when the published bytes differ
    from the expected output digest, and ``PrivacyArtifactVerificationError``
    for any other receipt or binding failure.
    """

    try:
        receipt = verify_published_artifact(
            artifact,
            expected_source_sha256=expected_source_sha256 or "",
            parent_artifact=parent_artifact,
        )
        if artifact.sha256 != expected_output_sha256:
            raise PrivacyArtifactOutputMismatch(
                "privacy artifact expected output mismatch"
            )
    except PrivacyArtifactOutputMismatch:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise PrivacyArtifactVerificationError(str(error)) from error
    return receipt


def trusted_regular_file(path: Path) -> Path:
    """Resolve and require a non-symlink regular file."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("privacy source must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("privacy source must be a regular file")
    return resolved


def assert_regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("privacy artifact must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("privacy artifact must be a regular file")
    return resolved


def file_sha256(path: Path) -> str:
    return hash_file(path)


def optional_file_sha256(path: str | Path | None) -> str | None:
    """Return the SHA-256 of one optional trusted regular file, or ``None``."""

    if not path:
        return None
    try:
        candidate = trusted_regular_file(Path(path))
        return file_sha256(candidate)
    except (OSError, ValueError):
        return None


def privacy_artifact_name(source: Path, *, stage: str) -> str:
    """Create a deterministic, application-owned sibling artifact name."""

    safe_stage = _safe_stage(stage)
    return f"{source.stem}.{safe_stage}{source.suffix}"


def _copy_and_hash(
    source: BinaryIO,
    destination_fd: int,
    digest: _HashWriter,
) -> int:
    total = 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("failed to snapshot privacy source")
            view = view[written:]
            total += written
    return total


def _write_exact_bytes(path: Path, payload: bytes) -> None:
    target = assert_regular_file(path)
    descriptor = os.open(target, os.O_WRONLY | os.O_TRUNC | _BINARY_MODE)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to write privacy artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    """Write exact bytes to an already exclusively created workspace file."""

    _write_exact_bytes(path, payload)


def _link_no_clobber(source: Path, destination: Path) -> None:
    os.link(source, destination, follow_symlinks=False)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(assert_regular_file(path), os.O_RDWR | _BINARY_MODE)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | _BINARY_MODE
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from error


def _safe_stage(stage: str) -> str:
    value = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in stage.strip().lower()
    ).strip("-")
    return value or "sanitized"


__all__ = [
    "DerivationReceipt",
    "POLICY_SHA256",
    "PrivacyArtifactOutputMismatch",
    "PrivacyArtifactVerificationError",
    "PrivacyArtifactWorkspace",
    "PublishedPrivacyArtifact",
    "SourceSnapshot",
    "assert_regular_file",
    "build_receipt",
    "canonical_sha256",
    "file_sha256",
    "optional_file_sha256",
    "privacy_artifact_name",
    "trusted_regular_file",
    "verify_artifact_binding",
    "verify_published_artifact",
    "write_exclusive_bytes",
]
