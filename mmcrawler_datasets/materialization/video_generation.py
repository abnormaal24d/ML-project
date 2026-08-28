"""Materialize video generation targets (tokens) from target_video_path."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import torch
from filelock import FileLock

from mmcrawler_datasets.training_samples.models import TrainingSample
from multimodal.tokenization.video import (
    VideoTokenizationResult,
    VideoTokenizer,
)


class VideoGenerationTargetMaterializer:
    """Turns target_video_path into target_video_tokens_path for generation training."""

    def __init__(
        self,
        *,
        tokenizer: VideoTokenizer,
        output_root: Path,
    ) -> None:
        self._tokenizer = tokenizer
        self._output_root = output_root

    def materialize(
        self,
        sample: TrainingSample,
        *,
        project_root: Path,
    ) -> TrainingSample:
        if sample.task_target.task_type not in {
            "text_to_video",
            "video_editing",
        }:
            return sample

        sample_id = sample.sample_id
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise RuntimeError("video generation sample_id must be non-empty")

        if not sample.task_target.target_video_path:
            raise RuntimeError(
                f"video generation sample {sample_id} has no target_video_path"
            )

        root = project_root.resolve(strict=True)
        relative_video_path = Path(sample.task_target.target_video_path)
        try:
            video_path = (root / relative_video_path).resolve(strict=True)
            video_path.relative_to(root)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"target video is outside the project for {sample_id}"
            ) from exc
        if not video_path.is_file():
            raise RuntimeError(
                f"target video is not a file for sample {sample_id}"
            )

        try:
            output_root = self._output_root.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "video token output escapes project root"
            ) from exc
        try:
            output_root.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                "video token output escapes project root"
            ) from exc

        # The external logical identifier is never interpreted as a path. The
        # complete digest keeps distinct identifiers distinct without leaking
        # path syntax into the filesystem layout.
        sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        try:
            sample_dir = (output_root / sample_key).resolve(strict=False)
            sample_dir.relative_to(output_root)
            sample_dir.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"video token output escapes project root for {sample_id}"
            ) from exc

        tokens_path = sample_dir / "target_video_tokens.pt"
        metadata_path = sample_dir / "target_video_tokens.json"
        _require_contained_artifact(tokens_path, sample_dir)
        _require_contained_artifact(metadata_path, sample_dir)

        # Tokenization may be expensive, but it must not produce filesystem
        # output until every derived destination has passed containment checks.
        result: VideoTokenizationResult = self._tokenizer.encode(video_path)

        payload = {
            "tokenizer_name": result.tokenizer_name,
            "vocab_size": result.vocab_size,
            "fps": result.fps,
            "frames": result.frames,
            "height": result.height,
            "width": result.width,
            "shape": list(result.tokens.shape),
            # Schema identifier, not credential material.
            "token_schema": "video_tokens_t_h_w_v1",  # nosec B105
        }

        try:
            output_root.mkdir(parents=True, exist_ok=True)
            _require_exact_contained_directory(
                output_root,
                expected=output_root,
                parent=root,
            )
            lock_dir = output_root / ".locks"
            lock_dir.mkdir(exist_ok=True)
            _require_exact_contained_directory(
                lock_dir,
                expected=lock_dir,
                parent=output_root,
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"could not prepare video token output for {sample_id}"
            ) from exc

        lock_path = lock_dir / f"{sample_key}.lock"
        _require_contained_artifact(lock_path, lock_dir)
        with FileLock(str(lock_path)):
            # Re-resolve after acquiring the shared per-digest lock. This both
            # serializes writers and detects a directory/symlink swap that
            # happened after the initial containment validation.
            _require_exact_contained_directory(
                lock_dir,
                expected=lock_dir,
                parent=output_root,
            )
            self._commit_artifacts(
                sample_id=sample_id,
                sample_dir=sample_dir,
                output_root=output_root,
                root=root,
                result=result,
                payload=payload,
            )

        tokens_path = sample_dir / "target_video_tokens.pt"
        metadata_path = sample_dir / "target_video_tokens.json"
        return replace(
            sample,
            task_target=replace(
                sample.task_target,
                target_video_tokens_path=tokens_path.relative_to(
                    root
                ).as_posix(),
                target_video_token_metadata_path=metadata_path.relative_to(
                    root
                ).as_posix(),
                # Schema identifier, not credential material.
                video_token_schema="video_tokens_t_h_w_v1",  # nosec B106
            ),
        )

    @staticmethod
    def _commit_artifacts(
        *,
        sample_id: str,
        sample_dir: Path,
        output_root: Path,
        root: Path,
        result: VideoTokenizationResult,
        payload: dict[str, object],
    ) -> None:
        sample_dir_created = False
        tokens_tmp: Path | None = None
        metadata_tmp: Path | None = None
        tokens_backup: Path | None = None
        metadata_backup: Path | None = None
        tokens_path = sample_dir / "target_video_tokens.pt"
        metadata_path = sample_dir / "target_video_tokens.json"
        tokens_had_previous = False
        metadata_had_previous = False
        tokens_backed_up = False
        metadata_backed_up = False
        tokens_committed = False
        metadata_committed = False
        try:
            try:
                sample_dir.mkdir(exist_ok=False)
                sample_dir_created = True
            except FileExistsError:
                pass

            _require_exact_contained_directory(
                sample_dir,
                expected=sample_dir,
                parent=output_root,
            )
            _require_exact_contained_directory(
                sample_dir,
                expected=sample_dir,
                parent=root,
            )
            _require_contained_artifact(tokens_path, sample_dir)
            _require_contained_artifact(metadata_path, sample_dir)

            # All transaction setup is guarded so even an unexpected UUID or
            # path-construction failure removes a newly created empty folder.
            transaction_id = uuid4().hex
            # Keep transaction filenames compact. The directory deliberately
            # uses the complete digest; repeating the artifact basename here
            # would push otherwise valid Windows paths beyond MAX_PATH.
            tokens_tmp = sample_dir / f".{transaction_id}.pt.tmp"
            metadata_tmp = sample_dir / f".{transaction_id}.json.tmp"
            tokens_backup = sample_dir / f".{transaction_id}.pt.backup"
            metadata_backup = sample_dir / f".{transaction_id}.json.backup"
            transaction_paths = (
                tokens_tmp,
                metadata_tmp,
                tokens_backup,
                metadata_backup,
            )
            for transaction_path in transaction_paths:
                _require_contained_artifact(transaction_path, sample_dir)

            tokens_had_previous = _path_exists(tokens_path)
            metadata_had_previous = _path_exists(metadata_path)
            # Pass an already-open stream to PyTorch. Its native path writer
            # still hits the legacy Windows path-length limit, while Python's
            # file API safely handles the full SHA-256 directory layout.
            with tokens_tmp.open("wb") as handle:
                torch.save(
                    result.tokens.cpu().long().contiguous(),
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            metadata_tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            if tokens_had_previous:
                tokens_path.replace(tokens_backup)
                tokens_backed_up = True
            if metadata_had_previous:
                metadata_path.replace(metadata_backup)
                metadata_backed_up = True

            tokens_tmp.replace(tokens_path)
            tokens_committed = True
            metadata_tmp.replace(metadata_path)
            metadata_committed = True
        except Exception:
            rollback_error: Exception | None = None
            if tokens_backup is not None and metadata_backup is not None:
                rollback_error = _rollback_video_artifacts(
                    tokens_path=tokens_path,
                    metadata_path=metadata_path,
                    tokens_backup=tokens_backup,
                    metadata_backup=metadata_backup,
                    tokens_had_previous=tokens_had_previous,
                    metadata_had_previous=metadata_had_previous,
                    tokens_backed_up=tokens_backed_up,
                    metadata_backed_up=metadata_backed_up,
                    tokens_committed=tokens_committed,
                    metadata_committed=metadata_committed,
                )
            _cleanup_optional_paths((tokens_tmp, metadata_tmp))
            if rollback_error is None:
                _cleanup_optional_paths((tokens_backup, metadata_backup))
            if sample_dir_created:
                _remove_empty_directory(sample_dir)
            if rollback_error is not None:
                raise RuntimeError(
                    "video token materialization failed and rollback was "
                    "incomplete"
                ) from rollback_error
            raise

        _cleanup_optional_paths((tokens_backup, metadata_backup))


def _require_contained_artifact(path: Path, directory: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(directory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "video token artifact escapes sample directory"
        ) from exc


def _require_exact_contained_directory(
    path: Path,
    *,
    expected: Path,
    parent: Path,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(parent)
        if resolved != expected or not resolved.is_dir():
            raise ValueError("directory identity changed")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "video token directory containment changed"
        ) from exc


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _rollback_video_artifacts(
    *,
    tokens_path: Path,
    metadata_path: Path,
    tokens_backup: Path,
    metadata_backup: Path,
    tokens_had_previous: bool,
    metadata_had_previous: bool,
    tokens_backed_up: bool,
    metadata_backed_up: bool,
    tokens_committed: bool,
    metadata_committed: bool,
) -> Exception | None:
    rollback_error: Exception | None = None
    for path, committed in (
        (tokens_path, tokens_committed),
        (metadata_path, metadata_committed),
    ):
        if not committed:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            rollback_error = rollback_error or exc

    for path, backup, had_previous, backed_up in (
        (
            tokens_path,
            tokens_backup,
            tokens_had_previous,
            tokens_backed_up,
        ),
        (
            metadata_path,
            metadata_backup,
            metadata_had_previous,
            metadata_backed_up,
        ),
    ):
        if not had_previous or not backed_up:
            continue
        try:
            backup.replace(path)
        except OSError as exc:
            rollback_error = rollback_error or exc
    return rollback_error


def _cleanup_transaction_paths(paths: tuple[Path, ...]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A completed pair remains valid even when the operating system
            # temporarily prevents removal of an obsolete backup artifact.
            continue


def _cleanup_optional_paths(paths: tuple[Path | None, ...]) -> None:
    _cleanup_transaction_paths(
        tuple(path for path in paths if path is not None)
    )


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        # Never remove recursively: a concurrently created or pre-existing
        # sentinel belongs to the caller and must be retained.
        pass
