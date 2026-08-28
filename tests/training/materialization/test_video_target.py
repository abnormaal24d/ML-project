from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pytest
import torch

from mmcrawler_datasets.materialization import video_generation
from mmcrawler_datasets.materialization.video_generation import (
    VideoGenerationTargetMaterializer,
)
from mmcrawler_datasets.training_samples.models import TrainingSample
from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget
from multimodal.tokenization.video import VideoTokenizationResult


class _VideoTokenizer:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        token_value: int = 1,
        tokenizer_name: str = "test-video-tokenizer",
        on_encode: Callable[[], None] | None = None,
    ) -> None:
        self.error = error
        self.token_value = token_value
        self.tokenizer_name = tokenizer_name
        self.on_encode = on_encode
        self.calls = 0

    def encode(self, video_path: Path) -> VideoTokenizationResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.on_encode is not None:
            self.on_encode()
        assert video_path.is_file()
        return VideoTokenizationResult(
            tokens=torch.full((1, 2, 2), self.token_value),
            fps=8,
            frames=1,
            height=16,
            width=16,
            tokenizer_name=self.tokenizer_name,
            vocab_size=256,
        )


def _sample(sample_id: str) -> TrainingSample:
    return TrainingSample(
        sample_id=sample_id,
        snapshot_id="snapshot-video",
        task_target=TrainingTaskTarget(
            task_type="text_to_video",
            target_video_path="target.mp4",
            output_modalities=("video",),
        ),
    )


def _materializer(
    tmp_path: Path,
    tokenizer: _VideoTokenizer,
) -> VideoGenerationTargetMaterializer:
    return VideoGenerationTargetMaterializer(
        tokenizer=tokenizer,
        output_root=tmp_path / "staging" / "video_tokens",
    )


@pytest.mark.parametrize(
    "sample_id",
    [
        "..",
        "../x",
        r"..\x",
        r"C:\outside\sample",
        "/outside/sample",
    ],
)
def test_video_sample_id_is_hashed_instead_of_used_as_a_path(
    tmp_path: Path,
    sample_id: str,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    tokenizer = _VideoTokenizer()

    materialized = _materializer(tmp_path, tokenizer).materialize(
        _sample(sample_id),
        project_root=tmp_path,
    )

    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    expected_dir = tmp_path / "staging" / "video_tokens" / sample_key
    tokens_path = expected_dir / "target_video_tokens.pt"
    metadata_path = expected_dir / "target_video_tokens.json"
    assert materialized.task_target.target_video_tokens_path == (
        f"staging/video_tokens/{sample_key}/target_video_tokens.pt"
    )
    assert materialized.task_target.target_video_token_metadata_path == (
        f"staging/video_tokens/{sample_key}/target_video_tokens.json"
    )
    assert torch.equal(
        torch.load(tokens_path, map_location="cpu", weights_only=True),
        torch.ones((1, 2, 2), dtype=torch.long),
    )
    assert (
        json.loads(metadata_path.read_text(encoding="utf-8"))["token_schema"]
        == "video_tokens_t_h_w_v1"
    )
    assert tokenizer.calls == 1


def test_video_sample_ids_that_previously_collided_are_distinct(
    tmp_path: Path,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    tokenizer = _VideoTokenizer()
    materializer = _materializer(tmp_path, tokenizer)

    with_slash = materializer.materialize(
        _sample("a/b"),
        project_root=tmp_path,
    )
    with_underscore = materializer.materialize(
        _sample("a_b"),
        project_root=tmp_path,
    )

    slash_path = with_slash.task_target.target_video_tokens_path
    underscore_path = with_underscore.task_target.target_video_tokens_path
    assert slash_path is not None
    assert underscore_path is not None
    assert slash_path != underscore_path
    assert (tmp_path / slash_path).is_file()
    assert (tmp_path / underscore_path).is_file()


@pytest.mark.parametrize("sample_id", ["", " ", "\t"])
def test_video_materialization_requires_non_empty_sample_id(
    tmp_path: Path,
    sample_id: str,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    tokenizer = _VideoTokenizer()

    with pytest.raises(RuntimeError, match="sample_id must be non-empty"):
        _materializer(tmp_path, tokenizer).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert tokenizer.calls == 0
    assert not (tmp_path / "staging").exists()


def test_failed_video_pair_commit_restores_old_files_and_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "commit-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key
    sample_dir.mkdir(parents=True)
    sentinel = sample_dir / "sentinel.keep"
    sentinel.write_text("do not remove", encoding="utf-8")
    tokens_path = sample_dir / "target_video_tokens.pt"
    metadata_path = sample_dir / "target_video_tokens.json"
    tokens_path.write_bytes(b"old tokens")
    metadata_path.write_text("old metadata", encoding="utf-8")
    original_replace = Path.replace

    def fail_metadata_commit(source: Path, target: Path) -> Path:
        if source.name.endswith(".json.tmp") and Path(target) == metadata_path:
            raise OSError("injected metadata commit failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_commit)

    with pytest.raises(OSError, match="injected metadata commit failure"):
        _materializer(tmp_path, _VideoTokenizer()).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not remove"
    assert tokens_path.read_bytes() == b"old tokens"
    assert metadata_path.read_text(encoding="utf-8") == "old metadata"
    assert sorted(path.name for path in sample_dir.iterdir()) == [
        "sentinel.keep",
        "target_video_tokens.json",
        "target_video_tokens.pt",
    ]


def test_failed_video_pair_commit_removes_partial_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "new-commit-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key
    sample_dir.mkdir(parents=True)
    sentinel = sample_dir / "sentinel.keep"
    sentinel.write_text("preserved", encoding="utf-8")
    metadata_path = sample_dir / "target_video_tokens.json"
    original_replace = Path.replace

    def fail_metadata_commit(source: Path, target: Path) -> Path:
        if source.name.endswith(".json.tmp") and Path(target) == metadata_path:
            raise OSError("injected metadata commit failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_commit)

    with pytest.raises(OSError, match="injected metadata commit failure"):
        _materializer(tmp_path, _VideoTokenizer()).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert list(sample_dir.iterdir()) == [sentinel]


def test_tokenizer_failure_does_not_create_sample_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "tokenizer-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key

    with pytest.raises(ValueError, match="injected tokenizer failure"):
        _materializer(
            tmp_path,
            _VideoTokenizer(error=ValueError("injected tokenizer failure")),
        ).materialize(_sample(sample_id), project_root=tmp_path)

    assert not sample_dir.exists()


def test_video_output_root_is_checked_before_tokenization(
    tmp_path: Path,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    tokenizer = _VideoTokenizer()
    materializer = VideoGenerationTargetMaterializer(
        tokenizer=tokenizer,
        output_root=tmp_path.parent / "external-video-output",
    )

    with pytest.raises(RuntimeError, match="escapes project root"):
        materializer.materialize(_sample("sample"), project_root=tmp_path)

    assert tokenizer.calls == 0


def test_same_sample_writers_are_serialized_and_leave_one_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "concurrent-sample"
    tokenizer_a = _VideoTokenizer(
        token_value=11,
        tokenizer_name="writer-a",
    )
    tokenizer_b = _VideoTokenizer(
        token_value=22,
        tokenizer_name="writer-b",
    )
    materializer_a = _materializer(tmp_path, tokenizer_a)
    materializer_b = _materializer(tmp_path, tokenizer_b)
    original_save = torch.save
    counter_lock = threading.Lock()
    active_saves = 0
    maximum_active_saves = 0

    def monitored_save(value: object, path: object) -> None:
        nonlocal active_saves, maximum_active_saves
        with counter_lock:
            active_saves += 1
            maximum_active_saves = max(maximum_active_saves, active_saves)
        try:
            time.sleep(0.05)
            original_save(value, path)
        finally:
            with counter_lock:
                active_saves -= 1

    monkeypatch.setattr(video_generation.torch, "save", monitored_save)
    start = threading.Barrier(2)

    def run(materializer: VideoGenerationTargetMaterializer) -> TrainingSample:
        start.wait(timeout=5)
        return materializer.materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run, materializer_a),
            executor.submit(run, materializer_b),
        ]
        materialized = [future.result(timeout=10) for future in futures]

    assert maximum_active_saves == 1
    assert (
        materialized[0].task_target.target_video_tokens_path
        == materialized[1].task_target.target_video_tokens_path
    )
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key
    tokens = torch.load(
        sample_dir / "target_video_tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = json.loads(
        (sample_dir / "target_video_tokens.json").read_text(encoding="utf-8")
    )
    pair = (int(tokens[0, 0, 0]), metadata["tokenizer_name"])
    assert pair in {(11, "writer-a"), (22, "writer-b")}
    assert not any(
        path.suffix in {".tmp", ".backup"} for path in sample_dir.iterdir()
    )


def test_incomplete_rollback_preserves_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "rollback-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key
    sample_dir.mkdir(parents=True)
    sentinel = sample_dir / "sentinel.keep"
    sentinel.write_text("preserved", encoding="utf-8")
    tokens_path = sample_dir / "target_video_tokens.pt"
    metadata_path = sample_dir / "target_video_tokens.json"
    tokens_path.write_bytes(b"old tokens")
    metadata_path.write_text("old metadata", encoding="utf-8")
    original_replace = Path.replace

    def fail_commit_and_token_restore(source: Path, target: Path) -> Path:
        target = Path(target)
        if source.name.endswith(".json.tmp") and target == metadata_path:
            raise OSError("injected metadata commit failure")
        if source.name.endswith(".pt.backup") and target == tokens_path:
            raise OSError("injected token rollback failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_commit_and_token_restore)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        _materializer(tmp_path, _VideoTokenizer()).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert not tokens_path.exists()
    assert metadata_path.read_text(encoding="utf-8") == "old metadata"
    backups = sorted(set(sample_dir.glob("*.backup")))
    assert len(backups) == 1
    assert backups[0].name.endswith(".pt.backup")
    assert backups[0].read_bytes() == b"old tokens"
    assert not list(sample_dir.glob("*.tmp"))
    assert not list(sample_dir.glob(".*.tmp"))


@pytest.mark.parametrize("failure_point", ["torch", "json"])
def test_fresh_write_failure_removes_partial_sample_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = f"fresh-{failure_point}-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key

    if failure_point == "torch":

        def fail_save(_value: object, target: object) -> None:
            target.write(b"partial tokens")
            raise OSError("injected torch save failure")

        monkeypatch.setattr(video_generation.torch, "save", fail_save)
    else:
        original_write_text = Path.write_text

        def fail_json_write(
            path: Path,
            data: str,
            *args: object,
            **kwargs: object,
        ) -> int:
            written = original_write_text(path, data, *args, **kwargs)
            if path.name.endswith(".json.tmp"):
                raise OSError("injected JSON write failure")
            return written

        monkeypatch.setattr(Path, "write_text", fail_json_write)

    with pytest.raises(OSError, match="injected"):
        _materializer(tmp_path, _VideoTokenizer()).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert not sample_dir.exists()


def test_sample_directory_symlink_swap_is_rejected_after_tokenization(
    tmp_path: Path,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "swapped-sample-directory"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    output_root = tmp_path / "staging" / "video_tokens"
    sample_dir = output_root / sample_key
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    sentinel = redirect / "sentinel.keep"
    sentinel.write_text("preserved", encoding="utf-8")

    def swap_directory() -> None:
        output_root.mkdir(parents=True)
        try:
            sample_dir.symlink_to(redirect, target_is_directory=True)
        except OSError as exc:
            if exc.winerror == 1314:
                pytest.skip(f"directory symlinks are unavailable: {exc}")
            raise

    tokenizer = _VideoTokenizer(on_encode=swap_directory)

    with pytest.raises(RuntimeError, match="containment changed"):
        _materializer(tmp_path, tokenizer).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert sorted(path.name for path in redirect.iterdir()) == [
        "sentinel.keep"
    ]


def test_transaction_setup_failure_removes_fresh_sample_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target.mp4").write_bytes(b"video")
    sample_id = "transaction-setup-failure"
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = tmp_path / "staging" / "video_tokens" / sample_key

    def fail_uuid() -> object:
        raise ValueError("injected UUID failure")

    monkeypatch.setattr(video_generation, "uuid4", fail_uuid)

    with pytest.raises(ValueError, match="injected UUID failure"):
        _materializer(tmp_path, _VideoTokenizer()).materialize(
            _sample(sample_id),
            project_root=tmp_path,
        )

    assert not sample_dir.exists()
