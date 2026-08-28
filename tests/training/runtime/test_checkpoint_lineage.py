"""Regression coverage for resumed checkpoint receipt lineage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_settings import TrainingSettings
from training.runtime.artifact_persistence import save_epoch_checkpoint
from training.runtime.checkpoint.io import atomic_torch_save, safe_torch_load
from training.runtime.checkpoint.metadata import (
    build_checkpoint_metadata,
    resolve_resumed_from_run_id,
)
from training.runtime.checkpoint.service import (
    resolve_resume_lineage,
    save_training_checkpoint,
)
from training.runtime.loop.state import TrainingLoopState
from training.runtime.trainer import train_and_collect_results


def _metadata(
    *,
    resume_from_checkpoint: str | None = None,
    run_id: str | None = None,
    resumed_from_run_id: str | None = None,
) -> dict[str, object]:
    settings = TrainingSettings(
        device="cpu",
        precision="fp32",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return build_checkpoint_metadata(
        model_settings=ModelSettings(),
        settings=settings,
        training_plan=SimpleNamespace(to_dict=lambda: {}),
        distributed_context={"strategy": "none", "world_size": 1},
        dataset_root=Path("dataset"),
        device="cpu",
        completed_epochs=1,
        sample_count=1,
        effective_training_split=SimpleNamespace(
            sample_count=1,
            task_counts={},
            modality_counts={},
        ),
        training_signal_by_modality={},
        total_batches=1,
        final_loss=0.5,
        train_loss=0.5,
        val_loss=0.5,
        test_loss=0.5,
        dataset_manifest_sha256="a" * 64,
        run_id=run_id,
        resumed_from_run_id=resumed_from_run_id,
    )


def test_resumed_checkpoint_metadata_records_current_and_parent_ids() -> None:
    metadata = _metadata(
        resume_from_checkpoint="source.pt",
        run_id="attempt-current",
        resumed_from_run_id="attempt-parent",
    )

    assert metadata["run_id"] == "attempt-current"
    assert metadata["resumed_from_run_id"] == "attempt-parent"


def test_resumed_checkpoint_metadata_requires_parent_id() -> None:
    with pytest.raises(ValueError, match="resumed_from_run_id"):
        _metadata(
            resume_from_checkpoint="source.pt",
            run_id="attempt-current",
        )


def test_resume_lineage_keeps_the_direct_parent(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    atomic_torch_save(
        payload={
            "checkpoint_payload_version": 1,
            "checkpoint_format": "single_file",
            "model_family": "multimodal_model",
            "artifact_version": "test-v1",
            "metadata": {
                "run_id": "attempt-direct-parent",
                "resumed_from_run_id": "attempt-grandparent",
                "checkpoint_schema": {
                    "model_config_fingerprint": "a" * 64,
                    "training_config_fingerprint": "b" * 64,
                    "dataset_fingerprint": "c" * 64,
                    "tokenizer_fingerprint": "d" * 64,
                    "label_mapping_fingerprint": "e" * 64,
                    "dependency_fingerprint": "f" * 64,
                },
            },
        },
        checkpoint_path=source,
    )

    assert (
        resolve_resume_lineage(
            settings=SimpleNamespace(resume_from_checkpoint=str(source))
        )
        == "attempt-direct-parent"
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        ({"attempt_id": "legacy-attempt"}, "legacy-attempt"),
        (
            {"run_id": None, "attempt_id": "legacy-attempt"},
            "legacy-attempt",
        ),
        ({"resumed_from_run_id": "legacy-parent"}, "legacy-parent"),
    ),
)
def test_resume_lineage_supports_legacy_source_metadata(
    metadata: dict[str, object],
    expected: str,
) -> None:
    assert resolve_resumed_from_run_id(metadata) == expected


def test_epoch_checkpoint_forwards_lineage_to_training_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import training.runtime.artifact_persistence as persistence

    captured: dict[str, object] = {}

    def record_save(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "checkpoint.pt"

    monkeypatch.setattr(persistence, "save_training_checkpoint", record_save)
    prepared = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        training_plan=SimpleNamespace(to_dict=lambda: {}),
        distributed_context={"strategy": "none", "world_size": 1},
        signal_tracker=SimpleNamespace(to_payload=lambda: {}),
        optimizer=torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1),
        scheduler=None,
        grad_scaler=None,
        initialization_metadata={},
        dataset_manifest_sha256="a" * 64,
        resumed_from_run_id="attempt-parent",
    )
    state = SimpleNamespace(
        final_loss=0.5,
        last_val_loss=0.5,
    )

    result = save_epoch_checkpoint(
        path=tmp_path / "checkpoint.pt",
        state=state,
        test_loss=0.5,
        prepared=prepared,
        model_settings=ModelSettings(),
        training_settings=TrainingSettings(device="cpu", precision="fp32"),
        dataset_root=tmp_path,
        device=torch.device("cpu"),
        sample_count=1,
        readiness=SimpleNamespace(),
        logger=SimpleNamespace(),
        run_id="attempt-current",
    )

    assert result == tmp_path / "checkpoint.pt"
    assert captured["run_id"] == "attempt-current"
    assert captured["resumed_from_run_id"] == "attempt-parent"


def test_regular_training_checkpoint_persists_lineage(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "checkpoint.pt"
    settings = TrainingSettings(
        device="cpu",
        precision="fp32",
        resume_from_checkpoint="source.pt",
    )

    save_training_checkpoint(
        model=model,
        checkpoint_path=checkpoint,
        model_settings=ModelSettings(),
        training_settings=settings,
        training_plan=SimpleNamespace(to_dict=lambda: {}),
        distributed_context={
            "enabled": False,
            "strategy": "none",
            "world_size": 1,
            "rank": 0,
        },
        sample_count=1,
        effective_training_split=SimpleNamespace(
            sample_count=1,
            task_counts={},
            modality_counts={},
        ),
        training_signal_by_modality={},
        loop_state=TrainingLoopState(),
        train_loss=0.5,
        val_loss=0.5,
        test_loss=0.5,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        dataset_root=tmp_path,
        dataset_manifest_sha256="a" * 64,
        run_id="attempt-current",
        resumed_from_run_id="attempt-parent",
    )

    payload = safe_torch_load(checkpoint)

    assert payload["metadata"]["run_id"] == "attempt-current"
    assert payload["metadata"]["resumed_from_run_id"] == "attempt-parent"


def test_fsdp_checkpoint_route_preserves_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import training.runtime.checkpoint.service as checkpoint_service

    captured: dict[str, object] = {}

    def record_distributed_save(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "checkpoint.pt"

    monkeypatch.setattr(
        checkpoint_service,
        "_save_distributed_training_checkpoint",
        record_distributed_save,
    )

    result = save_training_checkpoint(
        model=torch.nn.Linear(1, 1),
        checkpoint_path=tmp_path / "checkpoint.pt",
        model_settings=SimpleNamespace(),
        training_settings=SimpleNamespace(),
        training_plan=SimpleNamespace(),
        distributed_context={"strategy": "fsdp"},
        sample_count=1,
        effective_training_split=SimpleNamespace(),
        training_signal_by_modality={},
        loop_state=SimpleNamespace(),
        train_loss=0.5,
        val_loss=0.5,
        test_loss=0.5,
        optimizer=torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1),
        scheduler=None,
        dataset_root=tmp_path,
        dataset_manifest_sha256="a" * 64,
        run_id="attempt-current",
        resumed_from_run_id="attempt-parent",
    )

    assert result == tmp_path / "checkpoint.pt"
    assert captured["run_id"] == "attempt-current"
    assert captured["resumed_from_run_id"] == "attempt-parent"


def test_training_entrypoint_forwards_attempt_id_to_trainer() -> None:
    captured: dict[str, object] = {}
    expected_result = object()

    class RecordingTrainer:
        def train(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return expected_result

    result = train_and_collect_results(
        trainer=RecordingTrainer(),  # type: ignore[arg-type]
        training_root=Path("dataset"),
        checkpoint_path=Path("checkpoint.pt"),
        export_directory=Path("export"),
        dataset_manifest_sha256="a" * 64,
        run_id="attempt-current",
    )

    assert result is expected_result
    assert captured["run_id"] == "attempt-current"
