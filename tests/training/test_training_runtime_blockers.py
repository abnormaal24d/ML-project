from __future__ import annotations

import hashlib
import json
import tomllib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from config.multimodal.training_settings import TrainingSettings
from evaluator.loss import evaluate_final_losses, evaluate_loader_loss
from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from multimodal.model.contracts import CollatedBatch
from training.losses.objective import SUPPORTED_LOSS_KEYS
from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    safe_torch_load,
)
from training.runtime.checkpoint.service import (
    load_model_weights,
    restore_checkpoint_if_requested,
    save_checkpoint,
)
from training.runtime.checkpoint.state import (
    build_training_state_payload,
    capture_random_state,
    set_reproducible_seed,
)
from training.runtime.loop.model_selection import is_improvement
from training.runtime.loop.optimizer_step import _optimizer_step
from training.runtime.loop.runner import (
    TrainingLoopState,
    run_training_loop,
)
from training.runtime.optimization import build_lr_scheduler
from training.runtime.precision import (
    PrecisionRuntime,
    UnsupportedPrecisionError,
    build_grad_scaler,
    resolve_precision_runtime,
)
from training.runtime.preparation import (
    dense_batch_requires_causal_targets,
    validate_dense_decoder_batch,
)
from training.runtime.training_batch_processor import TrainingBatchProcessor


class _SignalTracker:
    def __init__(self) -> None:
        self.backward_calls = 0

    def record_after_backward(self) -> None:
        self.backward_calls += 1


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
        return {"logits": self.linear(batch.text)}


def _batch() -> CollatedBatch:
    return CollatedBatch(
        sample_ids=["sample"],
        text=torch.tensor([[1.0]]),
        image=torch.zeros((1, 1)),
        audio=torch.zeros((1, 1)),
        video=torch.zeros((1, 1)),
        modality_mask=torch.ones((1, 4)),
        labels=torch.tensor([[0.0]]),
    )


def _loss(
    model_output: torch.Tensor,
    batch: CollatedBatch,
    *,
    require_targets_for_generation: bool,
) -> torch.Tensor:
    assert require_targets_for_generation
    assert batch.labels is not None
    return torch.nn.functional.mse_loss(model_output, batch.labels)


class _TrainingLoss(torch.nn.Module):
    def forward(
        self,
        *,
        model_output: dict[str, torch.Tensor],
        batch: CollatedBatch,
        require_targets_for_generation: bool,
    ) -> torch.Tensor:
        assert require_targets_for_generation
        assert batch.labels is not None
        return torch.nn.functional.mse_loss(
            model_output["logits"], batch.labels
        )


def _fp32_runtime() -> PrecisionRuntime:
    return PrecisionRuntime(
        name="fp32",
        device_type="cpu",
        autocast_enabled=False,
        autocast_dtype=None,
        uses_grad_scaler=False,
    )


def _loop_settings(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "progress_log_interval_batches": 100,
        "gradient_clip_max_norm": None,
        "scheduler_interval": "step",
        "monitor_metric": "validation_loss",
        "monitor_mode": "min",
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_dense_decoder_validation_is_objective_aware() -> None:
    non_causal = SimpleNamespace(
        task_types=["text_pretrain", "image_text_pair"],
        decoder_input_ids=torch.empty((2, 0), dtype=torch.long),
        decoder_labels=torch.empty((2, 0), dtype=torch.long),
        decoder_attention_mask=torch.empty((2, 0), dtype=torch.bool),
    )
    assert dense_batch_requires_causal_targets(batch=non_causal) is False

    mixed = SimpleNamespace(
        task_types=["image_text_pair", "instruction_following"],
        decoder_input_ids=torch.tensor([[0, 0, 0], [1, 2, 3]]),
        decoder_labels=torch.tensor(
            [
                [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL],
                [IGNORE_LABEL, 2, 3],
            ]
        ),
        decoder_attention_mask=torch.tensor(
            [[False, False, False], [True, True, True]]
        ),
    )
    assert dense_batch_requires_causal_targets(batch=mixed) is True
    validate_dense_decoder_batch(batch=mixed)

    missing_target = SimpleNamespace(
        task_types=["instruction_following"],
        decoder_input_ids=torch.tensor([[1, 2, 3]]),
        decoder_labels=torch.full((1, 3), IGNORE_LABEL),
        decoder_attention_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="no supervised target token"):
        validate_dense_decoder_batch(batch=missing_target)


@pytest.mark.parametrize(
    ("batches", "accumulation", "epochs", "expected_steps"),
    ((4, 2, 2, 4), (5, 2, 2, 6)),
)
def test_cosine_scheduler_uses_optimizer_steps(
    batches: int,
    accumulation: int,
    epochs: int,
    expected_steps: int,
) -> None:
    settings = TrainingSettings(
        device="cpu",
        precision="fp32",
        epochs=epochs,
        gradient_accumulation_steps=accumulation,
        lr_scheduler="cosine",
        scheduler_interval="step",
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        settings=settings,
        num_training_batches=batches,
        completed_optimizer_steps=0,
    )

    assert scheduler is not None
    assert scheduler.T_max == expected_steps
    for _ in range((batches + accumulation - 1) // accumulation):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0

    for _ in range(expected_steps - scheduler.last_epoch):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_cosine_scheduler_resume_uses_completed_optimizer_steps() -> None:
    settings = TrainingSettings(
        device="cpu",
        precision="fp32",
        epochs=2,
        gradient_accumulation_steps=2,
        lr_scheduler="cosine",
        scheduler_interval="step",
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        settings=settings,
        num_training_batches=5,
        completed_optimizer_steps=3,
    )

    assert scheduler is not None
    assert scheduler.last_epoch == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_precision_is_explicit_and_fails_closed_on_cpu() -> None:
    fp32 = resolve_precision_runtime(
        settings=SimpleNamespace(precision="fp32"),
        device=torch.device("cpu"),
    )
    assert fp32.autocast_enabled is False

    with pytest.raises(
        UnsupportedPrecisionError, match="requires an available CUDA"
    ):
        resolve_precision_runtime(
            settings=SimpleNamespace(precision="fp16"),
            device=torch.device("cpu"),
        )

    with pytest.raises(ValidationError):
        TrainingSettings(mixed_precision=False)


def test_training_settings_and_shipped_configs_use_one_precision_contract() -> (
    None
):
    settings = TrainingSettings(
        device="cpu",
        precision="fp32",
        scheduler_interval="step",
        min_learning_rate=1e-5,
        monitor_metric="validation_loss",
        monitor_mode="min",
    )
    assert settings.precision == "fp32"
    assert settings.scheduler_interval == "step"

    with pytest.raises(ValidationError, match="min_learning_rate"):
        TrainingSettings(learning_rate=1e-5, min_learning_rate=1e-4)

    for config_path in (
        Path("config/profiles/dev.toml"),
        Path("config/profiles/prod.toml"),
    ):
        values = tomllib.loads(config_path.read_text(encoding="utf-8"))
        training = values["training"]
        assert "mixed_precision" not in training
        assert training["precision"] in {"fp32", "fp16", "bf16"}
        assert training["scheduler_interval"] in {"step", "epoch"}
        assert training["monitor_metric"] == "validation_loss"
        assert "epoch_metrics_directory" not in training
        assert "epoch_metrics_filename" not in training

    schema = json.loads(
        Path("docs/configuration_schema.json").read_text(encoding="utf-8")
    )
    properties = schema["$defs"]["TrainingSettings"]["properties"]
    assert "precision" in properties
    assert "mixed_precision" not in properties
    assert "epoch_metrics_directory" not in properties
    assert "epoch_metrics_filename" not in properties


def test_fp16_builds_grad_scaler_and_batch_processor_enters_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.runtime.training_batch_processor as batch_module

    created: dict[str, object] = {}

    class _Scaler:
        pass

    def create_scaler(**kwargs: object) -> _Scaler:
        created.update(kwargs)
        return _Scaler()

    monkeypatch.setattr(torch.amp, "GradScaler", create_scaler)
    fp16_runtime = PrecisionRuntime(
        name="fp16",
        device_type="cuda",
        autocast_enabled=True,
        autocast_dtype=torch.float16,
        uses_grad_scaler=True,
    )
    assert isinstance(build_grad_scaler(fp16_runtime), _Scaler)
    assert created == {"device": "cuda", "enabled": True}
    assert (
        build_grad_scaler(
            PrecisionRuntime(
                name="bf16",
                device_type="cuda",
                autocast_enabled=True,
                autocast_dtype=torch.bfloat16,
                uses_grad_scaler=False,
            )
        )
        is None
    )

    entered = False

    @contextmanager
    def tracked_autocast(_runtime: PrecisionRuntime):
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(batch_module, "autocast_context", tracked_autocast)
    model = _TinyModel()
    processor = TrainingBatchProcessor(
        device=torch.device("cpu"),
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        signal_tracker=_SignalTracker(),
        gradient_accumulation_steps=1,
        precision_runtime=_fp32_runtime(),
    )
    processor.process(
        _batch(),
        batch_count=0,
        total_batches=0,
        epoch_loss=0.0,
        total_loss_sum=0.0,
    )
    assert entered is True


def test_training_batch_processor_rejects_bare_callable_loss() -> None:
    model = _TinyModel()

    with pytest.raises(TypeError, match="torch.nn.Module"):
        TrainingBatchProcessor(
            device=torch.device("cpu"),
            model=model,
            loss_fn=_loss,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            signal_tracker=_SignalTracker(),
            gradient_accumulation_steps=1,
            precision_runtime=_fp32_runtime(),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA to exercise real mixed precision",
)
def test_cuda_fp16_forward_uses_autocast() -> None:
    runtime = resolve_precision_runtime(
        settings=SimpleNamespace(precision="fp16"),
        device=torch.device("cuda"),
    )
    model = torch.nn.Linear(2, 2, device="cuda")
    with torch.autocast(
        device_type=runtime.device_type,
        dtype=runtime.autocast_dtype,
        enabled=runtime.autocast_enabled,
    ):
        output = model(torch.ones((1, 2), device="cuda"))
    assert output.dtype is torch.float16


def test_best_checkpoint_callback_and_early_stopping_share_selection_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.runtime.loop.runner as loop_module

    validation_losses = iter((1.0, 1.1, 1.2))
    monkeypatch.setattr(
        loop_module,
        "evaluate_loader_loss",
        lambda **_kwargs: next(validation_losses),
    )
    model = _TinyModel()
    best_epochs: list[int] = []
    last_epochs: list[int] = []
    state, _history = run_training_loop(
        settings=_loop_settings(epochs=5, early_stopping_patience=2),
        device=torch.device("cpu"),
        logger=None,
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[_batch()],
        val_loader=[_batch()],
        test_loader=[_batch()],
        signal_tracker=_SignalTracker(),
        precision_runtime=_fp32_runtime(),
        distributed_context={"enabled": False, "rank": 0},
        best_epoch_checkpoint=lambda current: best_epochs.append(
            current.best_epoch if current.best_epoch is not None else -1
        ),
        last_epoch_checkpoint=lambda current: last_epochs.append(
            current.completed_epochs
        ),
    )

    assert best_epochs == [0]
    assert last_epochs == [1, 2, 3]
    assert state.completed_epochs == 3
    assert state.epochs_without_improvement == 2
    assert state.stop_reason == "early_stopping"
    assert (
        is_improvement(
            current=0.8,
            best=1.0,
            mode="min",
            min_delta=0.1,
        )
        is True
    )
    assert (
        is_improvement(
            current=1.05,
            best=1.0,
            mode="min",
            min_delta=0.1,
        )
        is False
    )
    assert (
        is_improvement(
            current=2.0,
            best=1.0,
            mode="max",
            min_delta=0.1,
        )
        is True
    )


def test_best_and_last_checkpoints_retain_different_epochs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import training.runtime.loop.runner as loop_module

    validation_losses = iter((1.0, 0.5, 0.7))
    monkeypatch.setattr(
        loop_module,
        "evaluate_loader_loss",
        lambda **_kwargs: next(validation_losses),
    )
    model = _TinyModel()
    best_path = tmp_path / "model.best.pt"
    last_path = tmp_path / "model.last.pt"

    def save(path: Path, state: TrainingLoopState) -> None:
        save_checkpoint(
            model=model,
            checkpoint_path=path,
            model_settings=_CheckpointModelSettings(),
            training_settings=_CheckpointTrainingSettings(),
            metadata={"epoch": state.completed_epochs},
        )

    run_training_loop(
        settings=_loop_settings(epochs=3),
        device=torch.device("cpu"),
        logger=None,
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[_batch()],
        val_loader=[_batch()],
        test_loader=[_batch()],
        signal_tracker=_SignalTracker(),
        precision_runtime=_fp32_runtime(),
        distributed_context={"enabled": False, "rank": 0},
        best_epoch_checkpoint=lambda state: save(best_path, state),
        last_epoch_checkpoint=lambda state: save(last_path, state),
    )

    best = safe_torch_load(best_path)
    last = safe_torch_load(last_path)
    assert best["metadata"]["epoch"] == 2
    assert last["metadata"]["epoch"] == 3


def test_resume_state_preserves_early_stopping_and_optimizer_state() -> None:
    resume_state = {
        "epoch": 3,
        "global_step": 7,
        "total_batches": 11,
        "final_loss": 0.7,
        "cumulative_loss_sum": 3.2,
        "epoch_losses": [1.2, 0.9, 0.7],
        "epoch_history": [{"epoch": 2}],
        "best_metric": 0.6,
        "best_epoch": 2,
        "epochs_without_improvement": 1,
        "stop_reason": None,
        "last_val_loss": 0.7,
        "last_gradient_norm": 0.8,
        "gradient_clip_count": 2,
    }
    state = TrainingLoopState.from_resume_state(resume_state)

    assert state.completed_epochs == 3
    assert state.completed_optimizer_steps == 7
    assert state.total_batches == 11
    assert state.best_metric == pytest.approx(0.6)
    assert state.epochs_without_improvement == 1

    payload = build_training_state_payload(
        state=state,
        val_loss=0.7,
        random_state={},
        training_plan=SimpleNamespace(to_dict=lambda: {}),
    )
    assert payload["global_step"] == 7
    assert payload["epochs_without_improvement"] == 1
    assert payload["best_epoch"] == 2


def test_resume_does_not_restart_a_completed_early_stop() -> None:
    state = TrainingLoopState.from_resume_state(
        {
            "epoch": 3,
            "global_step": 3,
            "total_batches": 3,
            "final_loss": 0.5,
            "cumulative_loss_sum": 1.5,
            "epoch_losses": [0.7, 0.6, 0.5],
            "epoch_history": [],
            "best_metric": 0.4,
            "best_epoch": 1,
            "epochs_without_improvement": 2,
            "stop_reason": "early_stopping",
            "last_val_loss": 0.5,
            "last_gradient_norm": None,
            "gradient_clip_count": 0,
        }
    )
    tracker = _SignalTracker()
    model = _TinyModel()
    completed, _history = run_training_loop(
        settings=_loop_settings(epochs=5, early_stopping_patience=2),
        device=torch.device("cpu"),
        logger=None,
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[_batch()],
        val_loader=[_batch()],
        test_loader=[_batch()],
        signal_tracker=tracker,
        precision_runtime=_fp32_runtime(),
        distributed_context={"enabled": False, "rank": 0},
        loop_state=state,
    )

    assert completed.completed_epochs == 3
    assert completed.stop_reason == "early_stopping"
    assert tracker.backward_calls == 0


class _RecordingSgd(torch.optim.SGD):
    seen_gradient: float | None = None

    def step(self, closure=None):
        parameter = self.param_groups[0]["params"][0]
        self.seen_gradient = float(parameter.grad.detach().abs().max())
        return super().step(closure=closure)


def _linear_with_gradient(value: float) -> tuple[_TinyModel, _RecordingSgd]:
    model = _TinyModel()
    model.linear.weight.data.zero_()
    model.linear.weight.grad = torch.tensor([[value]])
    return model, _RecordingSgd(model.parameters(), lr=1.0)


def test_gradient_clipping_happens_before_optimizer_step() -> None:
    model, optimizer = _linear_with_gradient(10.0)
    result = _optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_interval="step",
        grad_scaler=None,
        gradient_clip_max_norm=1.0,
    )

    assert result.gradient_norm == pytest.approx(10.0)
    assert result.gradient_was_clipped is True
    assert optimizer.seen_gradient == pytest.approx(1.0)
    assert float(model.linear.weight.detach()) == pytest.approx(-1.0)


def test_fp16_unscales_before_clipping_and_nonfinite_gradients_abort() -> None:
    model, optimizer = _linear_with_gradient(4.0)
    calls: list[str] = []

    class _Scaler:
        def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
            calls.append("unscale")
            assert model.linear.weight.grad is not None
            model.linear.weight.grad.div_(2.0)

        def step(self, wrapped_optimizer: torch.optim.Optimizer) -> None:
            calls.append("step")
            wrapped_optimizer.step()

        def update(self) -> None:
            calls.append("update")

    _optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_interval="step",
        grad_scaler=_Scaler(),
        gradient_clip_max_norm=1.0,
    )
    assert calls == ["unscale", "step", "update"]
    assert optimizer.seen_gradient == pytest.approx(1.0)

    model, optimizer = _linear_with_gradient(float("inf"))
    before = model.linear.weight.detach().clone()
    with pytest.raises(ValueError, match="non-finite gradient"):
        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scheduler_interval="step",
            grad_scaler=None,
            gradient_clip_max_norm=1.0,
        )
    assert torch.equal(model.linear.weight.detach(), before)


def test_gradient_clipping_runs_once_per_accumulated_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.nn.utils.clip_grad_norm_
    calls = 0

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted)
    model = _TinyModel()
    state, _history = run_training_loop(
        settings=_loop_settings(
            gradient_accumulation_steps=2,
            gradient_clip_max_norm=1.0,
        ),
        device=torch.device("cpu"),
        logger=None,
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[_batch(), _batch(), _batch()],
        val_loader=[_batch()],
        test_loader=[_batch()],
        signal_tracker=_SignalTracker(),
        precision_runtime=_fp32_runtime(),
        distributed_context={"enabled": False, "rank": 0},
    )

    assert state.completed_optimizer_steps == 2
    assert calls == 2


class _CheckpointModelSettings:
    artifact_version = "test-v1"
    model_family = "multimodal_model"

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {}


class _CheckpointTrainingSettings:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {}

    def effective_min_task_samples(self) -> dict[str, int]:
        return {}


def test_checkpoint_versioning_does_not_consume_training_rng(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint_path = tmp_path / "model.pt"
    seed_settings = SimpleNamespace(seed=314159, deterministic=False)

    set_reproducible_seed(settings=seed_settings)
    python_rng_state = capture_random_state()["python"]

    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
        training_state={"random_state": {"python": python_rng_state}},
    )

    assert capture_random_state()["python"] == python_rng_state

    # A resumed run restores the state captured before the previous save.
    # The next save must therefore create a new artifact version without
    # reproducing the previous version-directory identifier.
    set_reproducible_seed(settings=seed_settings)
    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
        training_state={"random_state": {"python": python_rng_state}},
    )

    version_root = tmp_path / "model.pt.d"
    version_directories = sorted(
        path for path in version_root.iterdir() if path.is_dir()
    )
    assert len(version_directories) == 2
    assert safe_torch_load(checkpoint_path)["model_family"] == (
        "multimodal_model"
    )


def test_corrupted_manifest_fails_closed(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
        training_state={"global_step": 17},
    )

    checkpoint_path.write_text('{"corrupted"', encoding="utf-8")

    assert not checkpoint_is_available(checkpoint_path)
    with pytest.raises(FileNotFoundError, match="not a canonical manifest"):
        safe_torch_load(checkpoint_path)


def test_missing_manifest_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
        training_state={"global_step": 9},
    )
    checkpoint_path.unlink()

    assert not checkpoint_is_available(checkpoint_path)
    with pytest.raises(FileNotFoundError, match="not a canonical manifest"):
        safe_torch_load(checkpoint_path)


def test_resume_fails_without_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import training.runtime.checkpoint.service as checkpoint_service

    restored_model = torch.nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(),
        lr=0.1,
    )
    checkpoint_path = tmp_path / "missing.pt"

    monkeypatch.setattr(
        checkpoint_service,
        "build_checkpoint_fingerprint_schema",
        lambda **_kwargs: {},
    )
    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        restore_checkpoint_if_requested(
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=None,
            settings=SimpleNamespace(
                resume_from_checkpoint=str(checkpoint_path)
            ),
            model_settings=_CheckpointModelSettings(),
            dataset_root=tmp_path,
            dataset_manifest_sha256="d" * 64,
        )


def test_manifest_publication_failure_removes_incomplete_version_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import training.runtime.checkpoint.io as checkpoint_io

    real_atomic_write_json = checkpoint_io._atomic_write_json

    def fail_manifest_publication(
        *,
        path: Path,
        payload: dict[str, object],
    ) -> None:
        if path == checkpoint_io.Path(tmp_path / "model.pt"):
            raise OSError("simulated manifest publication failure")
        real_atomic_write_json(path=path, payload=payload)

    monkeypatch.setattr(
        checkpoint_io,
        "_atomic_write_json",
        fail_manifest_publication,
    )
    checkpoint_path = tmp_path / "model.pt"
    with pytest.raises(
        OSError, match="simulated manifest publication failure"
    ):
        save_checkpoint(
            model=torch.nn.Linear(1, 1, bias=False),
            checkpoint_path=checkpoint_path,
            model_settings=_CheckpointModelSettings(),
            training_settings=_CheckpointTrainingSettings(),
            metadata={},
        )

    version_root = tmp_path / "model.pt.d"
    assert not version_root.exists() or list(version_root.iterdir()) == []
    assert not checkpoint_path.exists()


def test_manifest_replacement_is_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import training.runtime.checkpoint.io as checkpoint_io

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
    )
    previous_manifest = checkpoint_path.read_text(encoding="utf-8")

    real_replace = checkpoint_io.os.replace

    def interrupt_manifest_replace(source: str, destination: str) -> None:
        if Path(destination) == checkpoint_path:
            raise OSError("simulated interruption before manifest replace")
        real_replace(source, destination)

    monkeypatch.setattr(
        checkpoint_io.os,
        "replace",
        interrupt_manifest_replace,
    )
    model.weight.data.fill_(2.0)
    with pytest.raises(OSError, match="simulated interruption"):
        save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            model_settings=_CheckpointModelSettings(),
            training_settings=_CheckpointTrainingSettings(),
            metadata={},
        )

    assert checkpoint_path.read_text(encoding="utf-8") == previous_manifest
    payload = safe_torch_load(checkpoint_path)
    assert payload["model_state"]["weight"].item() == pytest.approx(1.0)
    assert not list(tmp_path.glob(".model.pt.*.tmp"))


def test_manifest_cannot_resolve_outside_checkpoint_version_root(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    outside_path = tmp_path / "outside.pt"
    torch.save({"model_state": {"weight": torch.tensor([1.0])}}, outside_path)
    checksum = hashlib.sha256(outside_path.read_bytes()).hexdigest()
    outside_path.with_name("outside.pt.sha256").write_text(
        f"{checksum}  outside.pt\n",
        encoding="ascii",
    )

    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "versioned",
                "version_dir": "..",
                "file": "outside.pt",
                "sha256": checksum,
            }
        ),
        encoding="utf-8",
    )

    assert not checkpoint_is_available(checkpoint_path)
    with pytest.raises(FileNotFoundError, match="not a canonical manifest"):
        safe_torch_load(checkpoint_path)


def test_failed_checkpoint_save_removes_incomplete_version_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import training.runtime.checkpoint.io as checkpoint_io

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated torch save failure")

    monkeypatch.setattr(checkpoint_io.torch, "save", fail_save)
    checkpoint_path = tmp_path / "model.pt"
    with pytest.raises(RuntimeError, match="simulated torch save failure"):
        save_checkpoint(
            model=torch.nn.Linear(1, 1, bias=False),
            checkpoint_path=checkpoint_path,
            model_settings=_CheckpointModelSettings(),
            training_settings=_CheckpointTrainingSettings(),
            metadata={},
        )

    version_root = tmp_path / "model.pt.d"
    assert version_root.is_dir()
    assert list(version_root.iterdir()) == []


def test_best_checkpoint_weights_can_be_selected_for_export(
    tmp_path: Path,
) -> None:
    selected_model = torch.nn.Linear(1, 1, bias=False)
    selected_model.weight.data.fill_(2.0)
    best_checkpoint = tmp_path / "model.best.pt"
    save_checkpoint(
        model=selected_model,
        checkpoint_path=best_checkpoint,
        model_settings=_CheckpointModelSettings(),
        training_settings=_CheckpointTrainingSettings(),
        metadata={},
    )
    last_model = torch.nn.Linear(1, 1, bias=False)
    last_model.weight.data.fill_(9.0)

    load_model_weights(
        model=last_model,
        checkpoint_path=best_checkpoint,
        model_settings=_CheckpointModelSettings(),
    )
    payload = safe_torch_load(best_checkpoint)

    assert float(last_model.weight.detach()) == pytest.approx(2.0)
    assert isinstance(payload, dict)
    assert payload["model_state"]["weight"].item() == pytest.approx(2.0)


def test_training_backend_resolution_fails_fast_for_unimplemented_backends() -> (
    None
):
    from config.multimodal.training_settings import (
        IMPLEMENTED_TRAINING_BACKENDS,
        SUPPORTED_TRAINING_BACKENDS,
    )
    from training.runtime.preparation import (
        PreparedTrainingBackend,
        prepare_training_backend,
    )

    prepared = prepare_training_backend(
        training_settings=TrainingSettings(
            training_backend="pipeline_smoke",
            device="cpu",
            precision="fp32",
        )
    )
    assert isinstance(prepared, PreparedTrainingBackend)
    assert prepared.name == "pipeline_smoke"
    assert prepared.requires_distributed_runtime is False
    assert prepared.requires_gpu is False
    assert prepared.requires_dense_sequence_targets is False

    unimplemented = sorted(
        SUPPORTED_TRAINING_BACKENDS - IMPLEMENTED_TRAINING_BACKENDS
    )
    assert unimplemented
    for backend in unimplemented:
        with pytest.raises(ValueError, match="not implemented"):
            prepare_training_backend(
                training_settings=TrainingSettings(
                    training_backend=backend,
                    device="cpu",
                    precision="fp32",
                )
            )


def test_collator_and_dataloader_reject_unimplemented_training_backends() -> (
    None
):
    from config.multimodal.training_settings import (
        IMPLEMENTED_TRAINING_BACKENDS,
        SUPPORTED_TRAINING_BACKENDS,
    )
    from mmcrawler_datasets.collation.multimodal import (
        MultimodalCollator,
        UnsupportedFeatureBackendError,
    )
    from mmcrawler_datasets.dataloader import (
        _validate_collator_backend_compatibility,
    )

    _validate_collator_backend_compatibility(backend="pipeline_smoke")

    unimplemented = sorted(
        SUPPORTED_TRAINING_BACKENDS - IMPLEMENTED_TRAINING_BACKENDS
    )
    assert unimplemented
    for backend in unimplemented:
        with pytest.raises(ValueError, match="unsupported training_backend"):
            _validate_collator_backend_compatibility(backend=backend)
        with pytest.raises(
            UnsupportedFeatureBackendError,
            match="unsupported training_backend",
        ):
            MultimodalCollator(
                tokenizer=object(),
                training_backend=backend,
                text_dim=128,
                image_dim=128,
                audio_dim=128,
                video_dim=128,
                raw_text_max_tokens=128,
                raw_text_vocab_size=4096,
                raw_image_size=64,
                raw_audio_num_samples=8000,
                raw_video_frames=2,
                video_generation_frames=8,
                audio_token_codec="discrete",
                mlm_probability=0.15,
                image_mask_probability=0.2,
                audio_mask_probability=0.15,
                materialized_tensors_enabled=False,
                base_seed=0,
            )


def test_shipped_configs_use_only_implemented_training_backends() -> None:
    from config.multimodal.training_settings import (
        IMPLEMENTED_TRAINING_BACKENDS,
    )

    for config_path in (
        Path("config/profiles/test.toml"),
        Path("config/profiles/dev.toml"),
        Path("config/profiles/prod.toml"),
    ):
        values = tomllib.loads(config_path.read_text(encoding="utf-8"))
        training = values["training"]
        assert "encoder_backend" not in training
        assert training["training_backend"] in IMPLEMENTED_TRAINING_BACKENDS


def test_contrastive_outputs_and_objective_use_aligned_batch_rows() -> None:
    from multimodal.model.outputs.builders import _build_contrastive_outputs
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    batch = CollatedBatch(
        sample_ids=["one", "two"],
        text=torch.zeros((2, 2)),
        image=torch.zeros((2, 2)),
        audio=torch.zeros((2, 2)),
        video=torch.zeros((2, 2)),
        modality_mask=torch.ones((2, 4), dtype=torch.bool),
        labels=None,
        task_types=["image_text_pair", "image_text_pair"],
        alignment_scores=torch.tensor([1.0, 1.0]),
        text_mask=torch.ones(2, dtype=torch.bool),
        image_mask=torch.ones(2, dtype=torch.bool),
    )
    embeddings = {
        "text_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "image_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    outputs = _build_contrastive_outputs(
        batch=batch,
        modality_outputs=embeddings,
    )

    logits = outputs["contrastive_logits"]
    assert logits.shape == (2, 2)
    assert torch.diagonal(logits).gt(logits.flip(1).diagonal()).all()

    losses = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )(
        model_output=outputs,
        batch=batch,
    )
    assert set(losses) == {"contrastive", "total"}
    assert torch.isfinite(losses["contrastive"])


def test_causal_language_and_ocr_losses_use_decoder_labels() -> None:
    from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    batch = CollatedBatch(
        sample_ids=["instruction", "ocr"],
        text=torch.zeros((2, 2)),
        image=torch.zeros((2, 2)),
        audio=torch.zeros((2, 2)),
        video=torch.zeros((2, 2)),
        modality_mask=torch.ones((2, 4), dtype=torch.bool),
        labels=None,
        task_types=["instruction_following", "ocr_parse"],
        decoder_labels=torch.tensor(
            [
                [IGNORE_LABEL, IGNORE_LABEL, 2, 3],
                [IGNORE_LABEL, IGNORE_LABEL, 1, 4],
            ],
            dtype=torch.long,
        ),
    )
    logits = torch.full((2, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    logits[1, 1, 1] = 10.0
    logits[1, 2, 4] = 10.0

    losses = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )(
        model_output={"sequence_logits": logits},
        batch=batch,
    )

    assert "language_modeling" in losses
    assert "ocr_sequence" in losses
    assert losses["language_modeling"].item() < 1e-5
    assert losses["ocr_sequence"].item() < 1e-5


def test_shipped_prod_candidate_campaign_resolves_dense_backend(
    production_whisper_env: None,
) -> None:
    from config.load import load_settings
    from training.runtime.preparation import prepare_training_backend

    settings = load_settings(
        "prod",
        project_root=Path(__file__).resolve().parents[2],
        environment="prod",
    )
    assert settings.training.release_stage == "candidate"
    assert settings.training.training_backend == "dense_transformer"
    backend = prepare_training_backend(
        training_settings=settings.training,
    )
    assert backend.name == "dense_transformer"
    assert backend.requires_dense_sequence_targets is True


def test_dense_transformer_forward_loss_and_generation_are_sequence_based() -> (
    None
):
    from config.multimodal.encoder_settings import EncoderSettings
    from config.multimodal.generation_settings import GenerationSettings
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_head_settings import DecoderSettings
    from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
    from orchestration.composition.runtime.training import build_model
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    encoder = EncoderSettings(
        input_dim=16, hidden_dim=16, output_dim=16, dropout=0.0
    )
    settings = ModelSettings(
        text=encoder,
        document=encoder,
        image=encoder,
        audio=encoder,
        video=encoder,
        fusion_dim=16,
        projection_dim=16,
        raw_text_vocab_size=269,
        raw_text_max_tokens=16,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        enabled_modalities=("text",),
        output_modalities=("text",),
        text_decoder=DecoderSettings(
            enabled=True,
            vocab_size=269,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_target_tokens=16,
            max_context_tokens=16,
            max_text_context_tokens=16,
            max_document_context_tokens=0,
            max_image_context_tokens=0,
            max_audio_context_tokens=0,
            max_video_context_tokens=0,
        ),
        generation=GenerationSettings(
            max_new_tokens=3, temperature=0.0, top_p=1.0, top_k=0
        ),
    )
    model = build_model(settings, training_backend="dense_transformer")
    batch = CollatedBatch(
        sample_ids=["sample"],
        text=torch.tensor([[2, 9, 20, 21, 10, 22, 3, 0]], dtype=torch.long),
        image=torch.zeros((1, 16)),
        audio=torch.zeros((1, 16)),
        video=torch.zeros((1, 16)),
        modality_mask=torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool),
        labels=None,
        task_types=["instruction_following"],
        text_mask=torch.tensor([True]),
        document_mask=torch.tensor([False]),
        image_mask=torch.tensor([False]),
        audio_mask=torch.tensor([False]),
        video_mask=torch.tensor([False]),
        decoder_input_ids=torch.tensor(
            [[2, 9, 20, 21, 10, 22, 3, 0]], dtype=torch.long
        ),
        decoder_attention_mask=torch.tensor(
            [[True, True, True, True, True, True, True, False]]
        ),
        decoder_labels=torch.tensor(
            [
                [
                    IGNORE_LABEL,
                    IGNORE_LABEL,
                    IGNORE_LABEL,
                    IGNORE_LABEL,
                    IGNORE_LABEL,
                    22,
                    3,
                    IGNORE_LABEL,
                ]
            ],
            dtype=torch.long,
        ),
        prompt_token_count=[5],
        answer_token_count=[2],
        conversation_flags=[True],
    )

    model.eval()
    outputs = model(batch, output_heads={"sequence", "modality_embeddings"})
    assert outputs["sequence_logits"].shape == (1, 8, 269)
    assert outputs["decoder_hidden_states"].shape == (1, 8, 16)
    losses = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )(model_output=outputs, batch=batch)
    language_modeling_loss = losses["language_modeling"]
    assert torch.isfinite(language_modeling_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    language_modeling_loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()

    model.eval()
    generated = model.generate(batch, max_new_tokens=3, temperature=0.0)
    assert generated.shape == (1, 3)


def test_dense_decoder_cached_step_matches_full_forward() -> None:
    from config.multimodal.encoder_settings import EncoderSettings
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_head_settings import DecoderSettings
    from multimodal.model.contracts import ModalityTokenSequence
    from multimodal.model.forward import DenseCausalDecoder

    encoder = EncoderSettings(
        input_dim=8, hidden_dim=8, output_dim=8, dropout=0.0
    )
    settings = ModelSettings(
        text=encoder,
        document=encoder,
        image=encoder,
        audio=encoder,
        video=encoder,
        fusion_dim=8,
        projection_dim=8,
        raw_text_vocab_size=269,
        raw_text_max_tokens=12,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        enabled_modalities=("text",),
        text_decoder=DecoderSettings(
            enabled=True,
            vocab_size=269,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            max_target_tokens=12,
            max_context_tokens=8,
            max_text_context_tokens=8,
            max_document_context_tokens=0,
            max_image_context_tokens=0,
            max_audio_context_tokens=0,
            max_video_context_tokens=0,
        ),
    )
    decoder = DenseCausalDecoder(config=settings).eval()
    context = ModalityTokenSequence(
        tokens=torch.randn(1, 2, 8),
        attention_mask=torch.tensor([[True, True]]),
        modality_ids=torch.zeros((1, 2), dtype=torch.long),
    )
    prompt = torch.tensor([[2, 9, 10]], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
    _prompt_output, cache = decoder(
        context=context,
        input_ids=prompt,
        attention_mask=prompt_mask,
        return_cache=True,
    )
    assert cache is not None
    step_logits, _ = decoder.decode_step(
        token_ids=torch.tensor([42]),
        cache=cache,
        step_valid_mask=torch.tensor([True]),
    )
    full_output, _ = decoder(
        context=context,
        input_ids=torch.tensor([[2, 9, 10, 42]], dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    assert torch.allclose(
        step_logits, full_output.logits[:, -1, :], atol=1e-5, rtol=1e-5
    )


def test_dense_causal_loss_never_adds_legacy_sequence_loss() -> None:
    from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    batch = CollatedBatch(
        sample_ids=["dense"],
        text=torch.zeros((1, 2)),
        image=torch.zeros((1, 2)),
        audio=torch.zeros((1, 2)),
        video=torch.zeros((1, 2)),
        modality_mask=torch.ones((1, 4), dtype=torch.bool),
        labels=None,
        task_types=["instruction_following"],
        decoder_labels=torch.tensor(
            [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
        ),
        target_token_ids=torch.tensor([[2, 3, 0, 0]], dtype=torch.long),
        target_attention_mask=torch.tensor(
            [[True, True, False, False]], dtype=torch.bool
        ),
    )
    logits = torch.zeros((1, 4, 8))
    losses = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS},
        training_backend="dense_transformer",
    )(model_output={"sequence_logits": logits}, batch=batch)

    assert "language_modeling" in losses
    assert "sequence" not in losses


def test_preference_and_safety_losses_are_trainable() -> None:
    from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    labels = torch.tensor(
        [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
    )
    chosen = torch.zeros((1, 4, 6), requires_grad=True)
    rejected = torch.zeros((1, 4, 6), requires_grad=True)
    chosen.data[0, 1, 2] = 5.0
    chosen.data[0, 2, 3] = 5.0
    rejected.data[0, 1, 2] = -5.0
    rejected.data[0, 2, 3] = -5.0
    batch = CollatedBatch(
        sample_ids=["preference"],
        text=torch.zeros((1, 2)),
        image=torch.zeros((1, 2)),
        audio=torch.zeros((1, 2)),
        video=torch.zeros((1, 2)),
        modality_mask=torch.ones((1, 4), dtype=torch.bool),
        labels=None,
        chosen_labels=labels,
        rejected_labels=labels,
        safety_targets=torch.tensor([[1.0, 0.0]]),
        safety_target_mask=torch.tensor([[True, False]]),
    )
    safety_logits = torch.tensor([[2.0, 100.0]], requires_grad=True)
    losses = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS},
        training_backend="dense_transformer",
        training_stage="PREFERENCE_TUNING",
    )(
        model_output={
            "chosen_sequence_logits": chosen,
            "rejected_sequence_logits": rejected,
            "safety_logits": safety_logits,
        },
        batch=batch,
    )

    assert set(losses) == {"preference", "safety", "total"}
    assert losses["preference"].item() < 0.7
    losses["total"].backward()
    assert chosen.grad is not None
    assert rejected.grad is not None
    assert safety_logits.grad is not None
    assert safety_logits.grad[0, 1].item() == 0.0


def test_media_context_budget_samples_the_full_timeline() -> None:
    from multimodal.model.fusion import _context_token_indices

    selected = _context_token_indices(
        modality="video",
        token_count=10,
        selected_count=4,
        device=torch.device("cpu"),
    )

    assert selected.tolist() == [0, 3, 6, 9]


def test_evaluation_result_defaults_fail_closed() -> None:
    from evaluator.results import EvaluationResult

    result = EvaluationResult(validation_loss=1.0, test_loss=1.0)

    assert result.valid is False
    assert result.evaluation_mode == "not_evaluated"


def test_select_samples_accepts_validator_settings() -> None:
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )
    from mmcrawler_datasets.selection.pipeline import select_samples

    selected = select_samples(
        (),
        TrainingSnapshotAssemblerSettings(),
        DatasetValidatorSettings(),
    )

    assert selected == ()


class _ModeProbeLoss:
    """Record the model training flag at call time."""

    def __init__(self) -> None:
        self.modes: list[bool] = []

    def __call__(
        self,
        model_output: torch.Tensor,
        batch: CollatedBatch,
        *,
        require_targets_for_generation: bool,
    ) -> torch.Tensor:
        self.modes.append(model_output.requires_grad)
        return torch.tensor(0.25)


class _ProbeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        return self.linear(batch.text)


class _DictLoss:
    def __call__(self, **kwargs: object) -> tuple[torch.Tensor, float]:
        total = kwargs["model_output"]
        return total, 1.0


def _non_finite_loss(
    model_output: torch.Tensor,
    batch: CollatedBatch,
    *,
    require_targets_for_generation: bool,
) -> torch.Tensor:
    return torch.tensor(float("nan"))


def test_evaluate_loader_loss_restores_training_mode_on_model() -> None:
    model = _ProbeModel()
    model.train()
    loss_probe = _ModeProbeLoss()

    evaluate_loader_loss(
        model=model,
        loss_fn=loss_probe,
        loader=[_batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert model.training is True


def test_evaluate_loader_loss_keeps_eval_mode_on_model() -> None:
    model = _ProbeModel()
    model.eval()
    loss_probe = _ModeProbeLoss()

    evaluate_loader_loss(
        model=model,
        loss_fn=loss_probe,
        loader=[_batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert model.training is False


def test_evaluate_loader_loss_empty_loader_fails() -> None:
    model = _ProbeModel()

    with pytest.raises(ValueError, match="no evaluable batches"):
        evaluate_loader_loss(
            model=model,
            loss_fn=_loss,
            loader=[],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_evaluate_loader_loss_none_loader_fails() -> None:
    model = _ProbeModel()

    with pytest.raises(RuntimeError, match="evaluation loader is required"):
        evaluate_loader_loss(
            model=model,
            loss_fn=_loss,
            loader=None,
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_evaluate_loader_loss_non_finite_loss_fails() -> None:
    model = _ProbeModel()

    with pytest.raises(ValueError, match="non-finite loss"):
        evaluate_loader_loss(
            model=model,
            loss_fn=_non_finite_loss,
            loader=[_batch()],
            device=torch.device("cpu"),
            autocast_factory=nullcontext,
        )


def test_evaluate_loader_loss_dict_total_uses_exact_tensor() -> None:
    model = _ProbeModel()

    def dict_loss(
        model_output: torch.Tensor,
        batch: CollatedBatch,
        *,
        require_targets_for_generation: bool,
    ) -> dict[str, torch.Tensor]:
        return {"total": torch.tensor(4.0), "extra": torch.tensor(7.0)}

    result = evaluate_loader_loss(
        model=model,
        loss_fn=dict_loss,
        loader=[_batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert result == 4.0


def test_evaluate_loader_loss_dict_without_total_sums_tensors() -> None:
    model = _ProbeModel()

    def dict_loss(
        model_output: torch.Tensor,
        batch: CollatedBatch,
        *,
        require_targets_for_generation: bool,
    ) -> dict[str, torch.Tensor]:
        return {
            "left": torch.tensor(2.0),
            "right": torch.tensor(5.0),
            "note": "ignored",
        }

    result = evaluate_loader_loss(
        model=model,
        loss_fn=dict_loss,
        loader=[_batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert result == 7.0


def test_evaluate_loader_loss_dict_batch_moves_to_device() -> None:
    seen_device: list[torch.device] = []

    class DictModel(torch.nn.Module):
        def forward(self, **batch: torch.Tensor) -> torch.Tensor:
            seen_device.append(batch["text"].device)
            return batch["text"].sum().unsqueeze(0)

    def tensor_loss(
        model_output: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        require_targets_for_generation: bool,
    ) -> torch.Tensor:
        return torch.tensor(1.0)

    target = torch.device("cpu")
    result = evaluate_loader_loss(
        model=DictModel(),
        loss_fn=tensor_loss,
        loader=[{"text": torch.tensor([[1.0]])}],
        device=target,
        autocast_factory=nullcontext,
    )

    assert seen_device == [target]
    assert result == 1.0


def test_evaluate_loader_loss_collated_batch_uses_move_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluator.loss as loss_module

    calls: list[tuple[object, object]] = []

    def bound_move(
        batch: CollatedBatch, device: torch.device
    ) -> CollatedBatch:
        calls.append((batch, device))
        if device is not None:
            return batch

    monkeypatch.setattr(loss_module, "move_batch_to_device", bound_move)

    target = torch.device("cpu")
    result = evaluate_loader_loss(
        model=_ProbeModel(),
        loss_fn=_loss,
        loader=[_batch()],
        device=target,
        autocast_factory=nullcontext,
    )

    assert calls
    assert calls[0][1] == target
    assert isinstance(result, float)


def test_evaluate_loader_loss_reduces_distributed_total_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluator.loss as loss_module

    reduce_calls: list[int] = []

    class _FakeDist:
        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return True

        def all_reduce(self, tensor: torch.Tensor, op: object) -> None:
            reduce_calls.append(int(op))

        class ReduceOp:
            SUM = 0

    monkeypatch.setattr(loss_module, "dist", _FakeDist())

    result = evaluate_loader_loss(
        model=_ProbeModel(),
        loss_fn=_loss,
        loader=[_batch(), _batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert len(reduce_calls) == 1
    assert isinstance(result, float)


def test_evaluate_final_losses_calls_loader_evaluator_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluator.loss as loss_module

    calls: list[str] = []

    def fake_loader_loss(**kwargs: object) -> float:
        calls.append(str(kwargs["loader"]))
        return 1.0

    monkeypatch.setattr(loss_module, "evaluate_loader_loss", fake_loader_loss)

    train_loss, val_loss, test_loss = evaluate_final_losses(
        model=_ProbeModel(),
        loss_fn=_loss,
        train_loader=[_batch()],
        val_loader=[_batch()],
        test_loader=[_batch()],
        device=torch.device("cpu"),
        autocast_factory=nullcontext,
    )

    assert len(calls) == 3
    assert (train_loss, val_loss, test_loss) == (1.0, 1.0, 1.0)
