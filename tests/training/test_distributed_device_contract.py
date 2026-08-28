from __future__ import annotations

import pytest
import torch

from config.multimodal.training_settings import TrainingSettings
from training.runtime.device import (
    maybe_init_distributed,
    resolve_device,
    wrap_distributed_model,
)


def test_explicit_cuda_never_downgrades_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")


def test_ddp_requires_a_distributed_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="requires WORLD_SIZE > 1"):
        maybe_init_distributed(
            settings=TrainingSettings(distributed_strategy="ddp"),
            device=torch.device("cpu"),
        )


def test_invalid_distributed_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "not-an-integer")

    with pytest.raises(ValueError, match="WORLD_SIZE must be an integer"):
        maybe_init_distributed(
            settings=TrainingSettings(distributed_strategy="auto"),
            device=torch.device("cpu"),
        )


def test_ddp_supports_task_routed_unused_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ddp(model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        fake_ddp,
    )
    model = torch.nn.Linear(2, 2)

    wrapped = wrap_distributed_model(
        model=model,
        settings=TrainingSettings(distributed_strategy="ddp"),
        device=torch.device("cpu"),
        distributed_context={"enabled": True, "strategy": "ddp"},
    )

    assert wrapped is model
    assert captured["find_unused_parameters"] is True
