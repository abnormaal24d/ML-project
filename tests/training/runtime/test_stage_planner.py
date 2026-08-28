from __future__ import annotations

import pytest

from training.runtime.job_status.models import TrainingStageState
from training.runtime.planner import (
    StageExecutionArtifact,
    TrainingStage,
    TrainingStageExecutionError,
    TrainingStageExecutor,
)


def test_stage_executor_resumes_after_last_completed_stage() -> None:
    persisted: list[TrainingStageState] = []
    state = TrainingStageState(
        current_stage=TrainingStage.INSTRUCTION_TUNING,
        completed_stages=(
            TrainingStage.DATASET_FREEZE,
            TrainingStage.TOKENIZER_BUILD,
            TrainingStage.MODALITY_PRETRAIN,
            TrainingStage.CROSS_MODAL_ALIGNMENT,
            TrainingStage.MULTIMODAL_PRETRAIN,
        ),
        input_fingerprint="pretrain-sha",
        parent_checkpoint="pretrain.pt",
    )
    calls: list[tuple[str | None, str | None]] = []

    def instruction(
        input_fingerprint: str | None,
        parent_checkpoint: str | None,
    ) -> StageExecutionArtifact:
        calls.append((input_fingerprint, parent_checkpoint))
        return StageExecutionArtifact(
            output_fingerprint="instruction-sha",
            parent_checkpoint="instruction.pt",
        )

    executor = TrainingStageExecutor(
        handlers={TrainingStage.INSTRUCTION_TUNING: instruction},
        load_state=lambda: state,
        persist_state=persisted.append,
    )

    result = executor.run(
        start=TrainingStage.INSTRUCTION_TUNING,
        stop=TrainingStage.INSTRUCTION_TUNING,
    )

    assert calls == [("pretrain-sha", "pretrain.pt")]
    assert isinstance(result, TrainingStageState)
    assert result.current_stage is TrainingStage.PREFERENCE_TUNING
    assert result.output_fingerprint == "instruction-sha"
    assert result.parent_checkpoint == "instruction.pt"
    assert persisted[-1] == result


def test_stage_executor_persists_resumable_failure_and_retries() -> None:
    persisted: list[TrainingStageState] = []
    attempts = 0

    def preference(
        _input_fingerprint: str | None,
        _parent_checkpoint: str | None,
    ) -> StageExecutionArtifact:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("worker interrupted")
        return StageExecutionArtifact(output_fingerprint="preference-sha")

    current: TrainingStageState | None = None

    def load() -> TrainingStageState | None:
        return current

    def persist(value: object) -> None:
        nonlocal current
        assert isinstance(value, TrainingStageState)
        current = value
        persisted.append(value)

    executor = TrainingStageExecutor(
        handlers={TrainingStage.PREFERENCE_TUNING: preference},
        load_state=load,
        persist_state=persist,
    )

    with pytest.raises(
        TrainingStageExecutionError, match="worker interrupted"
    ):
        executor.run(
            start=TrainingStage.PREFERENCE_TUNING,
            stop=TrainingStage.PREFERENCE_TUNING,
        )

    assert current is not None
    assert current.failed_stage is TrainingStage.PREFERENCE_TUNING

    result = executor.run(
        start=TrainingStage.PREFERENCE_TUNING,
        stop=TrainingStage.PREFERENCE_TUNING,
    )
    assert isinstance(result, TrainingStageState)
    assert attempts == 2
    assert result.failed_stage is None
    assert result.current_stage is TrainingStage.SAFETY_TUNING
    assert any(state.attempt == 2 for state in persisted)
