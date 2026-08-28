from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.multimodal.training_settings import TrainingSettings
from config.settings.datasets import DatasetValidatorSettings
from schemas.multimodal_tasks import canonical_task_names


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("tasks", ("",)),
        ("approved_beta_tasks", (" ",)),
        ("curriculum_schedule", ("",)),
    ),
)
def test_training_settings_reject_blank_task_values(
    field_name: str,
    value: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(**{field_name: value})


def test_training_settings_reject_blank_task_map_key() -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(task_sampling_weights={"": 1.0})


def test_dataset_validator_rejects_blank_task_map_key_when_resolved() -> None:
    validator = DatasetValidatorSettings(min_task_samples={"": 1})

    with pytest.raises(ValueError, match="task name must not be empty"):
        validator.effective_min_task_samples()


def test_cross_section_task_normalizer_has_no_legacy_fallback() -> None:
    with pytest.raises(ValueError, match="contains an invalid task name"):
        canonical_task_names(("",), field_name="tasks")
