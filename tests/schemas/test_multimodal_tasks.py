from __future__ import annotations

import pytest

from schemas.multimodal_tasks import canonical_task_name


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("text_to_image", "text_to_image"),
        (" Text-To-Image ", "text_to_image"),
        ("task123", "task123"),
    ),
)
def test_canonical_task_name_normalizes_serialized_values(
    value: object,
    expected: str,
) -> None:
    assert canonical_task_name(value) == expected


@pytest.mark.parametrize("value", ("", "   ", "invalid task", "_task"))
def test_canonical_task_name_rejects_empty_or_invalid_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        canonical_task_name(value)
