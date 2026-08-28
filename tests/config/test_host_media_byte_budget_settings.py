"""Configuration contract for per-host media byte budgets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.collection.discovery import SchedulingSettings


def test_host_media_byte_budgets_are_configurable() -> None:
    settings = SchedulingSettings(
        max_media_bytes_per_host=1_000,
        max_media_bytes_per_host_by_kind={"image": 700, "video": 0},
    )

    assert settings.max_media_bytes_per_host == 1_000
    assert settings.max_media_bytes_per_host_by_kind == {
        "image": 700,
        "video": 0,
    }


@pytest.mark.parametrize(
    "per_kind",
    [
        {"page": 1},
        {"image": -1},
        {"image": 1_001},
    ],
)
def test_host_media_byte_budgets_reject_unsafe_values(
    per_kind: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        SchedulingSettings(
            max_media_bytes_per_host=1_000,
            max_media_bytes_per_host_by_kind=per_kind,
        )
