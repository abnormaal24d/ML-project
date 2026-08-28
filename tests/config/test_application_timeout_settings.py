"""Application timeout settings remain explicit and validated."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings.app import AppSettings


def test_data_checker_timeout_defaults_to_one_minute() -> None:
    settings = AppSettings()

    assert settings.data_checker_timeout_seconds == 60.0


@pytest.mark.parametrize("timeout_seconds", (0.0, -1.0))
def test_data_checker_timeout_must_be_positive(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        AppSettings(data_checker_timeout_seconds=timeout_seconds)
