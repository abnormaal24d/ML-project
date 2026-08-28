from __future__ import annotations

import pytest

from config.coverage.settings import CoverageSettings, CoverageTargetSettings
from crawler.coverage.state import CoverageState


def test_snapshot_is_immutable_and_state_version_tracks_real_changes() -> None:
    state = CoverageState.from_settings(
        CoverageSettings(
            targets=CoverageTargetSettings(
                modality_targets={"image": 2, "audio": 1}
            )
        )
    )
    first = state.snapshot()
    assert first.version == 0
    assert first.missing_by_kind["image"] == 2
    with pytest.raises(TypeError):
        first.missing_by_kind["image"] = 0  # type: ignore[index]

    state.record_collected(kind="image", count=1)
    second = state.snapshot()
    assert second.version == 1
    assert second.missing_by_kind["image"] == 1

    state.record_collected(kind="image", count=0)
    assert state.snapshot().version == 1
    assert not hasattr(state, "missing_by_kind")
