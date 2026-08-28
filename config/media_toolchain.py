"""Canonical media toolchain settings (FFmpeg/FFprobe identity and version pins).

Single owner for the shared external toolchain used by preprocessing privacy
operations, video augmentation, and runtime dependency preflight.
"""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel


class MediaToolchainSettings(SettingsModel):
    """Pinned local FFmpeg tooling identity for all consumers."""

    ffmpeg_executable: str = Field(default="ffmpeg", min_length=1)
    ffprobe_executable: str = Field(default="ffprobe", min_length=1)
    ffmpeg_expected_version: str | None = Field(
        default=None,
        pattern=r"^\d+(?:\.\d+){2}$",
    )
    ffprobe_expected_version: str | None = Field(
        default=None,
        pattern=r"^\d+(?:\.\d+){2}$",
    )
