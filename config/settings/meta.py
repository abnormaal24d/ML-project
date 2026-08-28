"""Resolved configuration identity and provenance metadata."""

from __future__ import annotations

from config.base.settings_model import SettingsModel
from config.profiles import Profile


class ConfigMeta(SettingsModel):
    """Profile identity and fingerprint of the resolved settings."""

    profile: Profile
    sha256: str
