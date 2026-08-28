"""Strict Pydantic base model for application settings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SettingsModel(BaseModel):
    """Strict, immutable base model for application settings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        str_strip_whitespace=True,
        validate_default=True,
    )
