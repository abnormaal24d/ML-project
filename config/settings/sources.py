"""Resolved source-registry settings for the canonical runtime tree."""

from __future__ import annotations

from pathlib import PurePath

from pydantic import Field, field_validator

from config.source_catalog.catalog_settings import SourceCatalogSettings


class SourcesSettings(SourceCatalogSettings):
    """Source-set identity plus the expanded active source profile.

    ``set`` and ``registry`` are provenance selectors from the profile TOML.
    The loader expands them into the inherited seed/scope/governance fields
    before constructing ``Settings``.  Runtime composition consumes
    ``sources.active`` and never reads configuration files itself.
    """

    set: str = Field(default="default", min_length=1)
    registry: str = Field(default="source_registry.json", min_length=1)

    # Manually-constructed Settings used by unit tests may omit a source
    # registry entirely.  Loaded runtime profiles explicitly set this True
    # after registry expansion.
    require_seed_urls: bool = False

    @field_validator("registry")
    @classmethod
    def _registry_must_be_filename(cls, value: str) -> str:
        path = PurePath(value)
        if path.name != value or value in {".", ".."}:
            raise ValueError("sources.registry must be a filename")
        return value
