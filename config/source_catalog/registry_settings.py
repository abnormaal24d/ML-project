"""Typed source registry selector settings."""

from __future__ import annotations

import re

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel

_SELECTOR_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class SourceRegistrySelector(SettingsModel):
    """One configured source-registry entry."""

    name: str
    max_seeds: int | None = Field(default=None, ge=1)

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        text = value.strip().lower()
        if not text:
            raise ValueError("source registry selector is missing a name")
        if not _SELECTOR_NAME_PATTERN.fullmatch(text):
            raise ValueError(
                "source registry selector name must match "
                f"{_SELECTOR_NAME_PATTERN.pattern!r}: {text!r}"
            )
        return text


class SourceRegistrySettings(SettingsModel):
    """Strictly typed registry source selectors."""

    selectors: tuple[SourceRegistrySelector, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_selectors(self) -> SourceRegistrySettings:
        seen: set[str] = set()
        duplicates: list[str] = []
        for selector in self.selectors:
            if selector.name in seen:
                duplicates.append(selector.name)
                continue
            seen.add(selector.name)
        if duplicates:
            raise ValueError(
                "sources.registry_sources must not select the same source "
                f"twice: {sorted(set(duplicates))}"
            )
        return self

    def selector_dicts(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "name": selector.name,
                **(
                    {"max_seeds": selector.max_seeds}
                    if selector.max_seeds is not None
                    else {}
                ),
            }
            for selector in self.selectors
        )
