"""Crawler identity settings for network etiquette."""

from __future__ import annotations

import re

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class IdentitySettings(SettingsModel):
    """Fixed crawler identity. Builds token and User-Agent.

    Placeholder domains (example.org / example.invalid) are rejected so a
    product run cannot ship a fictional contact identity.
    """

    name: str = Field(default="MultimodalCrawler", min_length=1)
    version: str = Field(default="1.0", min_length=1)
    url: str = Field(
        default="https://github.com/multimodal-crawler/multimodal-crawler",
        min_length=1,
    )
    email: str = Field(
        default="crawler-ops@multimodal-crawler.dev",
        min_length=1,
    )

    @property
    def user_agent(self) -> str:
        return f"{self.name}/{self.version} (+{self.url}; mailto:{self.email})"

    @model_validator(mode="after")
    def validate_identity(self) -> IdentitySettings:
        if not self.name.strip():
            raise ValueError("identity.name must not be empty")
        if not self.version.strip():
            raise ValueError("identity.version must not be empty")
        url = self.url.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("identity.url must be absolute http(s) URL")
        lowered_url = url.casefold()
        if "example.org" in lowered_url or "example.invalid" in lowered_url:
            raise ValueError(
                "identity.url must not use placeholder example domains"
            )
        email = self.email.strip()
        if email.lower().startswith("mailto:"):
            email = email[7:].strip()
        if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) is None:
            raise ValueError("identity.email must be a valid contact address")
        email_domain = email.rsplit("@", 1)[-1].casefold()
        if email_domain in {"example.org", "example.invalid", "example.com"}:
            raise ValueError(
                "identity.email must not use placeholder example domains"
            )
        return self
