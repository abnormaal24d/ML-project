"""Resolved filesystem path settings.

Profile files may provide relative strings; the loader resolves them once at
the configuration boundary.  Runtime code therefore always receives absolute
``Path`` values for root/data/cache/output.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import Field

from config.base.settings_model import SettingsModel


class PathSettings(SettingsModel):
    """Project root plus resolved data/cache/output directories."""

    root: Path = Field(default_factory=Path.cwd)
    data: Path = Field(default=cast(Path, "data"))
    cache: Path = Field(default=cast(Path, "runtime/cache"))
    output: Path = Field(default=cast(Path, "runtime/output"))
