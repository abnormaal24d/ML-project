"""Profile identity: exactly three official runtime profiles.

A profile describes how the application behaves (test/dev/prod), not the
state of a model or release. Candidate/accepted/rejected are release
process states and are deliberately not profiles.
"""

from __future__ import annotations

from typing import Final, Literal

Profile = Literal["test", "dev", "prod"]

PROFILES: Final[tuple[Profile, ...]] = ("test", "dev", "prod")

PROFILE_NAMES: Final[set[str]] = set(PROFILES)


def normalize_profile(value: str) -> Profile:
    """Normalize a requested profile name and reject unknown values."""

    profile = value.strip().lower()
    if profile in PROFILES:
        return profile
    raise ValueError(
        f"unknown profile: {value!r}; expected one of {', '.join(PROFILES)}"
    )
