"""Regression tests for release utilities canonical API."""

from __future__ import annotations

import release.release_utilities as utilities_module


def test_release_utilities_has_no_legacy_compatibility_aliases() -> None:
    """Ensure legacy private/alias names are not present in release_utilities."""
    legacy_names = {
        "_atomic_write_json",
        "_cleanup_staging_directories",
        "_sha256",
        "_fsync_tree",
        "_fsync_directory",
    }

    unexpected = legacy_names.intersection(vars(utilities_module))

    assert not unexpected, (
        f"Legacy release utility names must not be restored: "
        f"{sorted(unexpected)}"
    )


def test_release_utilities_exports_canonical_names() -> None:
    """Ensure canonical public names are available."""
    canonical_names = {
        "atomic_write_json",
        "cleanup_staging_directories",
        "sha256",
        "fsync_tree",
        "fsync_directory",
    }

    missing = canonical_names - set(vars(utilities_module).keys())

    assert not missing, (
        f"Canonical release utility names are missing: {sorted(missing)}"
    )
