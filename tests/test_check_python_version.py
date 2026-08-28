from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.check_python_version import SUPPORTED_PYTHON, is_supported


def test_only_python_312_is_supported() -> None:
    assert is_supported((3, 12))
    assert not is_supported((3, 11))
    assert not is_supported((3, 13))
    assert not is_supported((3, 14))


def test_interpreter_guard_matches_package_metadata() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert SUPPORTED_PYTHON == (3, 12)


def test_sdist_manifest_includes_documented_install_inputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = (project_root / "MANIFEST.in").read_text(encoding="utf-8")

    for path in (
        "Makefile",
        "scripts/check_python_version.py",
        "requirements/constraints-py312-cpu.txt",
        "requirements/constraints-py312-preprocessing.txt",
    ):
        assert f"include {path}" in manifest
