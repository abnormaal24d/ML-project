"""Environment-access boundary from ADR-0004 (rules 4 and 10)."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ENVIRONMENT_READERS = frozenset(
    {
        "config/environment/runtime_environment.py",
        "config/environment/source_selection.py",
    }
)
PRODUCTION_PACKAGES = (
    "augmentation",
    "config",
    "crawler",
    "datachecker",
    "evaluator",
    "logger",
    "mmcrawler_datasets",
    "multimodal",
    "orchestration",
    "preprocessing",
    "schemas",
    "shared",
    "training",
)


def _uses_environment_api(path: Path) -> tuple[int, str] | None:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in {"getenv", "environ"}:
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id == "os":
            return node.lineno, node.attr
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        ):
            return node.lineno, node.attr
    return None


def test_direct_environment_access_only_in_config_facades() -> None:
    violations: list[str] = []

    for package in PRODUCTION_PACKAGES:
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            relative_path = path.relative_to(PROJECT_ROOT)
            if "__pycache__" in relative_path.parts:
                continue

            result = _uses_environment_api(path)
            if result is None:
                continue
            if (
                str(relative_path).replace("\\", "/")
                in ALLOWED_ENVIRONMENT_READERS
            ):
                continue

            line_number, attribute = result
            violations.append(
                f"{relative_path}:{line_number}: direct os.{attribute} access"
            )

    assert not violations, (
        "Direct environment access outside config facades "
        "(ADR-0004 rule 4):\n" + "\n".join(sorted(violations))
    )
