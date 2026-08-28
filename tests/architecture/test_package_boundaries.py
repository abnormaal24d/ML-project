"""Machine-enforced package boundaries from ADR-0001."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _package_root(package: str) -> Path:
    return PROJECT_ROOT.joinpath(*package.split("."))


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    source: str
    forbidden_imports: tuple[str, ...]
    allowed_imports: tuple[tuple[str, str], ...] = ()


BOUNDARY_RULES = (
    BoundaryRule(
        source="shared",
        # Shared adapters are a lowest-level reusable layer. They may depend
        # on the standard library and third-party backends, not product
        # packages or application composition.
        forbidden_imports=(
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
            "release",
            "schemas",
            "training",
        ),
    ),
    BoundaryRule(
        source="config",
        forbidden_imports=(
            "augmentation",
            "crawler",
            "datachecker",
            "evaluator",
            "logger",
            "mmcrawler_datasets",
            "multimodal",
            "orchestration",
            "preprocessing",
            "release",
            "shared",
            "training",
        ),
    ),
    BoundaryRule(
        source="schemas",
        forbidden_imports=("config", "multimodal", "orchestration"),
    ),
    BoundaryRule(
        source="augmentation",
        forbidden_imports=(
            "orchestration",
            "crawler.runtime",
            "training",
        ),
    ),
    BoundaryRule(
        source="crawler",
        forbidden_imports=(
            "orchestration",
            "training",
            "datachecker",
        ),
    ),
    BoundaryRule(
        source="crawler.runtime",
        forbidden_imports=(
            "orchestration",
            "training",
            "crawler.storage.datasets",
            "mmcrawler_datasets",
        ),
    ),
    BoundaryRule(
        source="crawler.storage.datasets",
        forbidden_imports=(
            "orchestration",
            "crawler.runtime",
            "training",
        ),
    ),
    BoundaryRule(
        source="preprocessing",
        forbidden_imports=(
            "orchestration",
            "crawler",
        ),
    ),
    BoundaryRule(
        source="training",
        forbidden_imports=(
            "orchestration",
            "crawler.runtime",
        ),
    ),
    BoundaryRule(
        source="datachecker",
        forbidden_imports=(
            "orchestration",
            "crawler",
            "training.acceptance",
        ),
    ),
    BoundaryRule(
        source="multimodal",
        forbidden_imports=("orchestration",),
    ),
    BoundaryRule(
        source="mmcrawler_datasets",
        forbidden_imports=(
            "orchestration",
            "crawler",
            "training",
        ),
    ),
    BoundaryRule(
        source="evaluator",
        forbidden_imports=(
            "orchestration",
            "crawler.runtime",
            "training",
            "preprocessing",
        ),
    ),
)


def _imported_modules(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    imports: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)

        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
        ):
            imports.append((node.lineno, node.module))

    return tuple(imports)


def _matches_prefix(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(f"{forbidden}.")


def test_package_boundaries() -> None:
    violations: list[str] = []

    for rule in BOUNDARY_RULES:
        package_root = _package_root(rule.source)

        for path in package_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue

            relative_posix = path.relative_to(PROJECT_ROOT).as_posix()
            for line_number, imported_module in _imported_modules(path):
                if any(
                    relative_posix.startswith(allowed_path)
                    and _matches_prefix(imported_module, allowed_prefix)
                    for allowed_path, allowed_prefix in rule.allowed_imports
                ):
                    continue
                forbidden = next(
                    (
                        prefix
                        for prefix in rule.forbidden_imports
                        if _matches_prefix(imported_module, prefix)
                    ),
                    None,
                )

                if forbidden is None:
                    continue

                relative_path = path.relative_to(PROJECT_ROOT)
                violations.append(
                    f"{relative_path}:{line_number}: "
                    f"{rule.source} imports forbidden module "
                    f"{imported_module!r}"
                )

    assert not violations, "Package-boundary violations:\n" + "\n".join(
        sorted(violations)
    )


def test_removed_legacy_configuration_packages_are_not_imported() -> None:
    """Keep the canonical Settings tree and loader as the only load path."""

    forbidden = ("config.loader", "config.settings_tree")
    violations: list[str] = []
    for package in ("config", "orchestration"):
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for line_number, imported_module in _imported_modules(path):
                if any(
                    _matches_prefix(imported_module, prefix)
                    for prefix in forbidden
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{line_number}: "
                        f"imports removed legacy config module "
                        f"{imported_module!r}"
                    )

    assert not violations, "Legacy configuration imports:\n" + "\n".join(
        sorted(violations)
    )


def test_removed_legacy_configuration_paths_do_not_exist() -> None:
    """Prevent deleted compatibility trees from silently returning."""

    removed_paths = (
        "config/loader",
        "config/settings_tree",
        "config/files/defaults",
        "config/files/environments",
        "config/files/multimodal_profiles",
    )
    existing = [
        relative
        for relative in removed_paths
        if (PROJECT_ROOT / relative).exists()
        and any(
            path.name != "__pycache__"
            for path in (PROJECT_ROOT / relative).iterdir()
        )
    ]

    assert existing == [], "Legacy configuration paths remain: " + ", ".join(
        existing
    )


def test_all_package_initializers_are_byte_empty() -> None:
    package_roots = (
        *(PROJECT_ROOT / rule.source for rule in BOUNDARY_RULES),
        PROJECT_ROOT / "logger",
        PROJECT_ROOT / "release",
        PROJECT_ROOT / "tests",
    )
    offenders = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for package_root in package_roots
        for path in package_root.rglob("__init__.py")
        if path.stat().st_size != 0 and "__pycache__" not in path.parts
    )

    assert offenders == [], (
        "__init__.py files must be exactly 0 bytes:\n" + "\n".join(offenders)
    )
