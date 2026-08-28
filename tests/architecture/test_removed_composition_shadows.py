"""Regression checks for removed orchestration composition shadows."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
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
REMOVED_COMPOSITION_SHADOWS = (
    "multimodal_constants.py",
    "multimodal_output_head_resolver.py",
    "multimodal_task_registry.py",
    "tokenizer_training.py",
)
REMOVED_IMPORT_PREFIXES = (
    "orchestration.composition.runtime.multimodal_constants",
    "orchestration.composition.runtime.multimodal_output_head_resolver",
    "orchestration.composition.runtime.multimodal_task_registry",
    "orchestration.composition.runtime.tokenizer_training",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _production_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for root_name in PRODUCTION_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            if "__pycache__" not in path.parts:
                modules[_module_name(path)] = path
    return modules


def _absolute_import(
    *,
    current_module: str,
    current_path: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module

    current_parts = current_module.split(".")
    package_parts = (
        current_parts
        if current_path.name == "__init__.py"
        else current_parts[:-1]
    )
    parents_to_remove = node.level - 1
    if parents_to_remove > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - parents_to_remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _imports(
    *,
    module: str,
    path: Path,
) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = _absolute_import(
                current_module=module,
                current_path=path,
                node=node,
            )
            if imported:
                imports.append((node.lineno, imported))
    return tuple(imports)


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def test_removed_composition_shadows_do_not_return() -> None:
    runtime_root = PROJECT_ROOT / "orchestration" / "composition" / "runtime"
    existing = [
        name
        for name in REMOVED_COMPOSITION_SHADOWS
        if (runtime_root / name).exists()
    ]
    assert not existing, (
        f"removed composition shadow modules returned: {existing}"
    )


def test_removed_composition_shadow_imports_are_forbidden() -> None:
    violations: list[str] = []
    modules = _production_modules()
    for module, path in modules.items():
        for line, imported in _imports(module=module, path=path):
            if any(
                _matches_prefix(imported, prefix)
                for prefix in REMOVED_IMPORT_PREFIXES
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"imports removed module {imported!r}"
                )
    assert not violations, (
        "Removed composition imports returned:\n"
        + "\n".join(sorted(violations))
    )
