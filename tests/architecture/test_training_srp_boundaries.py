"""Regression checks for the completed training-domain SRP migration."""

from __future__ import annotations

import ast
from collections import defaultdict
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
REMOVED_MODULE_PREFIXES = (
    "training.materialization",
    "training.pairing",
    "training.samples",
    "training.selection",
    "training.snapshots",
    "training.tokenization",
    "training.runtime.training_epoch_support",
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


def test_removed_training_modules_have_no_imports_or_files() -> None:
    violations: list[str] = []
    modules = _production_modules()
    for module, path in modules.items():
        if any(
            _matches_prefix(module, prefix)
            for prefix in REMOVED_MODULE_PREFIXES
        ):
            violations.append(
                f"removed module still exists: {path.relative_to(PROJECT_ROOT)}"
            )
        for line, imported in _imports(module=module, path=path):
            if any(
                _matches_prefix(imported, prefix)
                for prefix in REMOVED_MODULE_PREFIXES
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"imports removed module {imported!r}"
                )
    assert not violations, "Training migration regressions:\n" + "\n".join(
        sorted(violations)
    )


def test_training_status_and_release_contracts_have_correct_owners() -> None:
    misplaced = (
        PROJECT_ROOT / "orchestration/runtime/training_job_status.py",
        PROJECT_ROOT / "config/releases/release_status.py",
        PROJECT_ROOT / "config/releases/release_reasons.py",
    )
    assert not [path for path in misplaced if path.exists()]

    expected = (
        PROJECT_ROOT / "training/runtime/job_status/store.py",
        PROJECT_ROOT / "training/runtime/job_status/persistence.py",
        PROJECT_ROOT / "training/runtime/job_status/payloads.py",
        PROJECT_ROOT / "schemas/release.py",
    )
    assert all(path.is_file() for path in expected)


def test_training_builder_is_only_snapshot_build_orchestration() -> None:
    path = PROJECT_ROOT / "mmcrawler_datasets/snapshots/training_builder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden_import_roots = {"hashlib", "json", "os", "shutil", "uuid"}
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not (imported_roots & forbidden_import_roots)
    assert len(source.splitlines()) < 200
    assert "generate_report(" not in source
    assert "train_vocabulary_tokenizer(" not in source
    assert "write_snapshot_rows(" not in source
    assert "os.replace(" not in source


def test_training_package_has_no_wildcard_or_private_reexports() -> None:
    violations: list[str] = []
    for path in (PROJECT_ROOT / "training").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name == "*":
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        "wildcard import"
                    )
                if path.name == "__init__.py" and alias.name.startswith("_"):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"private re-export {alias.name!r}"
                    )
    assert not violations, "Invalid training package API:\n" + "\n".join(
        sorted(violations)
    )


def test_no_import_cycle_contains_training() -> None:
    modules = _production_modules()
    graph: dict[str, set[str]] = defaultdict(set)
    known = tuple(sorted(modules, key=len, reverse=True))

    for module, path in modules.items():
        for _line, imported in _imports(module=module, path=path):
            dependency = next(
                (
                    candidate
                    for candidate in known
                    if imported == candidate
                    or imported.startswith(f"{candidate}.")
                ),
                None,
            )
            if dependency is not None and dependency != module:
                graph[module].add(dependency)

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for dependency in graph[module]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[dependency])

        if lowlinks[module] != indices[module]:
            return

        component: list[str] = []
        while True:
            dependency = stack.pop()
            on_stack.remove(dependency)
            component.append(dependency)
            if dependency == module:
                break
        if len(component) > 1 and any(
            item == "training" or item.startswith("training.")
            for item in component
        ):
            cycles.append(tuple(sorted(component)))

    for module in modules:
        if module not in indices:
            visit(module)

    assert not cycles, "Import cycles involving training:\n" + "\n".join(
        " -> ".join(cycle) for cycle in sorted(cycles)
    )


def test_bootstrap_has_no_removed_factory_compatibility_plumbing() -> None:
    forbidden_names = (
        "_legacy_factories",
        "model_factory=model_factory",
        "loss_factory=loss_factory",
        "optimizer_factory=optimizer_factory",
        "scheduler_factory=scheduler_factory",
        "model_exporter=model_exporter",
    )
    violations: list[str] = []
    for path in (PROJECT_ROOT / "orchestration" / "bootstrap").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden_names:
            if name in source:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)} contains {name!r}"
                )
    assert not violations, "Removed factory plumbing returned:\n" + "\n".join(
        sorted(violations)
    )
