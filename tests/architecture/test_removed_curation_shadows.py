"""Regression checks for removed curation/extraction dead code."""

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
_REMOVED_MEDIA_CONTEXT_MODULES = (
    f"crawler.curation.media.context.{name}"
    for name in (
        "external_transcript",
        "feed_context",
        "governance_resolver",
        "html_context",
        "media_context",
        "parent_context_resolver",
        "persisted_enrichment",
        "timed_media_assembler",
        "timed_media_context_resolver",
        "timed_media_row_builder",
        "timed_media_scoring",
        "timed_media_trainability",
        "timed_media_types",
        "timed_media_video_fields",
        "url_lookup",
        "video_keyframes",
        "video_trainability",
    )
)

REMOVED_MODULES = (
    "crawler.analysis.enrichment.video.video_stream_metadata",
    "crawler.classification.content_fingerprint",
    "crawler.extraction.assets.html.html_asset_kind_inferrer",
    "crawler.curation.media.image.pair_assembler",
    "crawler.curation.media.image.image_pair_context_resolver",
    "crawler.curation.media.image.image_pair_quality_gate",
    "crawler.curation.media.image.context_extractor",
    "crawler.curation.media.image.image_pair_enrichment_reader",
    "crawler.curation.media.image.caption_selector",
    "crawler.curation.media.audio_pair_assembler",
    "crawler.curation.media.video_pair_assembler",
    "crawler.curation.media.image.record",
    "crawler.curation.records.media_records",
    "crawler.curation.records.snapshot",
    "crawler.curation.records.snapshot_reader",
    "crawler.curation.documents.records.document",
    "crawler.curation.documents.records.chunk",
    "crawler.curation.records.document_records",
    *_REMOVED_MEDIA_CONTEXT_MODULES,
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


def _scan_imports(
    *,
    module: str,
    current_path: Path,
    tree: ast.AST,
) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = _absolute_import(
                current_module=module,
                current_path=current_path,
                node=node,
            )
            if imported:
                imports.append((node.lineno, imported))
                imports.extend(
                    (node.lineno, f"{imported}.{alias.name}")
                    for alias in node.names
                    if alias.name != "*"
                )
    return imports


def _imports(
    *,
    module: str,
    path: Path,
) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(_scan_imports(module=module, current_path=path, tree=tree))


def _imports_from_source(
    *,
    module: str,
    source: str,
) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(source)
    return tuple(
        _scan_imports(
            module=module,
            current_path=Path("example.py"),
            tree=tree,
        )
    )


def _matches_module(module: str, removed: str) -> bool:
    return module == removed or module.startswith(f"{removed}.")


def test_import_scanner_recognizes_removed_module_forms() -> None:
    cases = (
        (
            "import crawler.classification.content_fingerprint",
            "crawler.classification.content_fingerprint",
        ),
        (
            "from crawler.classification.content_fingerprint "
            "import ContentFingerprint",
            "crawler.classification.content_fingerprint",
        ),
        (
            "from crawler.classification import content_fingerprint",
            "crawler.classification.content_fingerprint",
        ),
        (
            "from crawler.curation.media.image import pair_assembler",
            "crawler.curation.media.image.pair_assembler",
        ),
        (
            "from crawler.curation.media.image.pair_assembler import "
            "PairAssembler as Assembler",
            "crawler.curation.media.image.pair_assembler",
        ),
        (
            "from crawler.curation.media.image import context_extractor, "
            "pair_assembler",
            "crawler.curation.media.image.pair_assembler",
        ),
        (
            "from crawler.classification.content_fingerprint import *",
            "crawler.classification.content_fingerprint",
        ),
    )
    for source, expected in cases:
        imports = _imports_from_source(module="crawler.example", source=source)
        imported_names = {imported for _, imported in imports}
        assert expected in imported_names, (
            f"scanner missed {expected!r} in {source!r} "
            f"(got {sorted(imported_names)})"
        )


def test_removed_curation_modules_do_not_return() -> None:
    for module_name in REMOVED_MODULES:
        parts = module_name.split(".")
        path = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
        assert not path.exists(), (
            f"removed module returned: {module_name} at {path}"
        )


def test_removed_curation_module_imports_are_forbidden() -> None:
    violations: list[str] = []
    modules = _production_modules()
    for module, path in modules.items():
        for line, imported in _imports(module=module, path=path):
            if any(
                _matches_module(imported, removed)
                for removed in REMOVED_MODULES
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"imports removed module {imported!r}"
                )
    assert not violations, "Removed curation imports returned:\n" + "\n".join(
        sorted(violations)
    )


def test_datasets_forbids_crawler_imports() -> None:
    """mmcrawler_datasets must not import anything from crawler."""
    violations: list[str] = []
    modules = _production_modules()
    for module, path in modules.items():
        if not module.startswith("mmcrawler_datasets."):
            continue
        for line, imported in _imports(module=module, path=path):
            if _matches_module(imported, "crawler"):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"imports forbidden crawler module {imported!r}"
                )
    assert not violations, "datasets imports crawler:\n" + "\n".join(
        sorted(violations)
    )
