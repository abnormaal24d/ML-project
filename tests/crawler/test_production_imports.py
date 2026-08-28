"""Production import test: all crawler modules must be importable.

This test catches import failures like the one where
crawl_state_reader/crawl_state_writer referenced the removed
CrawlerRuntimeSettings name.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def test_all_crawler_modules_import() -> None:
    """Import every Python module under crawler/ to catch stale imports."""

    crawler_root = Path(__file__).parent.parent / "crawler"
    assert crawler_root.is_dir()

    failed: list[tuple[str, str]] = []

    for module_info in pkgutil.walk_packages([str(crawler_root)], "crawler."):
        name = module_info.name
        if "__pycache__" in name:
            continue
        # Skip test modules - they're not part of production code
        if (
            name.endswith("_test")
            or ".test_" in name
            or name == "crawler.test_production_imports"
        ):
            continue
        # Skip test files that happen to be at module level
        if name.split(".")[-1].startswith("test_"):
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    assert not failed, (
        f"{len(failed)} crawler module(s) failed to import:\n"
        + "\n".join(f"  {name}: {err}" for name, err in failed)
    )


def test_crawler_subpackage_imports() -> None:
    """Import key crawler subpackages to ensure no circular/stale issues."""

    subpackages = [
        "crawler.runtime",
        "crawler.runtime.state",
        "crawler.runtime.loop",
        "crawler.runtime.actions",
        "crawler.runtime.feedback",
        "crawler.runtime.control",
        "crawler.scheduling",
        "crawler.scheduling.admission",
        "crawler.scheduling.completion",
        "crawler.scheduling.dispatch",
        "crawler.scheduling.host_control",
        "crawler.scheduling.priority",
        "crawler.scheduling.queueing",
        "crawler.scheduling.dedupe",
        "crawler.scheduling.checkpointing",
        "crawler.worker",
        "crawler.worker.pool",
        "crawler.governance",
        "crawler.governance.url_filter",
        "crawler.governance.robots",
        "crawler.governance.network_access",
        "crawler.fetching",
        "crawler.fetching.network",
        "crawler.extraction",
        "crawler.extraction.modalities",
        "crawler.extraction.payloads",
        "crawler.storage",
        "crawler.storage.datasets",
        "crawler.crawl_tasks",
        "crawler.classification",
        "crawler.discovery",
        "crawler.numeric",
    ]

    failed: list[tuple[str, str]] = []
    for pkg in subpackages:
        try:
            importlib.import_module(pkg)
        except Exception as exc:  # noqa: BLE001
            failed.append((pkg, f"{type(exc).__name__}: {exc}"))

    assert not failed, (
        f"{len(failed)} crawler subpackage(s) failed to import:\n"
        + "\n".join(f"  {pkg}: {err}" for pkg, err in failed)
    )


if __name__ == "__main__":
    test_all_crawler_modules_import()
    test_crawler_subpackage_imports()
    print("All crawler imports OK")
