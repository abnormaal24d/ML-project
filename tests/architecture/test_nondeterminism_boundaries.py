"""Semantic nondeterminism and operational entropy have distinct boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _call_names(relative_path: str) -> frozenset[str]:
    tree = ast.parse(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    )
    return frozenset(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def test_evidence_sensitive_modules_do_not_read_wall_clock() -> None:
    for relative_path in (
        "preprocessing/privacy/artifacts.py",
        "training/export/export.py",
    ):
        calls = _call_names(relative_path)
        assert "datetime.now" not in calls, relative_path
        assert "datetime.datetime.now" not in calls, relative_path


def test_crawl_task_identity_has_no_random_fallback() -> None:
    calls = _call_names("crawler/crawl_tasks/crawl_task.py")

    assert "uuid.uuid4" not in calls
    assert "uuid4" not in calls


def test_runtime_primitive_contracts_have_one_owner() -> None:
    canonical_path = PROJECT_ROOT / "shared" / "runtime_primitives.py"
    violations: list[str] = []
    for package in (
        "augmentation",
        "crawler",
        "datachecker",
        "evaluator",
        "mmcrawler_datasets",
        "multimodal",
        "orchestration",
        "preprocessing",
        "release",
        "shared",
        "training",
    ):
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name in {"Clock", "IdGenerator"}
                    and path != canonical_path
                ):
                    violations.append(
                        f"{path}:{node.lineno} duplicates {node.name}"
                    )
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "crawler.runtime.runtime_dependencies"
                    and {alias.name for alias in node.names}
                    & {"Clock", "IdGenerator"}
                ):
                    violations.append(
                        f"{path}:{node.lineno} imports a retired runtime alias"
                    )
    assert not violations, "\n".join(violations)


def test_operational_entropy_remains_nondeterministic() -> None:
    privacy_calls = _call_names("preprocessing/privacy/artifacts.py")

    assert "secrets.token_hex" in privacy_calls
    assert "tempfile.mkdtemp" in privacy_calls
