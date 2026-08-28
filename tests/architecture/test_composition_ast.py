"""Architecture AST tests for composition layer.

Enforces ADR-0001 composition layer constraints:
- No runtime loops (for, while, async for)
- No async execution (await, async def)
- No runtime lifecycle calls (.start(), .run(), .crawl(), .train(), .execute(), .process(), .evaluate())
- LOC budgets (module <= 300, builder function <= 120)
- Cyclomatic complexity budget (builder <= 8)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

COMPOSITION_ROOT = (
    Path(__file__).parent.parent.parent / "orchestration" / "composition"
)

FORBIDDEN_RUNTIME_METHODS = frozenset(
    {
        "start",
        "run",
        "crawl",
        "train",
        "execute",
        "process",
        "evaluate",
    }
)

FORBIDDEN_NODE_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Await,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
)


def _get_composition_modules() -> list[Path]:
    """Get all Python modules in the composition layer."""
    return sorted(COMPOSITION_ROOT.rglob("*.py"))


def _get_builder_functions(module_path: Path) -> list[ast.FunctionDef]:
    """Extract top-level builder functions from a module."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (node.name.startswith("build_") or node.name == "build_crawler")
    ]


def _count_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, (ast.And, ast.Or)):
            # Boolean operators in conditions
            complexity += 1
    return complexity


def _has_runtime_call(node: ast.Call) -> str | None:
    """Check if a call is to a forbidden runtime method."""
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in FORBIDDEN_RUNTIME_METHODS:
            return node.func.attr
    return None


class TestCompositionLayerConstraints:
    """Tests enforcing composition layer architectural constraints."""

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_no_runtime_loops(self, module_path: Path) -> None:
        """Composition modules must not contain runtime loops."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                pytest.fail(
                    f"{module_path}:{node.lineno}: "
                    f"Runtime loop ({type(node).__name__}) found in composition layer. "
                    f"Loops belong in workflow/application layer."
                )

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_no_async_execution(self, module_path: Path) -> None:
        """Composition modules must not contain async execution."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Await, ast.AsyncFunctionDef, ast.AsyncWith)
            ):
                pytest.fail(
                    f"{module_path}:{node.lineno}: "
                    f"Async execution ({type(node).__name__}) found in composition layer. "
                    f"Async execution belongs in workflow/application layer."
                )

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_no_runtime_lifecycle_calls(self, module_path: Path) -> None:
        """Composition modules must not call runtime lifecycle methods."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method = _has_runtime_call(node)
                if method:
                    pytest.fail(
                        f"{module_path}:{node.lineno}: "
                        f"Runtime call .{method}() found in composition layer. "
                        f"Lifecycle calls belong in workflow/application layer."
                    )

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_module_loc_budget(self, module_path: Path) -> None:
        """Composition modules should not exceed 300 LOC."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        non_empty_lines = sum(
            1
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        assert non_empty_lines <= 300, (
            f"{module_path}: {non_empty_lines} LOC exceeds 300 limit. "
            f"Split into subgraphs."
        )

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_builder_function_loc_budget(self, module_path: Path) -> None:
        """Builder functions should not exceed 120 LOC."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(
                "build_"
            ):
                # Approximate LOC by counting lines in function body
                func_lines = (
                    node.end_lineno - node.lineno + 1
                    if hasattr(node, "end_lineno")
                    else 0
                )
                if func_lines > 120:
                    pytest.fail(
                        f"{module_path}:{node.lineno}: "
                        f"Builder function '{node.name}' has ~{func_lines} LOC, "
                        f"exceeds 120 limit. Decompose into subgraphs."
                    )

    @pytest.mark.parametrize("module_path", _get_composition_modules())
    def test_builder_complexity_budget(self, module_path: Path) -> None:
        """Builder functions should not exceed cyclomatic complexity of 8."""
        if module_path.name == "__init__.py":
            pytest.skip("init file")

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(
                "build_"
            ):
                complexity = _count_complexity(node)
                if complexity > 8:
                    pytest.fail(
                        f"{module_path}:{node.lineno}: "
                        f"Builder function '{node.name}' has cyclomatic complexity "
                        f"{complexity}, exceeds 8 limit. Simplify logic."
                    )


class TestWorkflowLayerConstraints:
    """Tests enforcing workflow layer architectural constraints."""

    # The workflow layer owns its phase contracts; it must not import any
    # bootstrap or composition module.
    ALLOWED_COMPOSITION_IMPORTS: frozenset[str] = frozenset()
    ALLOWED_BOOTSTRAP_IMPORTS: frozenset[str] = frozenset()

    def test_workflow_does_not_import_composition(self) -> None:
        """Workflow layer must not import concrete implementations from composition layer."""
        workflow_root = (
            Path(__file__).parent.parent.parent / "orchestration" / "workflow"
        )

        for module_path in workflow_root.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith(
                        "orchestration.composition"
                    ):
                        pytest.fail(
                            f"{module_path}:{node.lineno}: "
                            f"Workflow imports from composition: {node.module}. "
                            f"Workflow must receive dependencies via constructor injection."
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("orchestration.composition"):
                            pytest.fail(
                                f"{module_path}:{node.lineno}: "
                                f"Workflow imports from composition: {alias.name}. "
                                f"Workflow must receive dependencies via constructor injection."
                            )

    def test_workflow_does_not_import_bootstrap(self) -> None:
        """Workflow layer must not import concrete implementations from bootstrap layer."""
        workflow_root = (
            Path(__file__).parent.parent.parent / "orchestration" / "workflow"
        )

        for module_path in workflow_root.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith(
                        "orchestration.bootstrap"
                    ):
                        pytest.fail(
                            f"{module_path}:{node.lineno}: "
                            f"Workflow imports from bootstrap: {node.module}. "
                            f"Workflow must receive dependencies via constructor injection."
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("orchestration.bootstrap"):
                            pytest.fail(
                                f"{module_path}:{node.lineno}: "
                                f"Workflow imports from bootstrap: {alias.name}. "
                                f"Workflow must receive dependencies via constructor injection."
                            )

    def test_workflow_does_not_import_root_settings(self) -> None:
        """Workflow receives projected dependencies, never root Settings.

        Blocks every route into the root settings namespace: direct
        ``from config.settings.root import Settings``, module imports
        (``import config.settings.root``), and sibling imports
        (``from config.settings import root``).
        """

        workflow_root = (
            Path(__file__).parent.parent.parent / "orchestration" / "workflow"
        )

        violations: list[str] = []

        for module_path in workflow_root.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            tree = ast.parse(module_path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if not node.module:
                        continue
                    if node.module == "config.settings.root" or (
                        node.module == "config.settings"
                        and any(alias.name == "root" for alias in node.names)
                    ):
                        violations.append(f"{module_path}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "config.settings.root":
                            violations.append(f"{module_path}:{node.lineno}")

        assert not violations, (
            "Workflow must not import root Settings:\n"
            + "\n".join(sorted(violations))
        )

    def test_crawler_focus_does_not_import_root_settings(self) -> None:
        """The focus policy operates on config slices, never root Settings."""

        focus_path = (
            Path(__file__).parent.parent.parent
            / "crawler"
            / "coverage"
            / "focus.py"
        )
        tree = ast.parse(focus_path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module and node.module.startswith("config.settings."):
                pytest.fail(
                    f"{focus_path}:{node.lineno}: "
                    f"Focus policy imports {node.module}. "
                    f"Focus must operate on exact config slices only."
                )


class TestBootstrapLayerConstraints:
    """Tests enforcing bootstrap layer architectural constraints."""

    def test_application_container_does_not_retain_build_context(self) -> None:
        """The runtime container must not retain root config/build context."""

        path = (
            Path(__file__).parent.parent.parent
            / "orchestration"
            / "bootstrap"
            / "container.py"
        )

        tree = ast.parse(path.read_text(encoding="utf-8"))

        application_container = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ApplicationContainer"
        )

        fields = {
            node.target.id
            for node in application_container.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }

        assert "settings" not in fields
        assert "run_context" not in fields

    def test_bootstrap_does_not_instantiate_domain_services(self) -> None:
        """Bootstrap should not directly instantiate concrete domain services."""
        bootstrap_root = (
            Path(__file__).parent.parent.parent / "orchestration" / "bootstrap"
        )

        # Known domain service patterns that should not be instantiated in bootstrap
        forbidden_patterns = [
            "MultimodalTrainer(",
            "Crawler(",
            "DatasetWriter(",
            "UrlScheduler(",
        ]

        for module_path in bootstrap_root.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            source = module_path.read_text(encoding="utf-8")

            for pattern in forbidden_patterns:
                if pattern in source:
                    # Allow if it's in a type annotation or string
                    lines = source.splitlines()
                    for i, line in enumerate(lines, 1):
                        if pattern in line and not line.strip().startswith(
                            "#"
                        ):
                            # Check if it's a real instantiation (not type hint)
                            if "=" in line and pattern in line.split("=")[1]:
                                pytest.fail(
                                    f"{module_path}:{i}: "
                                    f"Bootstrap instantiates domain service: {pattern}. "
                                    f"Domain services must be constructed in composition layer."
                                )


class TestDomainLayerConstraints:
    """Tests enforcing domain layer architectural constraints."""

    # Product-domain packages that must never depend on the root settings
    # object. `config/` itself is excluded: it owns the settings tree.
    SETTINGS_FREE_DOMAIN_PACKAGES = (
        "crawler",
        "training",
        "preprocessing",
        "augmentation",
        "mmcrawler_datasets",
        "datachecker",
        "release",
        "evaluator",
    )

    def test_product_domains_do_not_import_root_settings(self) -> None:
        """Product-domain services receive subconfigs, not root Settings."""

        project_root = Path(__file__).parent.parent.parent

        violations: list[str] = []

        for package in self.SETTINGS_FREE_DOMAIN_PACKAGES:
            pkg_path = project_root / package
            if not pkg_path.exists():
                continue

            for module_path in pkg_path.rglob("*.py"):
                if module_path.name == "__init__.py":
                    continue

                tree = ast.parse(module_path.read_text(encoding="utf-8"))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "config.settings.root":
                            violations.append(f"{module_path}:{node.lineno}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "config.settings.root":
                                violations.append(
                                    f"{module_path}:{node.lineno}"
                                )

        assert not violations, (
            "Product domains must not import root Settings:\n"
            + "\n".join(sorted(violations))
        )

    def test_domain_does_not_import_orchestration(self) -> None:
        """Domain packages must not import from orchestration."""
        domain_packages = [
            "crawler",
            "training",
            "preprocessing",
            "mmcrawler_datasets",
            "datachecker",
            "release",
            "evaluator",
            "config",
            "logger",
            "schemas",
        ]

        project_root = Path(__file__).parent.parent.parent

        for pkg in domain_packages:
            pkg_path = project_root / pkg
            if not pkg_path.exists():
                continue

            for module_path in pkg_path.rglob("*.py"):
                if module_path.name == "__init__.py":
                    continue

                source = module_path.read_text(encoding="utf-8")
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith(
                            "orchestration."
                        ):
                            pytest.fail(
                                f"{module_path}:{node.lineno}: "
                                f"Domain imports from orchestration: {node.module}. "
                                f"Domain must not depend on orchestration."
                            )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("orchestration."):
                                pytest.fail(
                                    f"{module_path}:{node.lineno}: "
                                    f"Domain imports from orchestration: {alias.name}. "
                                    f"Domain must not depend on orchestration."
                                )
