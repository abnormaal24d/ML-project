import ast
from pathlib import Path


def _find_violations(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<syntax-error>"]

    violations: list[str] = []
    ALLOWED_CALLS = {
        "tuple",
        "frozenset",
        "dict",
        "set",
        "list",
        "range",
        "MappingProxyType",
    }

    def _is_allowed_call(call: ast.Call) -> bool:
        # allow simple builtin collection constructors and MappingProxyType
        if isinstance(call.func, ast.Name) and call.func.id in ALLOWED_CALLS:
            return True
        return False

    for node in tree.body:
        # module-level expression statements that are calls (e.g., some_func())
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if not _is_allowed_call(node.value):
                violations.append(
                    f"Expr call at module level: line {node.lineno}"
                )
        # module-level assignments whose value is a call (e.g., X = Something())
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if not _is_allowed_call(node.value):
                targets = ",".join(
                    getattr(t, "id", "<complex>")
                    for t in node.targets
                    if isinstance(t, ast.Name)
                )
                violations.append(
                    f"Assign call at module level ({targets}): line {node.lineno}"
                )
        # AnnAssign (variable: Type = Call)
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.value, ast.Call
        ):
            if not _is_allowed_call(node.value):
                target = getattr(node.target, "id", "<complex>")
                violations.append(
                    f"Annotated assign call at module level ({target}): line {node.lineno}"
                )
    return violations


def test_multimodal_no_top_level_initialization():
    base = Path(__file__).resolve().parents[2] / "multimodal"
    assert base.exists(), f"multimodal package not found at {base}"

    files_with_violations: dict[str, list[str]] = {}
    for path in sorted(base.rglob("*.py")):
        violations = _find_violations(path)
        if violations:
            files_with_violations[str(path.relative_to(base))] = violations

    if files_with_violations:
        # Provide a readable assertion message listing files and reasons.
        lines = [
            "Top-level calls/initializations detected in multimodal package:"
        ]
        for fn, vs in files_with_violations.items():
            lines.append(f"- {fn}:")
            for v in vs:
                lines.append(f"    {v}")
        raise AssertionError("\n".join(lines))
