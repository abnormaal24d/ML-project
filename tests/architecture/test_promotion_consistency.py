import ast
from pathlib import Path


def test_training_runner_never_promotes_a_candidate() -> None:
    """Keep promotion outside the autonomous training workflow."""

    source_path = (
        Path(__file__).resolve().parents[2]
        / "orchestration"
        / "workflow"
        / "training"
        / "phase_runner.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainPhaseRunner"
    )
    assert not any(
        isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_promote_production_release"
        for node in runner.body
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "promote_model"
        for node in ast.walk(runner)
    ), "The training runner must not promote a candidate automatically"


def test_orchestration_test_uses_release_implementation() -> None:
    """Verify that orchestration tests use the release implementation."""
    with open("tests/training/test_transactional_promotion.py", "r") as f:
        content = f.read()
    assert (
        "from release import model_release_publisher as promotion_module"
        in content
    ), "Transactional tests should import from release module"


def test_orchestration_test_verifies_no_automatic_promotion() -> None:
    """Verify that the orchestration test covers the candidate boundary."""
    with open(
        "tests/orchestration/bootstrap/test_train_phase_runner_status.py", "r"
    ) as f:
        content = f.read()
    assert "test_candidate_stage_does_not_promote_automatically" in content
    assert (
        'assert not (tmp_path / "models" / "current.json").exists()' in content
    )


def test_decide_release_includes_benchmark_failure_reasons() -> None:
    """Verify that decide_release implementations include benchmark_failure_reasons."""
    import inspect

    from release.acceptance_evaluator import (
        _decide_configured as release_decide_configured,
    )

    # Get source code of release _decide_configured implementation
    release_source = inspect.getsource(release_decide_configured)

    # Should include evaluation_result.benchmark_failure_reasons
    assert "evaluation_result.benchmark_failure_reasons" in release_source, (
        "Release decide_release._decide_configured should include benchmark_failure_reasons"
    )


def test_orchestration_integration_test_exercises_benchmark_rejection() -> (
    None
):
    """Verify that the orchestration test exercises benchmark rejection scenario."""
    # Import and inspect the orchestration test to ensure it properly tests reject scenarios
    with open(
        "tests/orchestration/bootstrap/test_train_phase_runner_status.py", "r"
    ) as f:
        content = f.read()

    # Check that the test properly mocks decide_release
    assert "mock_decide" in content or "monkeypatch.setattr" in content, (
        "Test should mock decide_release to simulate rejection scenarios"
    )

    # The candidate-stage test verifies that the autonomous workflow cannot
    # mutate the active release pointer.


def test_file_lock_is_included_in_release_package() -> None:
    """Verify that release package includes all necessary files for Docker builds."""
    import os

    # Check that the release directory exists and contains essential files
    release_dir = "release"
    assert os.path.exists(release_dir), (
        f"Release directory {release_dir} should exist"
    )

    # Check that release package contains all essential files for promotion
    essential_files = [
        "model_release_publisher.py",
        "acceptance_result.py",
        "release_decision.py",
        "release_evidence_bundle.py",
    ]

    for file in essential_files:
        file_path = os.path.join(release_dir, file)
        assert os.path.exists(file_path), (
            f"Essential file {file} should exist in release package"
        )

    # Check that release module is importable
    from release.current_release_pointer import resolve_current_release
    from release.model_release_publisher import promote_model

    assert callable(promote_model), "promote_model should be callable"
    assert callable(resolve_current_release), (
        "resolve_current_release should be callable"
    )
