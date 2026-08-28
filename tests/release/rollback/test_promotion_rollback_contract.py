from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from release import model_release_publisher as promotion_module
from release import release_utilities as utilities_module
from release.release_utilities import (
    ProductionPromotionLockError,
    ProductionPromotionValidationError,
    atomic_write_json,
    cleanup_staging_directories,
    contained_relative_path,
    current_pointer_references,
    read_json_object,
    require_separate_roots,
    required_sha256,
    safe_segment,
)
from tests.support.release_fixtures import fixture_requirements

# --- rollback decision criterion -------------------------------------------


def _publish_pointer(production: Path, release_directory: Path) -> None:
    relative = release_directory.relative_to(production).as_posix()
    atomic_write_json(
        path=production / "current.json",
        payload={"release_directory": relative},
    )


def test_pointer_reference_marks_the_published_release(
    tmp_path: Path,
) -> None:
    production = tmp_path / "prod"
    production.mkdir()
    release = production / "releases" / "release-abc"
    release.mkdir(parents=True)
    _publish_pointer(production, release)
    assert (
        current_pointer_references(
            production_directory=production,
            release_directory=release,
        )
        is True
    )


def test_pointer_reference_is_false_when_no_pointer_exists(
    tmp_path: Path,
) -> None:
    release = tmp_path / "prod" / "releases" / "release-abc"
    release.mkdir(parents=True)
    assert (
        current_pointer_references(
            production_directory=tmp_path / "prod",
            release_directory=release,
        )
        is False
    )


def test_pointer_reference_is_false_for_another_release(
    tmp_path: Path,
) -> None:
    production = tmp_path / "prod"
    first = production / "releases" / "release-aaa"
    second = production / "releases" / "release-bbb"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _publish_pointer(production, first)
    assert (
        current_pointer_references(
            production_directory=production,
            release_directory=second,
        )
        is False
    )


def test_pointer_reference_is_false_for_malformed_pointer(
    tmp_path: Path,
) -> None:
    production = tmp_path / "prod"
    release = production / "releases" / "release-abc"
    release.mkdir(parents=True)
    (production / "current.json").write_text("{not json", encoding="utf-8")
    assert (
        current_pointer_references(
            production_directory=production,
            release_directory=release,
        )
        is False
    )


def test_pointer_reference_is_false_when_release_missing(
    tmp_path: Path,
) -> None:
    production = tmp_path / "prod"
    production.mkdir()
    atomic_write_json(
        path=production / "current.json",
        payload={"release_directory": "releases/release-ghost"},
    )
    release = production / "releases" / "release-ghost"
    assert (
        current_pointer_references(
            production_directory=production,
            release_directory=release,
        )
        is False
    )


# --- atomic pointer publication --------------------------------------------


def test_atomic_write_publishes_and_leaves_no_temp_residue(
    tmp_path: Path,
) -> None:
    target = tmp_path / "current.json"
    atomic_write_json(
        path=target,
        payload={"release_id": "release-1", "ok": True},
    )
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "release_id": "release-1",
        "ok": True,
    }
    assert not list(tmp_path.glob(".current.json.*.tmp"))


def test_atomic_write_failure_leaves_target_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "current.json"
    target.write_text("prior", encoding="utf-8")

    def explode(payload: object, handle: object, **kwargs: object) -> None:
        del payload, handle, kwargs
        raise OSError("serialization exploded")

    monkeypatch.setattr(
        utilities_module.json,
        "dump",
        explode,
    )
    with pytest.raises(OSError, match="serialization exploded"):
        atomic_write_json(path=target, payload={"x": 1})

    assert target.read_text(encoding="utf-8") == "prior"
    assert not list(tmp_path.glob(".current.json.*.tmp"))


def test_staging_cleanup_removes_crashed_promotion_residue(
    tmp_path: Path,
) -> None:
    releases = tmp_path / "releases"
    (releases / ".staging-crashed").mkdir(parents=True)
    (releases / ".staging-crashed" / "checkpoint.pt").write_bytes(b"x")
    (releases / ".staging-file").write_bytes(b"y")
    (releases / "release-live").mkdir()

    cleanup_staging_directories(releases)

    assert not (releases / ".staging-crashed").exists()
    assert not (releases / ".staging-file").exists()
    assert (releases / "release-live").is_dir()


# --- promotion aborts before any mutation ----------------------------------


def test_promotion_aborts_on_missing_checkpoint_before_mutation(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    dataset = tmp_path / "dataset"
    candidate.mkdir()
    dataset.mkdir()
    production = tmp_path / "production"
    training_result = SimpleNamespace(
        artifacts=SimpleNamespace(checkpoint_path=tmp_path / "missing.pt"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="promotion checkpoint is unavailable",
    ):
        promotion_module.promote_model(
            candidate_directory=candidate,
            production_directory=production,
            evidence_bundle_path=tmp_path / "missing.json",
            settings=object(),
            training_result=training_result,
            evaluation_result=object(),
            dataset_root=dataset,
            release_requirements=fixture_requirements(),
        )

    assert not production.exists()


def test_promotion_rejects_overlapping_roots_before_mutation(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production"
    dataset = tmp_path / "dataset"
    production.mkdir()
    dataset.mkdir()

    with pytest.raises(ValueError, match="must not overlap"):
        promotion_module.promote_model(
            candidate_directory=production,
            production_directory=production,
            evidence_bundle_path=tmp_path / "missing.json",
            settings=object(),
            training_result=SimpleNamespace(
                artifacts=SimpleNamespace(
                    checkpoint_path=tmp_path / "missing.pt"
                ),
            ),
            evaluation_result=object(),
            dataset_root=dataset,
            release_requirements=fixture_requirements(),
        )


# --- rollback path containment ---------------------------------------------


def test_contained_relative_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "prod"
    root.mkdir()
    with pytest.raises(
        ProductionPromotionValidationError,
        match="production release path is unsafe",
    ):
        contained_relative_path(root=root, relative="../elsewhere")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="production release path is unsafe",
    ):
        contained_relative_path(root=root, relative="C:\\Windows\\system32")


def test_contained_relative_path_accepts_release_subpath(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prod"
    root.mkdir()
    assert (
        contained_relative_path(
            root=root,
            relative="releases/release-1",
        )
        == root / "releases" / "release-1"
    )


def test_separate_roots_contract(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not overlap"):
        require_separate_roots(
            candidate_directory=outer,
            production_directory=inner,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        require_separate_roots(
            candidate_directory=inner,
            production_directory=outer,
        )
    other = tmp_path / "other"
    other.mkdir()
    require_separate_roots(
        candidate_directory=outer,
        production_directory=other,
    )


def test_safe_segment_rejects_unsafe_artifact_names() -> None:
    for value in ("", ".", "..", "a/b", "/abs", "a\\b", "..\\x"):
        with pytest.raises(
            ValueError, match="release artifact name is unsafe"
        ):
            safe_segment(value)
    assert safe_segment("model.safetensors") == "model.safetensors"


def test_required_sha256_contract() -> None:
    with pytest.raises(
        ProductionPromotionValidationError,
        match="must be SHA-256",
    ):
        required_sha256({"value": "zzz"}, "value")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="must be SHA-256",
    ):
        required_sha256({"value": "g" * 64}, "value")
    digest = "a" * 64
    assert required_sha256({"value": digest}, "value") == digest


# --- pointer artifact reader ------------------------------------------------


def test_read_json_object_rejects_invalid_content(tmp_path: Path) -> None:
    broken = tmp_path / "current.json"
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="invalid JSON artifact",
    ):
        read_json_object(broken)

    list_root = tmp_path / "list.json"
    list_root.write_text("[]", encoding="utf-8")
    with pytest.raises(
        ProductionPromotionValidationError,
        match="root must be an object",
    ):
        read_json_object(list_root)

    oversized = tmp_path / "big.json"
    oversized.write_text(
        '{"x": ' + "1" * (1024 * 1024) + "}", encoding="utf-8"
    )
    with pytest.raises(
        ProductionPromotionValidationError,
        match="exceeds size limit",
    ):
        read_json_object(oversized)


# --- error contract ---------------------------------------------------------


def test_promotion_error_types_are_runtime_errors() -> None:
    assert issubclass(ProductionPromotionLockError, RuntimeError)
    assert issubclass(ProductionPromotionValidationError, RuntimeError)
    assert (
        promotion_module.ProductionPromotionLockError
        is ProductionPromotionLockError
    )
    assert (
        promotion_module.ProductionPromotionValidationError
        is ProductionPromotionValidationError
    )
