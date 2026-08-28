from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from crawler.analysis.enrichment.video.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)
from crawler.processing.page_discovery_reporting import (
    AdmissionCounts,
    PageDiscoveryReporting,
)
from crawler.processing.page_discovery_scheduler_admission import (
    PageDiscoverySchedulerAdmission,
)
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowAction,
    WorkflowDecisionReason,
    decide_workflow_action,
)
from multimodal.model.contracts import CollatedBatch
from orchestration.composition.runtime.fetch import normalized_host_allowlist
from training.runtime.loop.runner import run_training_loop
from training.runtime.precision import PrecisionRuntime


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class _SignalTracker:
    def __init__(self) -> None:
        self.backward_calls = 0

    def record_after_backward(self) -> None:
        self.backward_calls += 1


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch: CollatedBatch) -> dict[str, torch.Tensor]:
        return {"logits": self.linear(batch.text)}


class _TrainingLoss(torch.nn.Module):
    def forward(
        self,
        *,
        model_output: dict[str, torch.Tensor],
        batch: CollatedBatch,
        require_targets_for_generation: bool,
    ) -> torch.Tensor:
        assert require_targets_for_generation
        assert batch.labels is not None
        return torch.nn.functional.mse_loss(
            model_output["logits"], batch.labels
        )


def _batch() -> CollatedBatch:
    return CollatedBatch(
        sample_ids=["sample"],
        text=torch.tensor([[1.0]]),
        image=torch.zeros((1, 1)),
        audio=torch.zeros((1, 1)),
        video=torch.zeros((1, 1)),
        modality_mask=torch.ones((1, 4)),
        labels=torch.tensor([[0.0]]),
    )


def _valid(_phase: WorkflowAction) -> ValidationResult:
    return ValidationResult.valid(
        reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
    )


def test_workflow_rules_preserve_priority_and_special_routing() -> None:
    plan = decide_workflow_action(
        crawl=_valid(WorkflowAction.CRAWL),
        preprocessing=ValidationResult.invalid(
            reason=WorkflowDecisionReason.TRAINING_SNAPSHOT_INVALID,
        ),
        augmentation=_valid(WorkflowAction.AUGMENT),
        training=_valid(WorkflowAction.TRAIN),
        ordered_actions=(
            WorkflowAction.CRAWL,
            WorkflowAction.PREPROCESS,
            WorkflowAction.TRAIN,
        ),
        optional_actions=(WorkflowAction.AUGMENT,),
        seed_url_count=1,
        require_seed_urls=True,
        augmentation_enabled=False,
        training_input_mode_is_augmented_required=False,
        training_root=Path("training-snapshot"),
        dataset_manifest_hash="dataset-fingerprint",
    )
    assert plan.action is WorkflowAction.TRAIN
    assert plan.reason is WorkflowDecisionReason.TRAINING_SNAPSHOT_INVALID


def test_workflow_context_block_rule_runs_first() -> None:
    plan = decide_workflow_action(
        crawl=_valid(WorkflowAction.CRAWL),
        preprocessing=_valid(WorkflowAction.PREPROCESS),
        augmentation=_valid(WorkflowAction.AUGMENT),
        training=_valid(WorkflowAction.TRAIN),
        ordered_actions=(
            WorkflowAction.CRAWL,
            WorkflowAction.PREPROCESS,
            WorkflowAction.TRAIN,
        ),
        optional_actions=(WorkflowAction.AUGMENT,),
        seed_url_count=0,
        require_seed_urls=True,
        augmentation_enabled=True,
        training_input_mode_is_augmented_required=False,
    )
    assert plan.action is WorkflowAction.BLOCKED
    assert plan.reason is WorkflowDecisionReason.CONFIG_NO_SEED_URLS


def test_training_loop_uses_extracted_batch_processor() -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tracker = _SignalTracker()
    loader = [_batch()]
    state, history = run_training_loop(
        settings=SimpleNamespace(
            epochs=1,
            gradient_accumulation_steps=1,
            progress_log_interval_batches=1,
            gradient_clip_max_norm=None,
            scheduler_interval="step",
            monitor_metric="validation_loss",
            monitor_mode="min",
            early_stopping_patience=None,
            early_stopping_min_delta=0.0,
        ),
        device=torch.device("cpu"),
        logger=_Logger(),
        model=model,
        loss_fn=_TrainingLoss(),
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        signal_tracker=tracker,
        precision_runtime=PrecisionRuntime(
            name="fp32",
            device_type="cpu",
            autocast_enabled=False,
            autocast_dtype=None,
            uses_grad_scaler=False,
        ),
        distributed_context={"enabled": False, "rank": 0},
    )
    assert state.completed_epochs == 1
    assert state.total_batches == 1
    assert state.test_loss is not None
    assert len(history) == 1
    assert tracker.backward_calls == 1


def test_video_limit_normalization_isolated_from_analyzer() -> None:
    logger = _Logger()
    service = VideoFrameAnalysisService(
        frame_sampler=None,  # type: ignore[arg-type]
        frame_text_extraction_service=None,  # type: ignore[arg-type]
        transcription_executor=None,  # type: ignore[arg-type]
        keyframe_selector=lambda **_: (),
        video_reader=None,
        frame_processor=None,
        logger=logger,  # type: ignore[arg-type]
    )
    assert service.normalize_limits(
        max_sampled_frames=-4,
        max_duration_seconds=-1.0,
    ) == (0, 0.0)


def test_scheduler_pressure_policy_remains_independently_testable(
    monkeypatch,
) -> None:
    import crawler.processing.page_discovery_scheduler_admission as module

    target = SimpleNamespace(target=True)
    ordinary_a = SimpleNamespace(target=False)
    ordinary_b = SimpleNamespace(target=False)
    monkeypatch.setattr(
        module,
        "is_coverage_recovery_target_task",
        lambda task: bool(task.target),
    )
    service = object.__new__(PageDiscoverySchedulerAdmission)
    selected = service.tasks_allowed_under_admission_pressure(
        selected_tasks=(ordinary_a, target, ordinary_b),
        drain_budget=1,
    )
    assert selected == (ordinary_a, target)


def test_discovery_result_metrics_are_instance_safe() -> None:
    reporting = object.__new__(PageDiscoveryReporting)
    admission = AdmissionCounts(
        accepted=2,
        scheduler_filtered=1,
        scheduler_rejected=1,
        accepted_by_kind={},
        rejected_by_kind={},
        rejected_by_reason={},
        metrics=SimpleNamespace(as_payload=lambda: {}),
    )
    result = reporting.build_result_metrics(
        selection=SimpleNamespace(
            discovered_count=5,
            filtered_count=1,
            truncated_count=1,
            duplicate_count=0,
        ),
        budget=SimpleNamespace(discovery_scan_budget=9),
        selection_metrics={},
        admission=admission,
    )
    assert result == {
        "discovered": 5,
        "scheduled": 2,
        "filtered": 2,
        "rejected": 1,
        "scope_blocked": 0,
        "capacity_skipped": 0,
        "truncated": 1,
        "duplicates": 0,
        "discovery_scan_budget": 9,
    }


def test_fetch_host_allowlist_normalization_isolated() -> None:
    settings = SimpleNamespace(
        collection=SimpleNamespace(
            fetcher=SimpleNamespace(
                head_preflight_host_allowlist=("EXAMPLE.com", "bad host"),
            )
        )
    )
    normalizer = SimpleNamespace(
        normalize=lambda host: host.casefold() if " " not in host else None
    )
    assert normalized_host_allowlist(
        settings=settings,
        host_normalizer=normalizer,
    ) == frozenset({"example.com"})


def test_leakage_url_and_video_fingerprints_survive_decomposition(
    tmp_path: Path,
) -> None:
    from evaluator.leakage.contracts import CATEGORIES
    from evaluator.leakage.report import generate_report

    report = generate_report(
        left_records=[
            {
                "dataset_id": "left",
                "sample_id": "video-1",
                "partition": "train",
                "lineage_key": "left-video-1",
                "modality": "video",
                "source_url": "HTTPS://Example.com:443/watch?v=1",
                "content_hash": "a" * 64,
                "content_fingerprints": {
                    "video_keyframe_phashes": ["0" * 16, "f" * 16],
                },
            }
        ],
        right_records=[],
        output_path=tmp_path / "leakage.json",
        minimum_coverage=dict.fromkeys(CATEGORIES, 0.0),
    )

    coverage = report.coverage_by_category
    assert coverage["canonical_url_sha256"].left.with_evidence == 1
    assert coverage["scheme_agnostic_url_sha256"].left.with_evidence == 1
    assert coverage["video_keyframe_sequence"].left.with_evidence == 1


def test_public_hotspots_are_coordinators_not_monoliths() -> None:
    limits = {
        "evaluator/leakage/report.py": 360,
        "orchestration/composition/runtime/fetch.py": 500,
        "crawler/processing/page_discovery_admission.py": 360,
        "crawler/analysis/enrichment/video/video_analyzer.py": 300,
    }
    for filename, maximum in limits.items():
        assert len(Path(filename).read_text().splitlines()) <= maximum

    function_limits = {
        ("datachecker/workflow_decision.py", "decide_workflow_action"): 80,
        ("training/runtime/loop/runner.py", "run_training_loop"): 190,
    }
    for (filename, function_name), maximum in function_limits.items():
        tree = ast.parse(Path(filename).read_text())
        node = next(
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.FunctionDef)
            and candidate.name == function_name
        )
        assert node.end_lineno is not None
        assert node.end_lineno - node.lineno + 1 <= maximum


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_ROOTS = (
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


def _production_paths() -> list[Path]:
    paths: list[Path] = []
    for root_name in _PRODUCTION_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            if "__pycache__" not in path.parts:
                paths.append(path)
    return paths


def _defined_functions(path: Path, name: str) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def _imported_from(module_path: Path) -> dict[str, list[str]]:
    tree = ast.parse(
        module_path.read_text(encoding="utf-8"),
        filename=str(module_path),
    )
    imported: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            imported.setdefault(node.module, []).extend(
                alias.name for alias in node.names
            )
    return imported


def test_loss_evaluation_functions_defined_exactly_once() -> None:
    for name in ("evaluate_final_losses", "evaluate_loader_loss"):
        definitions = {
            path.relative_to(PROJECT_ROOT)
            for path in _production_paths()
            if _defined_functions(path, name)
        }
        assert definitions == {Path("evaluator/loss.py")}, (
            f"{name!r} must be owned solely by evaluator/loss.py"
        )


def test_old_loss_evaluation_module_is_removed() -> None:
    assert not (
        PROJECT_ROOT / "training/validation/loss_evaluation.py"
    ).exists(), "training/validation/loss_evaluation.py must be removed"
    assert not (PROJECT_ROOT / "training/validation").exists(), (
        "training/validation/ must be removed once empty"
    )


def test_task_metrics_functions_defined_exactly_once() -> None:
    for name, expected_path in (
        ("evaluate_task_metrics", Path("evaluator/task_metrics.py")),
        ("summarize_task_metrics", Path("evaluator/aggregation.py")),
    ):
        definitions = {
            path.relative_to(PROJECT_ROOT)
            for path in _production_paths()
            if _defined_functions(path, name)
        }
        assert definitions == {expected_path}, (
            f"{name!r} must be owned solely by {expected_path}"
        )


def test_no_production_module_imports_loss_owner_from_training() -> None:
    violations: list[str] = []
    for path in _production_paths():
        for imported_module, names in _imported_from(path).items():
            if (
                imported_module == "training.validation.loss_evaluation"
                and any(
                    name in ("evaluate_final_losses", "evaluate_loader_loss")
                    for name in names
                )
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)} imports {names} "
                    f"from training.validation.loss_evaluation"
                )
    assert not violations, (
        "Loss owner imports must point to evaluator.loss:\n"
        + ("\n".join(sorted(violations)))
    )


def test_no_production_module_imports_task_metrics_from_training() -> None:
    violations: list[str] = []
    for path in _production_paths():
        for imported_module, names in _imported_from(path).items():
            if (
                imported_module == "training.validation.loss_evaluation"
                and any(
                    name in ("evaluate_task_metrics", "summarize_task_metrics")
                    for name in names
                )
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)} imports {names} "
                    f"from training.validation.loss_evaluation"
                )
    assert not violations, (
        "Task metric imports must point to evaluator.task_metrics:\n"
        + ("\n".join(sorted(violations)))
    )


def test_trainer_imports_task_metrics_from_evaluator() -> None:
    imported = _imported_from(PROJECT_ROOT / "training/runtime/trainer.py")
    assert "evaluate_task_metrics" in imported.get(
        "evaluator.task_metrics", []
    ), "training/runtime/trainer.py must import evaluate_task_metrics"
    assert "summarize_task_metrics" in imported.get(
        "evaluator.task_metrics", []
    ), "training/runtime/trainer.py must import summarize_task_metrics"


def test_trainer_imports_final_losses_from_evaluator() -> None:
    imported = _imported_from(PROJECT_ROOT / "training/runtime/trainer.py")
    assert "evaluate_final_losses" in imported.get("evaluator.loss", []), (
        "training/runtime/trainer.py must import evaluate_final_losses "
        "from evaluator.loss"
    )


def test_evaluator_loss_does_not_import_from_downstream_domains() -> None:
    imported = _imported_from(PROJECT_ROOT / "evaluator/loss.py")
    forbidden_roots = ("training", "orchestration", "config.releases")
    violations = [
        module
        for module in imported
        if any(
            module == root or module.startswith(f"{root}.")
            for root in forbidden_roots
        )
    ]
    assert not violations, (
        "evaluator/loss.py must not depend on downstream domains: "
        f"{sorted(violations)}"
    )


def test_evaluator_task_metrics_does_not_import_from_downstream_domains() -> (
    None
):
    imported = _imported_from(PROJECT_ROOT / "evaluator/task_metrics.py")
    forbidden_roots = ("training", "orchestration", "config.releases")
    violations = [
        module
        for module in imported
        if any(
            module == root or module.startswith(f"{root}.")
            for root in forbidden_roots
        )
    ]
    assert not violations, (
        "evaluator/task_metrics.py must not depend on downstream domains: "
        f"{sorted(violations)}"
    )
