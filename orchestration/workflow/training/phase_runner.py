"""Training phase entrypoint — thin coordinator for campaign execution."""

from __future__ import annotations

from config.releases.release_requirements import ReleaseRequirements
from datachecker.workflow_decision import WorkflowExecutionPlan
from orchestration.workflow.phase import PhaseOutcome
from orchestration.workflow.training.campaign_runner import (
    CampaignInputs,
    TrainingCampaignRunner,
)


class TrainPhaseRunner:
    """Coordinate train -> evaluate -> accept -> persist.

    Thin entrypoint that resolves inputs and delegates to
    TrainingCampaignRunner. Release requirements are fixed at composition
    time; they are invariant for one constructed workflow.
    """

    def __init__(
        self,
        *,
        campaign_runner: TrainingCampaignRunner,
        release_requirements: ReleaseRequirements | None,
    ) -> None:
        self._campaign_runner = campaign_runner
        self._release_requirements = release_requirements

    async def run(self, plan: WorkflowExecutionPlan) -> PhaseOutcome:
        training_root = plan.training_root
        if training_root is None:
            raise ValueError("workflow plan is missing training_root")

        snapshot_id = _require_training_snapshot_id(plan)
        dataset_manifest_hash = _require_dataset_manifest_hash(plan)

        release_requirements = self._release_requirements

        policy = (
            release_requirements.reproducibility
            if release_requirements is not None
            else None
        )
        if policy is not None and not policy.seeds:
            raise ValueError(
                "reproducibility policy must declare at least one seed"
            )

        deterministic_override = (
            True
            if (policy is not None and policy.require_deterministic_execution)
            else None
        )

        inputs = CampaignInputs(
            snapshot_id=snapshot_id,
            dataset_manifest_hash=dataset_manifest_hash,
            training_root=training_root,
            seeds=(tuple(policy.seeds) if policy is not None else (None,)),
            policy=policy,
            release_requirements=release_requirements,
            deterministic_override=deterministic_override,
        )

        return await self._campaign_runner.run(inputs)


def _require_training_snapshot_id(plan: WorkflowExecutionPlan) -> str:
    snapshot_id = plan.training_snapshot_id
    if snapshot_id is None or not str(snapshot_id).strip():
        raise ValueError("workflow plan is missing training_snapshot_id")
    return str(snapshot_id).strip()


def _require_dataset_manifest_hash(plan: WorkflowExecutionPlan) -> str:
    value = plan.dataset_manifest_hash
    if value is None or not value.strip():
        raise ValueError(
            "workflow plan is missing selected_dataset_manifest_hash"
        )
    return value.strip()
