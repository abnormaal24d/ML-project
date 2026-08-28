"""Strict recovery reconciliation for interrupted, schema-valid crawl states."""

from __future__ import annotations

from pathlib import Path

from config.settings.root import Settings
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus
from logger.project_logger import ProjectLogger
from orchestration.composition.runtime.workflow_manifest_writers import (
    WorkflowManifestWriters,
)
from orchestration.workflow.phase import RunBlocking

RECOVERABLE_STATUSES = frozenset(
    {
        WorkflowLifecycleStatus.RECOVERING,
        WorkflowLifecycleStatus.RUNNING,
    }
)


async def reconcile_crawl_state(
    *,
    settings: Settings,
    logger: ProjectLogger,
    manifest_writers: WorkflowManifestWriters,
    run_blocking: RunBlocking,
    io_timeout_seconds: float,
) -> bool:
    """Reconcile a recoverable interrupted crawl into canonical form.

    Returns True if a promotion happened, False otherwise.
    Never promotes unless all strict conditions are met.
    """
    timeout = io_timeout_seconds

    # 1. Read current state. The workflow lock is already held by the caller,
    # so a persisted RUNNING state cannot belong to another live writer: it is
    # crash/interruption evidence from the process being resumed.
    state = await run_blocking(
        manifest_writers.crawl_state_manifest_writer.read_current_state,
        timeout_seconds=timeout,
    )
    if state is None:
        return False

    status = state.status
    if status not in RECOVERABLE_STATUSES:
        return False

    # 2. Durably claim stale RUNNING state for recovery before inspecting or
    # replacing its artifacts. This prevents the next crawl attempt from
    # silently overwriting the interrupted attempt.
    if status is WorkflowLifecycleStatus.RUNNING:
        state = await run_blocking(
            manifest_writers.crawl_state_manifest_writer.write_crawl_state_recovering,
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
            error_type="interrupted_crawl_detected",
            error_message=(
                "exclusive resume found a crawl attempt still marked running"
            ),
            timeout_seconds=timeout,
        )
        logger.info(
            "crawl_state_reconcile_recovery_started",
            previous_status=status.value,
            attempt_id=state.attempt_id,
        )
        status = state.status

    if (
        not state.attempt_id
        or not state.raw_run_id
        or not state.crawl_session_id
    ):
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="incomplete_crawl_identity",
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )
        return False

    # 3. Compute current expected hashes only after the interrupted attempt is
    # durably in recovery.
    from config.settings.fingerprint_sections import (
        build_settings_payloads,
    )
    from datachecker.fingerprints import (
        SettingsFingerprintCalculator,
        SourceFingerprintCalculator,
    )

    settings_fingerprint_calculator = SettingsFingerprintCalculator()
    source_fingerprint_calculator = SourceFingerprintCalculator(
        settings_fingerprint_calculator=settings_fingerprint_calculator,
    )
    payloads = build_settings_payloads(
        settings=settings,
        checker_settings=settings.collection.datachecker,
    )
    current_source_hash = source_fingerprint_calculator.calculate(
        seed_urls=tuple(settings.sources.active.seed_urls),
        source_profile=settings.sources.active,
    )
    current_crawl_hash = settings_fingerprint_calculator.calculate(
        payload=payloads.crawl,
    )

    # 4. Idempotently resume an already committed canonical manifest.
    manifest_path = manifest_writers.crawl.crawl_manifest_path()
    if manifest_path.exists():
        try:
            existing_manifest = await run_blocking(
                manifest_writers.crawl_promotion.resume_existing,
                state=state,
                timeout_seconds=timeout,
            )
        except Exception as exc:
            logger.info(
                "crawl_state_reconcile_existing_manifest_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            raise
        if existing_manifest is None:
            logger.info(
                "crawl_state_reconcile_skipped_manifest_identity_mismatch"
            )
            await _abandon_interrupted_attempt(
                manifest_writers=manifest_writers,
                run_blocking=run_blocking,
                logger=logger,
                timeout_seconds=timeout,
                status=status,
                reason="canonical_manifest_identity_mismatch",
                raw_run_directory=state.raw_run_directory,
                run_summary_path=state.run_summary_path,
            )
            return False
        logger.info(
            "crawl_state_reconciled_idempotent_existing_manifest",
            manifest=str(manifest_path),
        )
        return True

    # 5. Hash safety: persisted hashes must match the current settings.
    if state.source_registry_hash != current_source_hash:
        logger.info(
            "crawl_state_reconcile_skipped_hash_mismatch",
            field="source_registry_hash",
            state_hash=state.source_registry_hash,
            current_hash=current_source_hash,
        )
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="source_registry_hash_mismatch",
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )
        return False
    if state.crawl_settings_hash != current_crawl_hash:
        logger.info(
            "crawl_state_reconcile_skipped_hash_mismatch",
            field="crawl_settings_hash",
            state_hash=state.crawl_settings_hash,
            current_hash=current_crawl_hash,
        )
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="crawl_settings_hash_mismatch",
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )
        return False

    # 6. Check if we can resolve a complete attempt.
    latest_attempt = await run_blocking(
        manifest_writers.crawl_state_manifest_writer.resolve_latest_crawl_attempt,
        timeout_seconds=timeout,
    )
    if not latest_attempt.has_complete_files_on_disk():
        logger.info(
            "crawl_state_reconcile_skipped_no_complete_attempt",
            status=status.value,
        )
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="missing_complete_attempt_artifacts",
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )
        return False
    if (
        latest_attempt.raw_run_directory is None
        or latest_attempt.run_summary_path is None
    ):
        logger.info(
            "crawl_state_reconcile_skipped_missing_paths",
            status=status.value,
        )
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="missing_attempt_paths",
            raw_run_directory=state.raw_run_directory,
            run_summary_path=state.run_summary_path,
        )
        return False

    finalized_output_exists = await run_blocking(
        manifest_writers.crawl.has_finalized_raw_output,
        raw_run_directory=latest_attempt.raw_run_directory,
        run_summary_path=latest_attempt.run_summary_path,
        attempt_id=state.attempt_id,
        raw_run_id=state.raw_run_id,
        crawl_session_id=state.crawl_session_id,
        timeout_seconds=timeout,
    )
    if not finalized_output_exists:
        await _abandon_interrupted_attempt(
            manifest_writers=manifest_writers,
            run_blocking=run_blocking,
            logger=logger,
            timeout_seconds=timeout,
            status=status,
            reason="missing_finalized_raw_output",
            raw_run_directory=latest_attempt.raw_run_directory,
            run_summary_path=latest_attempt.run_summary_path,
        )
        return False

    # 7. Commit through the same canonical promotion owner used by the
    #    normal crawl phase.
    try:
        await run_blocking(
            manifest_writers.crawl_promotion.commit,
            state=state,
            attempt=latest_attempt,
            timeout_seconds=timeout,
        )
        logger.info(
            "crawl_state_reconciled_to_canonical",
            raw_run_directory=str(latest_attempt.raw_run_directory),
            crawl_manifest_path=str(manifest_path),
            previous_status=status.value,
        )
        return True
    except Exception as exc:  # exception-rules: boundary-wrap-and-raise
        logger.info(
            "crawl_state_reconcile_promotion_failed",
            error_type=type(exc).__name__,
            error_message=str(exc) or None,
            previous_status=status.value,
        )
        raise


async def _abandon_interrupted_attempt(
    *,
    manifest_writers: WorkflowManifestWriters,
    run_blocking: RunBlocking,
    logger: ProjectLogger,
    timeout_seconds: float,
    status: WorkflowLifecycleStatus,
    reason: str,
    raw_run_directory: Path | None,
    run_summary_path: Path | None,
) -> None:
    """Close one irrecoverable interrupted attempt in crawl-state storage.

    The raw summary remains immutable crash evidence. Only its live dataset
    writer owns journal closure and raw-run finalization; recovery records the
    authoritative terminal decision in the crawl-state manifest instead.
    """

    await run_blocking(
        manifest_writers.crawl_state_manifest_writer.write_crawl_state_abandoned,
        reason=reason,
        raw_run_directory=raw_run_directory,
        run_summary_path=run_summary_path,
        timeout_seconds=timeout_seconds,
    )
    logger.info(
        "crawl_state_reconcile_abandoned_partial_run",
        status=status.value,
        reason=reason,
    )
