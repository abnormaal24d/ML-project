# Crawl Terminal Outcomes

Status: Normative
Source of truth for: how a crawl run ends, what evidence each outcome requires,
whether retry is automatic, and what an operator must do.

Related:
- [crawl_run_summary.py](../../crawler/runtime/loop/crawl_run_summary.py) — `CrawlStopTrigger`, `CrawlTerminalOutcome`, `CrawlRunResult`
- [crawl_run_supervisor.py](../../crawler/runtime/loop/crawl_run_supervisor.py) — `_classify_terminal_outcome()`
- [application.py](../../orchestration/bootstrap/application.py) — dataset outcome mapping
- [scheduler_retry_semantics.md](scheduler_retry_semantics.md) — retry policy at task level

## Definitions

A crawl run ends with exactly one **stop trigger** (the mechanical reason the run
stopped) and one **terminal outcome** (the functional result for the dataset).

> `frontier_exhausted` (in code: `frontier_drained`) is a mechanical stop trigger, not a functional end result.

### Stop triggers (`CrawlStopTrigger`)

| Trigger                      | Meaning                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `frontier_drained`           | Scheduler queue, delayed queue, and in-flight work all empty   |
| `stop_requested`             | Operator requested stop via control file                       |
| `cancelled`                  | `asyncio.CancelledError` during the run                        |
| `interrupted`                | Keyboard interrupt (`Ctrl+C`) during the run                   |
| `delayed_backlog_deferred`   | Run ended while delayed backlog exceeded the idle-drain budget |
| `no_accepted_seeds`          | No seeds accepted and no restored work                         |
| `failed`                     | Unrecoverable runtime error                                     |

### Terminal outcomes

| Outcome                               | Definition                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `completed_ready`                     | Frontier drained, readiness gate passed: output is complete and eligible for the next phase         |
| `completed_below_readiness_threshold` | Frontier drained, but one or more readiness thresholds were not met (records, requests, quality, modality) |
| `completed_no_eligible_content`       | Frontier drained and ready, but no eligible records were produced; nothing worth promoting          |
| `completed_governance_blocked`        | Frontier drained and ready, but all root seeds were blocked by governance (robots, source rules)    |
| `incomplete_transient_infrastructure` | Run did not reach the frontier; transient infrastructure failures (DNS, network, robots fetch, 5xx) |
| `incomplete_dependency_failure`       | Run did not reach the frontier; a required dependency (config, native library, backend) failed      |

## Outcome evidence and behavior

### completed_ready

- **Definition**: The frontier was exhausted and the output-readiness gate passed.
- **Required evidence fields**: `stop_trigger=frontier_drained`, `terminal_outcome=success`, `unmet_requirements=[]`, `requests_total`, `successful_requests_total`, `object_records_total`, `quality_score`.
- **Exit / workflow behavior**: Dataset run is committed as `completed` with `final=true`. Exit code `0`.
- **Automatic retry**: No.
- **Operator action**: Start the next phase (curation, training).
- **Example log**:
  ```
  crawler_frontier_drained
  crawler_finished stop_trigger=frontier_drained
  application_runtime_completed exit_code=0
  ```

### completed_below_readiness_threshold

- **Definition**: The frontier was exhausted but the readiness gate reported unmet requirements.
- **Required evidence fields**: `stop_trigger=frontier_drained`, `terminal_outcome=incomplete`, `unmet_requirements=[...]` (e.g. `min_raw_objects_total`, `min_successful_requests_total`, `min_quality_score`), `object_records_total`, `requests_total`, `successful_requests_total`, `quality_score`, `modality_counts`.
- **Exit / workflow behavior**: Dataset run is marked `incomplete`, not `completed`. Exit code `1` (outcome is not `COMPLETED`).
- **Automatic retry**: No — thresholds were evaluated at true frontier exhaustion; retrying the same sources is not automatic.
- **Operator action**: Review thresholds and sources; adjust `crawl_output_gate` thresholds or source profiles, then re-run.
- **Example log**:
  ```
  crawler_training_data_not_ready stop_trigger=frontier_drained unmet_requirements=["min_raw_objects_total"] object_records_total=12
  ```
- **Difference with stop trigger**: The stop trigger is `frontier_drained`; the outcome is decided by the readiness gate, not by the trigger itself.

### completed_no_eligible_content

- **Definition**: Frontier drained, readiness passed, but zero eligible records were produced (no content passed extraction/admission gates).
- **Required evidence fields**: `object_records_total=0`, `root_seeds_total>=1`, `root_seeds_succeeded`/`root_seeds_transient_failed`/`root_seeds_governance_blocked`, `eligible_records_total=0`.
- **Exit / workflow behavior**: Marked `incomplete`; nothing is promoted.
- **Automatic retry**: No.
- **Operator action**: Investigate extraction and source selection; verify the source profile actually yields eligible content.
- **Example log**:
  ```
  crawler_finished stop_trigger=frontier_drained completed_tasks=0 object_records_total=0
  ```
- **Difference with stop trigger**: Again `frontier_drained` — the distinction is the eligibility evidence, not the trigger.

### completed_governance_blocked

- **Definition**: Frontier drained and readiness passed, but every root seed was blocked by governance (robots disallow, source rules, URL admission).
- **Required evidence fields**: `root_seeds_total>=1`, `root_seeds_governance_blocked==root_seeds_total`, `root_seeds_succeeded=0`.
- **Exit / workflow behavior**: Marked `incomplete`; nothing is promoted.
- **Automatic retry**: No.
- **Operator action**: Review governance and source rules (robots cache invalidation, source registry status, URL admission rules).
- **Example log**:
  ```
  seed_rejected url=... reason=robots_disallowed
  crawler_finished stop_trigger=frontier_drained root_seeds_governance_blocked=3
  ```
- **Difference with stop trigger**: `frontier_drained` again; the outcome is determined by the root-seed governance counters.

### incomplete_transient_infrastructure

- **Definition**: The run ended before the frontier was exhausted because of transient infrastructure failures (DNS, network, robots fetch errors, 5xx backends) that consumed the retry budget.
- **Required evidence fields**: `root_seeds_transient_failed>0`, `retry_exhausted_total>0`, or `unmet_requirements` containing transient dependency failures.
- **Exit / workflow behavior**: Marked `incomplete`.
- **Automatic retry**: **Yes** — this is the only outcome with automatic retry. The run may be re-executed without source or threshold changes.
- **Operator action**: Restore DNS/network/robots access; simply re-run.
- **Example log**:
  ```
  scheduler_task_retry_exhausted outcome=deferred reason=max_deferrals_exceeded
  crawler_finished stop_trigger=frontier_drained root_seeds_transient_failed=2
  ```
- **Difference with stop trigger**: The stop trigger may be `frontier_drained` or `failed`; the transient evidence decides the outcome.

### incomplete_dependency_failure

- **Definition**: The run ended before the frontier because a required dependency failed: missing processing-activity config, missing native dependency, unavailable backend.
- **Required evidence fields**: `required_dependency_failures>0` or startup error (`ProcessingActivityConfigError` → exit code `2`).
- **Exit / workflow behavior**: Marked `incomplete` at dataset level; startup config failures exit `2`.
- **Automatic retry**: No.
- **Operator action**: Fix the deployment or dependency, then re-run.
- **Example log**:
  ```
  Startup configuration error: component=processing_activity_registry issue=processing_activity_config_missing
  ```
- **Difference with stop trigger**: A startup failure never reaches the crawl loop, so there is no `CrawlStopTrigger`; the error is handled by the CLI before the supervisor starts.

## Retry policy matrix

| Outcome                               | Retry | Operator action                         |
| ------------------------------------- | ----: | --------------------------------------- |
| `completed_ready`                     |   No  | Start the next phase                    |
| `completed_below_readiness_threshold` |   No  | Review sources and thresholds           |
| `completed_no_eligible_content`       |   No  | Investigate extraction and source selection |
| `completed_governance_blocked`        |   No  | Review governance and source rules      |
| `incomplete_transient_infrastructure` |   Yes | Restore DNS/network/robots; re-run      |
| `incomplete_dependency_failure`       |   No  | Fix deployment or dependency; re-run    |

## Relationship to code

The classifier `_classify_terminal_outcome()` in `crawl_run_supervisor.py` derives
the four-value `CrawlTerminalOutcome` (`success`, `incomplete`, `cancelled`,
`failed`) from the stop trigger and the readiness gate:

```
stop_trigger == FRONTIER_DRAINED and readiness.ready  -> success
stop_trigger == FRONTIER_DRAINED and not readiness.ready -> incomplete
stop_trigger == CANCELLED                              -> cancelled
stop_trigger == FAILED                                 -> failed
stop_trigger in {STOP_REQUESTED, INTERRUPTED,
                 DELAYED_BACKLOG_DEFERRED, NO_ACCEPTED_SEEDS} -> incomplete
```

The six functional outcomes above refine `incomplete`/`success` using the evidence
fields on `CrawlRunResult`:

| Functional outcome                       | Code derivation                                             |
| ---------------------------------------- | ----------------------------------------------------------- |
| `completed_ready`                        | `success`                                                   |
| `completed_below_readiness_threshold`    | `incomplete` + `unmet_requirements` non-empty               |
| `completed_no_eligible_content`          | `incomplete`/`success` + `eligible_records_total == 0` and `root_seeds_total > 0` |
| `completed_governance_blocked`           | `incomplete` + `root_seeds_governance_blocked == root_seeds_total > 0` |
| `incomplete_transient_infrastructure`    | `incomplete` + `root_seeds_transient_failed > 0` or retry exhaustion |
| `incomplete_dependency_failure`          | `incomplete` + `required_dependency_failures > 0`, or startup error (exit 2) |

Evidence fields are populated from `crawl_output_readiness_report()` (requests,
records, quality, modality) and from worker-pool root-seed counters
(`root_seeds_total/succeeded/transient_failed/governance_blocked`).

## Exit codes

| Exit code | Meaning                                   |
| --------: | ----------------------------------------- |
|      `0`  | Success (`completed_ready`)               |
|      `1`  | Runtime failure or non-completed outcome  |
|      `2`  | Startup/configuration error               |
|    `130`  | Keyboard interrupt                        |
|    `131`  | Cancelled                                 |

Only codes actually implemented in `orchestration/main.py` and
`orchestration/bootstrap/application.py` are listed.
