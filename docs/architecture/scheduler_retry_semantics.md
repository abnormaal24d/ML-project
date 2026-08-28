# Scheduler and Retry Semantics

Status: Normative
Source of truth for: the five scheduler outcomes, the retry budget, timed
deferrals, and host-level behavior. This document defines the *contract*; the
scheduler implementation is split across `crawler/scheduling/` and the worker
retry handler.

Related:
- [crawl_terminal_outcomes.md](crawl_terminal_outcomes.md) — run-level outcomes and operator actions
- [config/collection/discovery.py](../../config/collection/discovery.py) — `SchedulingSettings`
- [config/settings/crawler.py](../../config/settings/crawler.py) — checkpoint and dead-letter settings
- [scheduler_retry_budget.py](../../crawler/scheduling/completion/scheduler_retry_budget.py) — `SchedulerRetryBudget`, `TaskRetryState`
- [delayed_task_queue.py](../../crawler/scheduling/queueing/delayed_task_queue.py) — `DelayedTaskQueue`, `DelayedTaskEntry`
- [worker_task_retry_handler.py](../../crawler/worker/task_iteration/worker_task_retry_handler.py) — deferral/timeout decisions from the worker side
- [processor_failure_handler.py](../../crawler/processing/processors/processor_failure_handler.py) — `host_not_ready` deferrals
- [robots_request_gate.py](../../crawler/governance/robots/robots_request_gate.py) — robots-driven deferral waits

## The five scheduler outcomes

Every scheduler outcome for a task is exactly one of these. The terms are
normative and must not be conflated.

| Term                 | Definition                                                                                       | Code mapping                                        |
| -------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Fetch attempt        | A real network fetch of the task URL is performed. Only these consume the attempt budget.        | `TaskRetryState.attempts += 1`                      |
| Timed defer          | The task is parked in the delayed queue until `ready_at`; it does not count as a fetch attempt.  | `DelayedTaskQueue.push(wait_seconds=...)`, `outcome="deferred"` |
| Immediate reschedule | The task returns to the ready queue without delay and without consuming budget.                  | `SchedulerRetryBudget.evaluate()` returning `terminal=False` without incrementing counters |
| Permanent rejection  | The task will never be fetched; dropped from scheduling and not retried in this run.             | `TaskRetryDecision(terminal=True)` (e.g. `reason="max_deferrals_exceeded"`, governance block) |
| Retry exhaustion     | The task consumed its entire retry budget and is abandoned for this run.                         | `TaskRetryDecision(terminal=True, reason="max_total_attempts_exceeded" \| "max_timeouts_exceeded" \| "max_deferrals_exceeded")` |

### Retry time budget

Normative: every task has a retry time budget defined by its scheduling settings.
The budget is measured in attempts, deferrals, and timeouts — not in wall clock
time. A task is not retried indefinitely: once any configured limit is reached,
the task is abandoned with a terminal decision.

| Setting                            | Default | Meaning                                              |
| ---------------------------------- | ------: | ---------------------------------------------------- |
| `max_total_attempts`               | `4`     | Maximum real fetch attempts per task (overridable per kind via `max_total_attempts_by_kind`) |
| `max_deferrals`                    | `3`     | Maximum timed deferrals that count toward the budget |
| `max_timeouts`                     | `1`     | Maximum timeouts per task                            |
| `feed_max_total_attempts`          | `3`     | Default attempts for `feed` tasks                    |
| `feed_max_deferrals`               | `2`     | Default deferrals for `feed` tasks                   |
| `feed_max_timeouts`                | `1`     | Default timeouts for `feed` tasks                    |
| `timeout_retry_wait_seconds`       | `5.0`   | Delay before requeueing a timeout retry              |
| `dead_letter_on_drain`             | `true`  | Abandon deferred feed tasks when the queue drains    |

Deferral that counts toward the budget is controlled by
`_counts_toward_retry_budget()` in `scheduler_retry_budget.py`: a `deferred`
outcome counts only when its reason is in the retry-budget reason set
(`retryable_fetch_error`, `transient_lock_race`, `processor_timeout`,
`fetch_timeout`, `body_timeout`, `transport_timeout`, `handler_timeout`) or its
retry class is in the equivalent class set. A `timeout` outcome always counts.

### Why a timed defer does not consume the immediate-deferral counter

A *timed defer* parks the task until `ready_at` (monotonic clock) in the delayed
queue; it never performed a fetch. The deferral counter (`deferrals`) is the
budget that bounds *how many times a task may be pushed back without a real
attempt*. Deferrals caused by reasons outside the retry-budget set
(e.g. `host_not_ready`) are not counted, so politeness-driven waits do not
exhaust a task's retry budget.

## next_eligible_at and retry deadline

Normative (target contract; the scheduler is being refactored toward this):

- Every scheduled task has a `next_eligible_at` monotonic timestamp. Until then
  the scheduler must not offer the task to a worker.
- A task whose `next_eligible_at` lies in the future is parked in the delayed
  queue (`DelayedTaskQueue`), ordered by `(ready_at, sequence)`.
- A **retry deadline** is the latest `next_eligible_at` at which the task may
  still be retried; beyond it the task is abandoned as retry-exhausted. The
  retry deadline is derived from the retry budget, not configured independently.

The current implementation computes `ready_at` from the configured wait seconds
at push time; the deadline derivation is part of the scheduled-refactor work.

## Immediate reschedule

A task may be returned to the ready host queue immediately when the failure is
not budget-consuming and no cooldown applies (for example a transient lock race
that resolves instantly). Immediate reschedules do not increment `attempts`,
`deferrals`, or `timeouts`, and do not create a delayed entry.

## Permanent rejection

A task is permanently rejected (for this run) when:

- The retry budget is exhausted (`max_total_attempts_exceeded`,
  `max_deferrals_exceeded`, `max_timeouts_exceeded`), or
- A governance gate blocks the URL permanently (robots `disallow`, unsafe
  redirect, or admission-rule denial) — see
  [source_governance.md](../source_governance.md), or
- The scheduler abandons deferred feed tasks on drain when
  `dead_letter_on_drain` is enabled (`reason="drain_mode_retry_task_abandoned"`).

Permanently rejected tasks contribute to the run-level root-seed counters
(`root_seeds_transient_failed`, `root_seeds_governance_blocked`) and therefore
influence the terminal outcome — see
[crawl_terminal_outcomes.md](crawl_terminal_outcomes.md).

### Dead-letter persistence

The completion handler persists recoverable terminal work after it has released
the scheduler condition lock. Until the append has been flushed and synced, the
task remains tracked as a pending dead letter, so scheduler drain/join cannot
finish early and checkpoint export can still recover it. A failed or cancelled
append returns the task to the frontier instead of dropping it. The writer is
constructed before the scheduler; the restart reader is a separate component
and receives the scheduler task deserializer only after scheduler construction.
This separation is normative: the write path must never depend on checkpoint
deserialization.

Dead-letter status describes the terminal disposition, while
`original_outcome` preserves the worker or processor outcome:

| Terminal completion | Dead-letter status |
| ------------------- | ------------------ |
| `deferred` or `timeout` with a terminal retry decision | `retry_exhausted` |
| `failure` or `failed` | `failed` |
| non-requeued `cancelled` or `interrupted` | `cancelled` |

Nonterminal retries, successful work, and intentional `dropped` outcomes do
not produce records. `crawler.state.dead_letter_statuses` is an allowlist over
the three canonical statuses above. A restart reads a bounded batch. When
clearing is enabled, it first persists an uncapped scheduler checkpoint and
only then removes records represented by that checkpoint (newly accepted or
already-scheduled duplicates). A failed checkpoint leaves the JSONL records
untouched.

## Robots, rate-limit and circuit-breaker outcomes

| Situation                              | Scheduler behavior                                                          |
| -------------------------------------- | --------------------------------------------------------------------------- |
| robots allows (`AUTHORITATIVE_ALLOW`)  | Fetch immediately; no deferral                                               |
| robots denies (`AUTHORITATIVE_DENY`)   | Permanent governance block; no retry                                         |
| robots 404/410 (no robots file)        | Treat as allow; fetch                                                       |
| robots 429 with `Retry-After`          | Timed defer honoring `retry_after_seconds`, capped by `max_retry_after_seconds` |
| robots 5xx / DNS / timeout             | Timed defer (fail-closed); transient error counts as budget-consuming when classified retryable |
| robots fetch error (transient)         | Timed defer; never treated as "no content"                                  |
| HTTP 429 from source host              | Timed defer honoring `Retry-After` (pacing `honor_retry_after=true`, cap `max_retry_after_seconds`) |
| HTTP 5xx                              | Timed defer (fail-closed)                                                    |
| DNS error                              | Timed defer (fail-closed)                                                    |
| Unsafe redirect                        | Permanent security block; no retry                                           |
| Retry deadline exceeded                | Retry exhausted; task abandoned                                              |

Robots decisions are produced by `robots_checker.py` / `robots_error_resolver.py`
as `RobotsCheckResult` with `retry_after_seconds`, `host_penalty`, and
`suggested_discovery_factor`. The deferred wait is passed to the request gate
(`robots_request_gate.py`) and translated into `outcome="deferred"` with
`wait_seconds` by the worker retry handler.

## Host-level cooldown and pacing

| Mechanism             | Default        | Meaning                                                       |
| --------------------- | -------------- | ------------------------------------------------------------- |
| Per-host failure cooldown | `30.0` s   | After a host failure, new tasks for the host are held until the cooldown elapses (`http_rules.py`) |
| `max_pending_per_host` | `8`           | Ready-queue cap per host (reduced to `2` under pressure, `1` critical) |
| `max_inflight_per_host`| `1`           | Concurrent in-flight fetches per host                          |
| Host not ready        | —              | Deferral with `reason="host_not_ready"`; does not consume the retry budget (`processor_failure_handler.py`) |

Host-level penalties apply to the host, not the task; a task that is deferred
because its host is cooling down keeps its full task-level retry budget.
