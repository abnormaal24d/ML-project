# ADR-0001: Layered Architecture — Bootstrap, Composition, Workflow

**Status:** Accepted  
**Date:** 2026-08-23

## Context

The `orchestration/` package has grown without explicit layer boundaries. Three distinct concerns are currently conflated under the label "composition root":

1. **Bootstrap** — Process entrypoint, CLI, settings loading, logging, signal handling, application graph construction, startup/shutdown.
2. **Composition** — Dependency construction, wiring concrete adapters, implementation selection based on validated config, returning complete object graphs.
3. **Workflow (Application Orchestration)** — Use-case control flow, state machines, loops, retries, campaign sequencing, cancellation, application-level error handling.

This conflation causes:
- `build_crawler()` (456 LOC) performing seed planning, state path resolution, and resource startup
- `TrainPhaseRunner` (1016 LOC) living in `bootstrap/` while implementing a training campaign state machine
- 7 composition modules >300 LOC, 9 bootstrap modules >300 LOC
- Runtime execution (`PrometheusExporter.start()`) during object construction

## Decision

Define three explicit layers with strict dependency direction:

```
bootstrap
    │
    ▼
composition
    │
    ▼
workflow
 ┌──┴──┐
 ▼     ▼
crawler training ...
```

### Layer Responsibilities

#### Bootstrap (`orchestration/bootstrap/`)
**Sole responsibility:** Process lifecycle boundary.

- CLI argument parsing (`orchestration/cli/`)
- Settings loading & validation (`orchestration/settings_loader.py`)
- Logging initialization (`orchestration/bootstrap/logging.py`)
- Signal handlers & shutdown orchestration (`orchestration/bootstrap/shutdown.py`)
- Workflow file locking (`orchestration/bootstrap/workflow_lock.py`)
- Run context / workflow identity (`orchestration/bootstrap/run_context.py`)
- Application container assembly (`orchestration/bootstrap/container.py`)
- Application execution & shutdown (`orchestration/bootstrap/application.py`)
- Workflow phase executor (`orchestration/bootstrap/workflow_executor.py`)

**Must NOT:**
- Contain domain workflows (crawl, training, preprocessing, augmentation)
- Execute runtime logic (loops, retries, state machines)
- Start runtime resources (`.start()`, `.run()`, `.crawl()`, `.train()`)

#### Composition (`orchestration/composition/`)
**Sole responsibility:** Declarative dependency construction.

- Build concrete adapters from validated settings
- Wire dependency graphs
- Select implementations based on configuration
- Return fully constructed object graphs

**Must NOT:**
- Contain runtime loops (`for`, `while`, `async for`)
- Contain `await` / `async def` (no async execution)
- Call runtime lifecycle methods: `.start()`, `.run()`, `.crawl()`, `.train()`, `.execute()`, `.process()`, `.evaluate()`
- Perform domain calculations (seed expansion, path policy, feed alternates)
- Perform I/O beyond reading config

**Allowed conditionals:** Implementation selection only.
```python
# ✅ Valid composition conditional
exporter = (
    PrometheusExporter(port=settings.port)
    if settings.prometheus_enabled
    else NullMetricsExporter()
)

# ❌ Invalid — starts runtime resource
if settings.prometheus_enabled:
    exporter = PrometheusExporter(...)
    exporter.start()
```

#### Workflow (`orchestration/workflow/`)
**Sole responsibility:** Application use-case orchestration.

- Campaign/phase runners (crawl, training, preprocessing, augmentation)
- State machines & control flow
- Loops, retries, sequencing
- Cancellation & failure transitions
- Application-level error handling

**Must NOT:**
- Import from `orchestration.composition` or `orchestration.bootstrap`
- Construct concrete infrastructure directly (receive via constructor injection)

#### Domain packages (`crawler/`, `training/`, `preprocessing/`, ...)
**Must NOT:**
- Import from `orchestration.*`

### Module Placement Rules

| Current Location | Correct Layer | Action |
|-----------------|---------------|--------|
| `orchestration/bootstrap/train_phase_runner.py` | `orchestration/workflow/training/phase_runner.py` | Move + split |
| `orchestration/bootstrap/crawl_phase_runner.py` | `orchestration/workflow/crawl/phase_runner.py` | Move |
| `orchestration/bootstrap/preprocess_phase_runner.py` | `orchestration/workflow/preprocessing/phase_runner.py` | Move |
| `orchestration/bootstrap/augment_phase_runner.py` | `orchestration/workflow/augmentation/phase_runner.py` | Move |
| `orchestration/composition/runtime/crawler.py` | `orchestration/composition/runtime/crawler.py` | Refactor to subgraphs |
| `orchestration/composition/runtime/training.py` | `orchestration/composition/runtime/training.py` | Remove domain validation |

### Enforcement (CI Gates)

1. **Import boundaries** — `workflow` must not import `composition`/`bootstrap`; domains must not import `orchestration`
2. **AST composition checks** — In `orchestration/composition/**`:
   - No `ast.For`, `ast.AsyncFor`, `ast.While`
   - No `ast.Await`, `ast.AsyncFunctionDef`
   - No calls to `.start()`, `.run()`, `.crawl()`, `.train()`, `.execute()`, `.process()`, `.evaluate()`
3. **LOC budgets** — Composition module ≤300 LOC; builder function ≤120 LOC; Workflow class ≤300 LOC; method ≤100 LOC
4. **Cyclomatic complexity** — Composition builder ≤8

## Consequences

- `TrainPhaseRunner` moves to `orchestration/workflow/training/` and splits into `PhaseRunner`, `CampaignRunner`, `AttemptRunner`, `PostTrainingRunner`, `StageExecutor`
- `build_crawler()` decomposes into `build_crawler_infrastructure()`, `build_crawler_governance()`, `build_crawler_state()`, `build_crawler_execution()`, `assemble_crawler_graph()`
- Seed expansion moves to `crawler/scheduling/seed_plan.py`
- `PrometheusExporter.start()` moves to application lifecycle (`container.aclose()` symmetry)
- `_resolve_state_directory()` moves to `crawler/runtime/state/state_path_resolver.py`
- Architecture tests become merge-blocking

## References

- `orchestration/bootstrap/workflow_executor.py` — Generic phase executor (stays in bootstrap)
- `orchestration/workflow/curated_snapshot_runtime.py` — Already correctly placed workflow example
- `crawler/scheduling/seed_plan.py` — New seed planning domain component