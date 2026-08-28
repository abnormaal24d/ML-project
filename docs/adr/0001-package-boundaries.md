# 0001. Package Boundaries and Composition Roots

## Status

Accepted.

## Context

The project has multiple domain packages that can be reused independently of
the command-line workflow. Refactors must keep those domains testable without
bootstrapping the full application graph.

Without a frozen, explicit architecture it is easy for dependency directions
to erode. This ADR serves as the single source of truth for package boundaries.
All future changes must be evaluated against it.

## Decision

`orchestration` is the application composition layer. It may import domain
packages and wire concrete implementations, but domain packages must not import
`orchestration`.

`crawler` owns crawl-task models, governance decisions, fetching, extraction,
processing, scheduling, and dataset-write coordination for crawl outputs. It
must not import `training`.

`preprocessing` owns reusable cleaning, metadata, OCR, media probing, and
transcription helpers. It should expose adapters that crawler handlers can
compose without reaching into orchestration.

## Small modules rules

A limited number of focused, small modules (< ~30 logical LOC) are permitted when
they encapsulate a single narrow responsibility (for example one rules adapter
or value object). When bundling related tiny modules reduces fragmentation,
prefer that. New small modules require explicit architectural review.

`datasets` owns snapshot and manifest reading/writing primitives. It must not
depend on orchestration.

`training` owns model tasks, sample building, training loops, and checkpoint
logic. It must not import crawler runtime services or orchestration wiring.

`datachecker` owns validation workflow decisions and report building. It should
consume explicit inputs rather than application containers.

`config`, `logger`, `augmentation`, and `multimodal` provide shared settings,
logging, augmentation, and model/task abstractions. They should not become
implicit composition roots.

`config` bevat uitsluitend configuratiemodellen (Pydantic of equivalent), parsing, validatie en environment-loading. Het
mag **geen** domeincode importeren en bevat geen businesslogica of runtime-beslissingen.

Concrete dependencies such as filesystem writers, clocks, network/session
providers, probes, and scheduler state stores are created in composition roots
and injected into domain services.

### Definitie: Composition Root

Een **composition root** is een module (of kleine groep modules) die **uitsluitend** verantwoordelijk is voor:

- het lezen van configuratie;
- het aanmaken van concrete implementaties van interfaces;
- het samenstellen van objectgrafen (wiring / dependency graph);
- het teruggeven van volledig geconfigureerde services (vaak via een dataclass of builder-return).

Een composition root bevat **geen** businesslogica, validaties, runtime-beslissingen, loops, conditionals op
domeinniveau of domeinspecifieke berekeningen.

Voorbeelden in dit project: `orchestration/composition/*`, `orchestration/bootstrap/*`, en delen van
`orchestration/workflow`.

Alle creatie van infrastructuur (aiohttp sessions, clocks, writers, adapters voor cv2/PIL etc.) gebeurt uitsluitend in
composition roots. Domeincode ontvangt alleen interfaces via constructor injection.

## Architecture

The following diagram and table **are the source of truth**. All code, PRs and tooling must be consistent with them.

```
                     orchestration
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    crawler          preprocessing        training
        │                  │                  │
        ├──────────────┐   │                  │
        │              │   │                  │
    datasets      datachecker             augmentation
        │
      config
```

### Core Package Boundaries (authoritative for reviews)

| Package       | Mag importeren                  | Mag niet importeren            |
|---------------|---------------------------------|--------------------------------|
| orchestration | alles                           | niets                          |
| crawler       | preprocessing, datasets, config | orchestration, training        |
| preprocessing | datasets, config                | crawler runtime, orchestration |
| training      | datasets, config                | crawler runtime                |
| datasets      | config                          | orchestration                  |

**Extended rules (full picture):**

| Package      | Mag importeren                  | Mag niet importeren                    |
|--------------|---------------------------------|----------------------------------------|
| datachecker  | datasets, config                | orchestration, crawler runtime         |
| augmentation | datasets, config, preprocessing | orchestration, crawler runtime         |
| config       | (zie definitie hieronder)       | orchestration, crawler, training, etc. |

### Sub-package boundaries (enforced)

- `crawler.runtime` mag niet importeren: `training`, `orchestration`, `crawler.*datasets`, `datasets`
- `crawler.storage.datasets` (crawler.datasets) mag niet importeren: `crawler.runtime`, `orchestration`, `training`
- `preprocessing.media` mag niet importeren: `crawler.runtime`, `orchestration`
- `datasets` mag niet importeren: `orchestration`, `crawler.runtime`

These rules are normative package-boundary guidance and must be enforced during
architecture review and dependency analysis.

### Fijnmazige laag-boundaries (package + layer niveau — review & tooling)

Naast package-niveau gelden de volgende preciezere regels (niet allemaal machine-afdwingbaar, maar wel bindend voor
reviews en toekomstige uitbreiding van `IMPORT_BOUNDARY_RULES`):

- `crawler.processing` → mag `preprocessing.adapters` en `preprocessing.interfaces` importeren
- `crawler.processing` → mag **niet** `preprocessing.runtime` of concrete preprocessing implementaties importeren
- `crawler.runtime` → mag **niet** `preprocessing.runtime` importeren
- `crawler.fetching` → mag `preprocessing.interfaces` importeren (niet de volledige preprocessing)
- `crawler.*` (algemeen) → mag geen directe concrete adapters uit preprocessing/augmentation importeren; alleen via
  interfaces of composition root

**Voorbeeld van te grove regel (te vermijden):**

- "crawler mag preprocessing importeren" ← te breed

**Betere specificatie:**

```
crawler.processing
    -> preprocessing.adapters
    X  preprocessing.runtime

crawler.fetching
    -> preprocessing.interfaces

crawler.runtime
    X  preprocessing.runtime
```

De tooling (`IMPORT_BOUNDARY_RULES`) wordt waar mogelijk uitgebreid met deze fijnmazige regels. Bij twijfel geldt de
laag-specifieke regel boven de package-regel.

## Architectural Invariants

Dit zijn de harde, review-bare regels die altijd gelden (onafhankelijk van fase). Iedere PR en review kan deze direct
afvinken:

- **Domeincode maakt nooit infrastructuur aan.**  
  Geen `aiohttp.ClientSession()`, `datetime.now()`, `Path().open()`, `cv2`, `PIL.Image.open()` etc. in `crawler/`,
  `preprocessing/`, `datasets/`, `training/`, `datachecker/` buiten expliciete adapter modules.

- **Iedere externe dependency heeft precies één composition root.**  
  Alle creatie en wiring van een bepaald type (klok, FS, HTTP, decoders) gebeurt op één centrale plek in
  `orchestration/composition/**`.

- **Iedere dependency wordt via constructorinjectie doorgegeven.**  
  Geen module-level globals, singletons, `from x import y` voor stateful services, of lazy imports in domeincode voor
  infrastructuur.

- **Domeinservices kennen geen globale state.**  
  Geen verborgen caches, module-level retry state, of gedeelde sessions die niet expliciet geïnjecteerd zijn.

- **Infrastructurele adapters mogen concrete libraries gebruiken.**  
  `HttpClientSessionProvider`, `FetchRetryExecutor`, `ResponseBodyReader`, OpenCV-adapters etc. mogen `aiohttp`, `cv2`,
  `av` etc. gebruiken — mits ze in de juiste laag zitten en hun publieke API alleen interfaces exposeert.

- **Interfaces worden gedefinieerd door de consumer, niet door de provider.**  
  De consumer (bijv. `crawler/fetching/fetcher.py`) bepaalt welk schema (`RetryExecutor`) hij nodig heeft. De
  provider (adapter) implementeert dat schema.

- **Composition roots bevatten geen business logic.**  
  Zie de formele definitie hierboven.

- **Config is een leaf en importeert geen domein.**  
  Zie definitie onder Package Boundaries.

Schendingen van deze invariants zijn altijd blokkerend voor merge, onafhankelijk van "het werkt".

## Consequences

- Every future PR can be quickly assessed against the diagram and table.
- Import-boundary checks (Fase 2) make the rules automatically enforced.
- Refactors that move files (Fase 5+) must preserve or explicitly update these boundaries.
- Composition roots (`orchestration/bootstrap`, `orchestration/composition`, `orchestration/workflow`) contain **only**
  wiring, construction, injection and return of objects (zie formele definitie hierboven). No business logic, loops,
  decision trees or validators.
- `__init__.py` files contain only re-exports (`from .x import Y; __all__ = [...]`). No executable logic.

The refactor is deliberately organized in **8 phases** around architectural risk, not "file cleanup":

1. Fase 0 — Architectuur bevriezen (this ADR)
2. Fase 1 — Guardrails herstellen (tooling/metrics reliability)
3. Fase 2 — Import boundaries afdwingen (automated checks)
4. Fase 3 — Composition roots opschonen
5. Fase 4 — `__init__.py` normaliseren
6. Fase 5 — Micro-modules consolideren (per bounded context)
7. Fase 6 — Naamgeving uniformiseren (where it hurts readability)
8. Fase 7 — Grote classes splitsen (SRP for the big orchestrators)
9. Fase 8 — Laatste optimalisaties + metrics targets
10. Fase 9 — Dependency Injection volledig afronden (zie `docs/architecture/dependency_inversion.md` en `docs/roadmaps/dependency_injection_phase_9.md`)

See the full phased plan and PR ordering in project docs or planning notes. The order ensures architecture is frozen and
enforced *before* large structural or behavioral changes.

See also ADR-0002 (scratch-only MultimodalModel + external models strictly limited to preprocessing) for the definitive
boundary on model training vs. derived preprocessing data.

When in doubt, the table and diagram above are authoritative.

## Target Metrics (Fase 8)

> **Fase 9** richt zich op de laatste grote architecturale zuivering: volledige Dependency Inversion. Zie
`docs/architecture/dependency_inversion.md` voor de normatieve regels en `docs/roadmaps/dependency_injection_phase_9.md`
> voor het gedetailleerde plan (huidige situatie, stappen, acceptatiecriteria, risico's, teststrategie en PR-opdeling).

| Metric                            | Huidig (approx) | Streefwaarde | Waarom                                          |
|-----------------------------------|----------------:|-------------:|-------------------------------------------------|
| Modules                           |            1218 |       < 1050 | Minder versnippering en eenvoudiger navigatie   |
| Tiny modules (<30 LOC, non-init)  |             163 |         < 75 | Minder import-overhead en duidelijkere domeinen |
| Vage namen                        |              52 |         < 10 | Betere leesbaarheid en stacktraces              |
| `__init__.py` met logica          |               0 |            0 | Geen verborgen uitvoering bij imports           |
| Boundary violations               |               0 |            0 | Architectuur blijft afdwingbaar                 |
| Grote orchestrators (>300 regels) |               5 |            0 | Hogere SRP en testbaarheid                      |
