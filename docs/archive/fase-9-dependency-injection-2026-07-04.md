# FASE 9 — Dependency Injection volledig afronden

**Status**: Archive / Historical planning snapshot (2026-07-04)
**Superseded by**: [../architecture/dependency_inversion.md](../architecture/dependency_inversion.md), [../architecture/infrastructure_boundaries.md](../architecture/infrastructure_boundaries.md), [../roadmaps/dependency_injection_phase_9.md](../roadmaps/dependency_injection_phase_9.md)

**Status**: Archive / Historical planning snapshot (2026-07-04)
**Superseded by**: [../architecture/dependency_inversion.md](../architecture/dependency_inversion.md), [../architecture/infrastructure_boundaries.md](../architecture/infrastructure_boundaries.md), [../roadmaps/dependency_injection_phase_9.md](../roadmaps/dependency_injection_phase_9.md)

**Volgt op**: Fase 8 (Laatste optimalisaties + metrics)  
**Gerelateerd aan**: ADR-0001 Package Boundaries and Composition Roots

**Aanscherpingen toegepast**:

1. Elk hoofdstuk expliciet projectspecifiek (bestandsnamen + packages genoemd).
2. Centrale dependency-grafiek met toegestaan/verboden pijlen toegevoegd.
3. Definition of Done per onderdeel (incl. exacte "Definition of Done — 062").
4. PR-opdeling voor 062 fijner gemaakt (062.1 t/m 062.5).
5. Risicoklasse tabel toegevoegd met ⭐⭐⭐ Hoog etc.
6. Volledige migratiestrategie sectie + "nooit alles in één PR" regel.
7. Onderscheid gemaakt tussen "toegestane infrastructuurimplementaties" (infra-laag) en "ongewenste lekken".
8. Rollback-strategie per (grote) fase toegevoegd.
9. Verwijzingen naar ADR-0001 (composition root definitie + Architectural Invariants + precieze boundaries + config
   nuance).

---

## Doel

Na deze fase bestaat er nog maar **één plaats waar infrastructuur wordt gecreëerd:** de composition roots in
`orchestration/composition/` (zie definitie en invariants in ADR-0001).

Alle domeinservices in `crawler/`, `preprocessing/`, `datasets/`, `datachecker/`, `augmentation/` en `training/` zijn
volledig afhankelijk van **expliciete interfaces** en kennen geen concrete implementaties van infrastructuur meer.

Concreet betekent dit dat:

* geen enkele domeinservice zelf HTTP clients, sessions of responses aanmaakt of in signatures accepteert;
* geen enkele domeinservice zelf `Path(...).open()`, `.mkdir()`, `tempfile` of directe bestandsmutaties uitvoert;
* geen enkele domeinservice `datetime.now()`, `uuid.uuid4()` of vergelijkbare niet-deterministische calls bevat;
* geen enkele domeinservice directe `import cv2`, `import av`, `from PIL import ...`, subprocess ffmpeg etc. bevat
  buiten expliciete adapters;
* geen enkele service verborgen singleton-state, module-level caches of globals voor retry, sessions of dependencies
  gebruikt;
* alle infrastructuur (netwerk, klok, FS, media decoders, ID's) vervangbaar is door testimplementaties (mocks, fakes,
  in-memory).

Hiermee wordt het project volledig conform **Dependency Inversion Principle (DIP)** en **Clean Architecture** zoals
vastgelegd in ADR-0001.

---

## Current Architecture — Huidige dependency-grafiek

Dit is de werkelijke situatie **vóór** de refactor (stand 2026-07-04, gebaseerd op de snapshot en statische analyse).

### Overkoepelende compositie

```
orchestration
        │
        ▼
build_fetch_services()   [orchestration/composition/runtime/build_fetch_services.py]
        │
        ▼
FetchServiceGraph (FetchServices dataclass)

        │
        ├── HttpClientSessionProvider          [crawler/fetching/network/session.py]
        ├── FetchRetryExecutor                 [crawler/fetching/execution/attempt.py]
        ├── FetchProfileExecutor               [crawler/fetching/execution/attempt.py]
        ├── ResponseBodyReader                 [crawler/fetching/network/body/reader.py]
        ├── FetchResponseStatusHandler         [crawler/fetching/response/status_rules.py]
        ├── FetchResponseValidator             [crawler/fetching/response/validator.py]
        ├── HeadPreflightService               [crawler/fetching/network/preflight/executor.py]
        ├── RetryManager                       [crawler/governance/retry/retry_manager.py]
        ├── RateLimiter
        ├── RedirectValidator
        └── ...
```

### Nog toegestane infrastructuurimplementaties

Deze componenten **mogen** concrete infrastructuur gebruiken omdat ze onderdeel zijn van de infrastructuurlaag (binnen
`crawler/fetching/network/` of equivalent). Het probleem zit niet in hun bestaan, maar in waar ze worden gebruikt en van
wie ze afhankelijk zijn.

- `HttpClientSessionProvider` (`crawler/fetching/network/session.py`)
- `ResponseBodyReader` (`crawler/fetching/network/body/reader.py`)
- `FetchRetryExecutor` (`crawler/fetching/execution/attempt.py`)
- `FetchProfileExecutor`, `HeadPreflightService` (mits hun publieke API geen concrete aiohttp types lekt)

Deze horen in de infrastructuurlaag en worden alleen via interfaces geëxposeerd aan hogere lagen.

### Nog ongewenste infrastructuurlekken

Dit zijn de echte schendingen die opgelost moeten worden (zie snapshot en per-sectie DoD):

- `FetchResult` (`crawler/fetching/results/result.py`) → `datetime.now()`
- `HostProfilePreferenceStore` → `datetime.now()`
- Domeinservices (`Fetcher`, `HeadPreflightService`, response assemblers, etc.) → directe `aiohttp.ClientSession` /
  `ClientResponse` in signatures of runtime
- Response assemblers en body readers lekken `ClientResponse` naar callers buiten de network laag
- Storage writers, manifests, snapshots, fetch results → directe `Path().open/mkdir`, `tempfile`, `uuid.uuid4()`
- `crawler/extraction/payloads/video_payload_extractor.py`,
  `preprocessing/privacy/inspection/local_visual_analysis.py`, `augmentation/image/*`,
  `augmentation/document/document_page_augmenter.py` → directe `cv2`, `PIL` (buiten adapters)
- `RetryManager` directe aanroepen vanuit domein fetch logica i.p.v. via `RetryExecutor` interface

Dit diagram toont exact waar een reviewer moet kijken: de directe pijlen van domeincode naar infrastructuur. Zie de
centrale grafiek hierboven voor toegestane vs verboden richtingen.

**Belangrijk voor reviewers**: Niet alles met aiohttp of datetime is een lek. Alleen wanneer het in domeincode of
publieke API's van domeinservices zit, of wanneer de infrastructuurlaag zelf afhangt van hogere lagen.

---

## Centrale Dependency Grafiek & Toegestane Richtingen

**Bron**: `orchestration/composition/` (build_fetch_services.py, build_*.py, crawler_infrastructure_services.py, etc.)

```
orchestration/composition
├── Fetch
│     sessions, retry, response
├── Domain services
│     crawler / preprocessing / datasets
├── Storage
│     clock, ids, atomic write
└── ...
```

**Regels (ADR-0001 + Fase 9)**

- **Toegestaan**: orchestration → crawler
- **Toegestaan**: orchestration → preprocessing
- **Toegestaan**: orchestration → datasets
- **Verboden**: crawler → orchestration
- **Verboden**: training → crawler.runtime
- **Verboden**: preprocessing/media → crawler.runtime, orchestration
- **Verboden**: datasets → orchestration
- **Verboden**: directe domeincode → `aiohttp.ClientSession`, `cv2`, `datetime.now()`, `Path().open/mkdir`, `uuid`, raw
  `tempfile` (behalve in expliciete adapters binnen infra-lagen)

**Belangrijk onderscheid** (zie ook "Nog toegestane..." vs "Nog ongewenste..."):

- Infrastructurele componenten in de juiste laag (HttpClientSessionProvider, FetchRetryExecutor, ResponseBodyReader,
  media adapters) **mogen** concrete libraries gebruiken.
- Lekken naar domeinservices, publieke API's, FetchResult, storage writers buiten adapters etc. zijn ongewenst.

Alleen composition roots in `orchestration/composition/` mogen concrete infrastructuur (aiohttp, cv2, FS-mutaties, echte
klok) kennen en injecteren via interfaces in de domeinlagen. Zie Architectural Invariants in ADR-0001.

---

## Scope van Fase 9 (overall)

**In scope**

✔ Fetch execution, sessions, retry, preflight, response parsing, redirects, payload assembly (in
`crawler/fetching/network/**`, `orchestration/composition/runtime/build_fetch_services.py`)
✔ Storage writers, manifests, snapshots, fetch results (klok + ID + atomic FS) in `crawler/storage/datasets/writing/`,
`mmcrawler_datasets/snapshots/`, `datachecker/artifacts/`, `crawler/fetching/results/`
✔ Media decoders, frame sampling, audio/video transcode, image ops (adapters) in `preprocessing/media/`,
`preprocessing/media/video/`, `preprocessing/media/audio/`, `augmentation/video/`, `augmentation/image/`
✔ Source registry loading (`crawler/governance/discovery/`, `config/`)
✔ Scheduler state / checkpoints / recovery stores (`crawler/scheduling/`, `crawler/worker/`)
✔ Dependency probing + optional backends (`orchestration/composition/runtime/optional_dependency_validator.py`)
✔ Composition root splitsing en builder hygiene (`orchestration/composition/**`)
✔ Import boundary tests + schemachecks + deterministic geïsoleerde validatiechecks

**Out of scope**

✖ Crawl scheduling algoritmes en queue strategie  
✖ Volledige dataset persistence / collation logica (alleen de writers/manifests)  
✖ Robots rules details en governance beslissingen (alleen de preflight abstractie)  
✖ Metrics verzameling implementatie  
✖ Training loop en model training internals  
✖ Volledige end-to-end crawler runs als vervanging voor geïsoleerde validatiechecks

---

## Definition of Done (fase-breed)

Een PR (of de hele fase) is klaar wanneer alle onderstaande items ✅ zijn:

□ Geen directe `aiohttp.ClientSession` / `ClientResponse` buiten
`crawler/fetching/network/session.py` + providers en
`orchestration/composition/runtime/build_fetch_services.py`.  
□ Geen `datetime.now()`, `uuid.uuid4()`, `time.time()` calls meer in domeincode van `crawler/`, `preprocessing/`,
`datasets/`, `training/`, `datachecker/` (vervangen door `Clock`/`IdGenerator` uit composition).  
□ Geen directe `Path(...).open/mkdir`, `tempfile`, `os.open` etc. in domein (behalve binnen `AtomicWriter`/`PathLayout`
adapters in `crawler/storage/datasets/writing/`, `mmcrawler_datasets/snapshots/`, `preprocessing/media/`).  
□ Geen `import cv2`, `import av`, `from PIL`, directe ffmpeg calls buiten `preprocessing/media/adapters/`,
`augmentation/*/adapters/`.  
□ Alle fetch-validatiechecks in `crawler/fetching/` draaien zonder sockets (via mock `HttpClientSessionProvider` +
`FakeRetryExecutor` in `orchestration/composition`).  
□ Alle storage/snapshot/manifest-validatiechecks zijn deterministisch (FixedClock + SequenceIdGenerator).  
□ Media-validatiechecks draaien zonder OpenCV/PyAV/FFmpeg/PIL (via fakes).  
□ Dependency graph in `orchestration/composition/runtime/build_fetch_services.py`, `build_*_services.py` is opgesplitst en
expliciet.  
□ ADR-0001 package boundaries niet geschonden (import-boundary checks groen in
de geautomatiseerde import-boundary check).  
□ Builder size checks en fragmentatie checks groen.  
□ Snapshot `fase-9-current-violations-snapshot.md` is bijgewerkt of expliciet geverifieerd voor de gewijzigde secties.  
□ Migratiestrategie gevolgd (interface → adapter → composition → migratie → checks → opruimen).

---

## Migratiestrategie (fase-breed — verplicht voor elke refactor)

**Belangrijk**: Nooit in één PR zowel interface ontwerpen, implementatie vervangen als businesslogica herschrijven. Dat
is de grootste regressiebron.

Per refactor-onderdeel (062.x, 063.x, ...):

1. **Nieuwe interface introduceren.**  
   Definieer de abstractie (bijv. `RetryExecutor`, `FileSystem`, `Clock`) in een geschikt package (vaak
   `crawler/shared/...` of `preprocessing/media/interfaces/`).

2. **Adapter implementeren.**  
   Bouw de concrete adapter(s) die de oude logica wrappen (bijv. `FetchRetryExecutor` als impl van `RetryExecutor`,
   `SystemClock`).

3. **Composition root aanpassen.**  
   In `orchestration/composition/runtime/build_fetch_services.py` (of `storage/`, `media/` builders) de nieuwe interface + impl
   aanmaken en injecteren.

4. **Oude code via de nieuwe interface laten lopen.**  
   Update callers in domein (crawler/fetching/* etc.) om de interface te gebruiken. Oude directe aanroepen tijdelijk nog
   ondersteund via adapter indien nodig.

5. **Tests migreren.**  
   Alle geïsoleerde validatiechecks draaien nu via fakes/mocks van de interface. Voeg import-boundary tests, schemachecks en snapshot
   checks toe. Update de geautomatiseerde import-boundary check.

6. **Oude constructor verwijderen.**  
   Verwijder oude directe constructors, directe `aiohttp`/`datetime`/ `Path` calls, en opruimen. Pas `build_*` aan.

**Volgorde altijd**: interface → adapter + composition → migratie van calls → checks → opruimen.

### Rollback-strategie (fase-breed)

Voor iedere grote refactor (vooral 062, 065, 066) geldt:

- Migreer alle call-sites binnen dezelfde wijziging; voeg geen parallelle
  constructor- of compatibiliteitsroute toe.
- Verwijder oude implementaties en directe infra-aanroepen zodra:
    - alle call-sites zijn gemigreerd,
    - import-boundary tests + relevante geïsoleerde validatiechecks groen zijn,
    - en minstens één canary / integratietest run succesvol is.
- Bij blokkerende regressie: revert de migratie-PR en houd de oude constructors aan tot een volgende iteratie. Nooit "
  alles of niets".

Dit minimaliseert blast radius bij grote structurele veranderingen.

---

# 062 — Dependency Injection volledig doorvoeren in Fetch Services

## Scope

**In scope**

✔ `HttpClientSessionProvider` + alle sessie creatie en lifecycle  
✔ `FetchRetryExecutor` + retry + backoff + redirect handling binnen fetch  
✔ `ResponseBodyReader`, status handling, payload assembly  
✔ `HeadPreflightService` + robots preflight + head checks  
✔ `FetchProfileExecutor`, `Fetcher`, request context assembly  
✔ `build_fetch_services()` en de FetchServiceGraph  
✔ Vervangen van directe `aiohttp` types + `datetime` in fetch resultaten

**Out of scope**

✖ Volledige robots rules en governance beslissingen (blijven in `crawler/governance/robots/`)  
✖ Rate limiting core (RateLimiter)  
✖ Crawl scheduling en task queuing  
✖ Content classification (wordt al via composition geïnjecteerd)

---

## Huidige situatie

De package `crawler/fetching/network/` bevat reeds de primaire netwerkabstracties:

- `HttpClientSessionProvider` (in `crawler/fetching/network/session.py`) beheert de lazy
  `aiohttp.ClientSession`.
- `FetchRetryExecutor` (in `crawler/fetching/execution/attempt.py`) bevat retry-, redirect- en rate-limit-hint
  logica.
- `FetchProfileExecutor`, `HeadPreflightService`, `FetchResponseStatusHandler`, `ResponseBodyReader` bestaan.
- De composition root bevindt zich in `orchestration/composition/runtime/build_fetch_services.py` (de functie
  `build_fetch_services()` bouwt de `FetchServices` dataclass met 15+ componenten).

De resterende overtredingen bevinden zich voornamelijk in:

- `crawler/fetching/results/result.py`
- `crawler/fetching/network/` (inclusief `body/`, `crawler/fetching/execution/attempt.py`, `crawler/fetching/network/preflight/executor.py`, `crawler/governance/retry/retry_manager.py`)
- `crawler/fetching/response/validator.py`
- `crawler/fetching/fetcher.py` (directe afhankelijkheid van `RetryManager`)
- `crawler/fetching/profiles/host_preferences.py`

### Nog directe infrastructuur (ongewenste lekken)

- `datetime.now(timezone.utc)` in `crawler/fetching/results/result.py:100` (binnen `FetchResult`)
- `datetime.now(timezone.utc)` in `crawler/fetching/profiles/host_preferences.py:148`

**Zie Snapshot**: Sectie 1 — Directe tijd- en ID-generatie.

**Let op**: `FetchRetryExecutor`, `ResponseBodyReader` en `HttpClientSessionProvider` zelf zijn *toegestane*
infrastructuurimplementaties (zie boven). Het lek zit in domeincode die er direct van afhangt of types lekt.

### Nog concrete libraries (deels toegestaan, deels lek)

- Directe `aiohttp` types in signatures en runtime in:
    - `crawler/fetching/execution/attempt.py` (FetchRetryExecutor — toegestaan zolang API schoon is)
    - `crawler/fetching/execution/attempt.py`
    - `crawler/fetching/response/body_reader.py`
    - `crawler/fetching/network/body/reader.py`, ... (ResponseBodyReader — toegestaan als impl, niet in publieke domein
      signatures)
    - `crawler/fetching/network/preflight/executor.py`
    - `crawler/fetching/response/validator.py`, `status_handler.py`
    - `crawler/fetching/fetcher.py` (indirect via executors)
    - `orchestration/composition/runtime/build_fetch_services.py` (verwachte plek voor concrete wiring — **toegestaan**)

**Zie Snapshot**: Sectie 4 — Fetch / HTTP / retry.

**Belangrijk**: De aanwezigheid van aiohttp in `crawler/fetching/network/*` is vaak correct (infra laag). Het probleem
is lekkage naar `crawler/fetching/fetcher.py`, `results/`, `profiles/`, of call-sites in domein.

### Nog verborgen state

- Gedeelde `ClientSession` state in `HttpClientSessionProvider` (moet via provider, maar types lekken nog).
- Caches in `ConditionalRepresentationCache`, host profiles etc. (deels ok, maar niet altijd expliciet geïnjecteerd).
- Retry state en feedback recording zijn deels verspreid.

### Nog gemengde verantwoordelijkheden

- Retry + redirect + rate limit hint extractie + backoff in `FetchRetryExecutor`.
- Response status + body lezen + partial payload + streaming in `ResponseBodyReader` + `FetchedPayloadAssembler`.
- Preflight (robots + head) + retry logic in `HeadPreflightService` + `Fetcher`.

## Mag vs Moet (precies)

De implementatie van retry en exponential backoff **mag** in `FetchRetryExecutor` (
`crawler/fetching/execution/attempt.py`) en `crawler/governance/retry/retry_manager.py` blijven. Dat is hun
verantwoordelijkheid.

Wat eruit **moet** (projectspecifiek):

- `crawler/fetching/fetcher.py` en `crawler/fetching/network/preflight/executor.py` mogen niet langer direct
  `RetryManager` aansturen of ontvangen in hun constructor op een manier die domeinlogica aan retry koppelt.
- `Fetcher` (`crawler/fetching/fetcher.py`) moet retry-gedrag via een `RetryExecutor` interface (of delegatie vanuit
  composition) ontvangen, niet door zelf `retry_manager.run(lambda: ...)` te doen.
- `ResponseBodyReader` (`crawler/fetching/network/body/reader.py`), `FetchRetryExecutor`, `FetchProfileExecutor` etc.
  mogen **geen** `aiohttp.ClientResponse` of `aiohttp.ClientSession` meer in hun publieke API hebben.
- `FetchResult` (`crawler/fetching/results/result.py`) en `HostProfilePreferenceStore` mogen geen `datetime.now()`
  meer aanroepen.
- `PartialPayloadStorage` en body writers mogen geen directe FS calls buiten `FileSystem`/`AtomicWriter` (zie 065).

Kort: de **afhankelijkheidsrichting** is het probleem (crawler → orchestration of directe infra), niet per se de
aanwezigheid van backoff-code binnen een executor.

## Gewenste architectuur

```
FetchCoordinator / Fetcher   (alleen interfaces)

        │
        ├────────────── HttpSessionProvider (interface + impl in network/)
        │
        ├────────────── RetryExecutor (interface)
        │                  └── FetchRetryExecutor (mag backoff + rate hints bevatten)
        │
        ├────────────── ResponseHandler (status + redirect + body)
        │                  └── ResponseBodyReader (abstract + concrete impl)
        │
        └────────────── PreflightService (robots + head + usefulness)
```

Alleen `orchestration/composition/runtime/build_fetch_services.py` (en eventueel kleinere builders) weet van concrete types zoals
`aiohttp` en bouwt de graph.

## Definition of Done — 062

□ Geen directe `aiohttp.ClientSession` buiten providers (`HttpClientSessionProvider` in
`crawler/fetching/network/session.py` en composition root
`orchestration/composition/runtime/build_fetch_services.py`).  
□ Geen `sleep()` of backoff-logica buiten `FetchRetryExecutor` (`crawler/fetching/execution/attempt.py`).  
□ Fetch geïsoleerde validatiechecks draaien zonder netwerk (via `MockHttpClientSessionProvider` + `FakeRetryExecutor`).  
□ `Fetcher`, `HeadPreflightService` en response handlers in `crawler/fetching/` ontvangen gedrag via interfaces (
`RetryExecutor`, abstract response reader etc.) en niet via directe `RetryManager` of aiohttp types.  
□ `FetchResult` (in `crawler/fetching/results/result.py`) en `HostProfilePreferenceStore` gebruiken `Clock` i.p.v.
`datetime.now(timezone.utc)`.  
□ Import-boundary checks groen (de geautomatiseerde import-boundary check).  
□ Snapshot-document (`fase-9-current-violations-snapshot.md` Sectie 1 + 4) bijgewerkt.  
□ ADR-0001 niet geschonden (geen `crawler` → `orchestration`, geen directe aiohttp in domein signatures buiten network
providers).

## Architectural Risk

| Onderdeel | Risico                                                                                                                                               |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 062 Fetch | ⭐⭐⭐ Hoog (Fetch is het hart van de crawler. Veranderingen raken vrijwel alle crawl taken, redirects, preflights, media body reads en rate limiting.) |

Mitigatie: canary runs, recorded sessions, schemachecks op interfaces, golden fetch responses.

**Zie Snapshot**: Sectie 4 (Fetch / HTTP / retry) en Sectie 1 (datetime in result.py).

## Concrete stappen (projectspecifiek)

1. Inventariseer **exact** alle directe `aiohttp.Client*`, `datetime.now`, `RetryManager` calls in:
    - `crawler/fetching/execution/attempt.py`
    - `crawler/fetching/execution/attempt.py` + `crawler/fetching/response/body_reader.py`
    - `crawler/fetching/network/body/reader.py` + `response_body_stream_reader.py` + `response_body_read_decoder.py` +
      `stream_writer.py` + `partial_store.py`
    - `crawler/fetching/network/preflight/executor.py`
    - `crawler/fetching/response/validator.py` + `status_handler.py`
    - `crawler/fetching/fetcher.py`
    - `crawler/fetching/results/result.py`
    - `crawler/fetching/profiles/host_preferences.py`
    - `orchestration/composition/runtime/build_fetch_services.py`
2. Definieer / versterk interfaces in de juiste packages:
    - `HttpClientSessionProvider` (opschonen in `crawler/fetching/network/session.py`)
    - `RetryExecutor` protocol
    - `ResponseBodyReader` + `ResponseHandler` abstracties
    - `PreflightService`
3. Vervang `datetime.now(timezone.utc)` in `crawler/fetching/results/result.py:100` en
   `crawler/fetching/profiles/host_preferences.py:148` door de `Clock` abstractie (zie 065 Storage).
4. Verplaats alle `aiohttp.ClientResponse` handling naar adapter-achtige lagen binnen `crawler/fetching/network/body/`
   en `response/`.
5. Laat `Fetcher` (`crawler/fetching/fetcher.py`) en `HeadPreflightService` (
   `crawler/fetching/network/preflight/executor.py`) retry delegeren via een `RetryExecutor` interface i.p.v. directe
   `RetryManager` (`crawler/governance/retry/retry_manager.py`).
6. Splits `build_fetch_services()` verder waar nodig (zie 068 Composition roots).
7. Update alle call sites in `crawler/fetching/*` en `crawler/` en zorg dat
   `orchestration/composition/runtime/build_fetch_services.py` de enige plaats is die concrete providers bouwt.

## Teststrategie

- Geïsoleerde validatiechecks gebruiken `MockHttpSessionProvider` + `FakeRetryExecutor` + `InMemoryResponseBodyReader`.
- Schemachecks voor "retry executor moet deterministic delays ondersteunen".
- Replaychecks met bestaande fetch logs.
- Meet % fetchchecks zonder netwerk (doel > 95%).

## PR-opdeling (fijnmazig — reviewbaar)

**062.1** — Inventarisatie fetch dependencies

- Volledige lijst van alle directe `aiohttp.ClientSession` / `ClientResponse`, `datetime`, retry aanroepen in
  `crawler/fetching/network/`, `crawler/fetching/fetcher.py`, `crawler/fetching/results/result.py`,
  `crawler/fetching/profiles/host_preferences.py` en `orchestration/composition/runtime/build_fetch_services.py`.
- Update `fase-9-current-violations-snapshot.md` Sectie 4 en 1.
- Huidige `FetchServices` dataclass + `build_fetch_services()` exact in kaart brengen.

**062.2** — Session provider

- Definieer / versterk `HttpClientSessionProvider` interface (bestaande in
  `crawler/fetching/network/session.py`).
- Alle sessie creatie/lifecycle uitsluitend via deze provider vanuit `orchestration/composition/runtime/build_fetch_services.py`.
- Verwijder `aiohttp` uit publieke signatures van `Fetcher`, `FetchProfileExecutor` etc.
- Voeg mock provider toe voor validatie.

**062.3** — Retry executor

- Introduceer `RetryExecutor` interface (of protocol).
- Centraliseer retry + backoff + rate limit hints volledig in `FetchRetryExecutor` (
  `crawler/fetching/execution/attempt.py`).
- Geen `sleep()` of backoff-logica buiten `FetchRetryExecutor`.
- `Fetcher` en `HeadPreflightService` ontvangen retry gedrag via de interface i.p.v. directe `RetryManager`.
- Geïsoleerde validatiechecks voor retry zijn deterministisch (geen echte delays).

**062.4** — Response handling

- `ResponseBodyReader`, `FetchResponseStatusHandler`, `FetchResponseValidator`, `FetchedPayloadAssembler`, redirect
  handling, `ResponseStreamWriter` etc.
- Verplaats alle `aiohttp.ClientResponse` / `ClientSession` gebruik naar interne adapters binnen
  `crawler/fetching/network/body/` en `response/`.
- Publieke API van response handling bevat geen aiohttp types meer.

**062.5** — Cleanup & oude constructors verwijderen

- Volledige abstractie van `HeadPreflightService` + overige.
- Verwijder oude constructors, directe aanroepen, imports van aiohttp in domeinlagen.
- Update alle call sites in `crawler/`.
- `build_fetch_services()` opschonen / splitsen (verwijzing naar 068).
- Import-boundary checks groen, snapshot bijgewerkt, oude code verwijderd.

**Zie Snapshot**: Sectie 1 en Sectie 4 voor de exacte bestanden die in deze PRs aangepakt moeten worden.

### Rollback — 062

- `Fetcher`, `FetchResult` en `RetryManager` hebben uitsluitend de actuele
  geïnjecteerde schema's; call-sites worden atomair gemigreerd.
- Verwijder oude code in dezelfde wijziging na geslaagde checks en validatie.
- Bij problemen: herstel de oude directe paden tijdelijk en herplan de abstractie-stap.

---

# 063 — Dependency Injection volledig doorvoeren in Media Processing

## Scope

**In scope**

✔ Alle directe `cv2`, `av`, `PIL` en ffmpeg aanroepen in `preprocessing/media/`, `preprocessing/media/video/`,
`preprocessing/media/audio/`, `preprocessing/validation/`, `augmentation/video/`, `augmentation/image/`
✔ Frame sampling, metadata extractie, transcode, blur, keyframe, audio decode, speech (in de bovengenoemde packages)
✔ Filesystem operaties binnen media (mkdir, temp files, output directories) — te vervangen door `FileSystem` /
`TempDirectoryStrategy` vanuit orchestration/composition
✔ Adapters + decoders + `FileSystem` / `TempDirectoryStrategy`

**Out of scope**

✖ Volledige preprocessing pipeline orchestration  
✖ Training image alignment (alleen de media delen)  
✖ Dataset collation / tensor materialisatie

---

## Huidige situatie

Directe of lazy native library imports buiten adapters in:

- `crawler/extraction/payloads/video_payload_extractor.py`
- `preprocessing/privacy/inspection/local_visual_analysis.py`
- `preprocessing/media/adapters/audio_decode.py`
- `crawler/analysis/enrichment/speech/speech_transcriber.py`
- `augmentation/video/video_keyframe_augmenter.py`, `augmentation/video/video_clip_augmenter.py`
- `augmentation/image/image_augmenter.py`, `augmentation/image/image_operations.py`
- `crawler/curation/snapshots/alignment_rows.py`

Daarnaast directe `Path(...).mkdir()`, `.open()`, `tempfile` logica in domeincode, o.a.
`crawler/analysis/enrichment/video/video_frame_sampler.py`, `augmentation/cache.py` en de
`crawler/storage/datasets/writing/...`-modulegroep.

**Zie Snapshot**: Sectie 2 — Media / native libraries.

## Mag vs Moet

Adapters mogen `cv2`/`av` lazy importeren.  
Wat eruit moet: directe imports en calls in `VideoFrameSampler`, `ImageAugmenter`, validators, `raw_inputs.py` etc.

## Gewenste architectuur

```
preprocessing/media/* + preprocessing/video/* + preprocessing/audio/* + augmentation/video/* + augmentation/image/*  (en augmentation)

        │
        ├──── VideoDecoder (interface)          [impl in preprocessing/media/adapters/opencv_video.py of pyav_adapter.py]
        ├──── ImageDecoder / ImageOperations
        ├──── AudioDecoder (preprocessing/media/adapters/audio_decode.py)
        ├──── FileSystem (interface)            [opgebouwd vanuit orchestration/composition/media/ of preprocessing/media/filesystem/]
        └──── TempDirectoryStrategy
```

Vervang directe Path-, tempfile- en open()-aanroepen in `preprocessing/media/`, `preprocessing/media/video/`,
`preprocessing/media/audio/` en gerelateerde augmentation packages door de gedeelde `FileSystem`- en
`TempDirectoryStrategy`-interfaces die vanuit `orchestration/composition/media/` (of
`orchestration/composition/runtime/build_document.py`) worden opgebouwd.

Concrete adapters leven in `preprocessing/media/adapters/` en `augmentation/.../adapters/`. Nooit directe cv2/av/PIL
imports buiten adapters.

## Definition of Done — 063

□ Geen `import cv2` / `import av` / `from PIL import ...` / directe ffmpeg calls buiten `preprocessing/media/adapters/`,
`augmentation/video/adapters/`, `augmentation/image/adapters/` en vergelijkbare adapter directories.  
□ Vervang directe `Path`, `tempfile` en `open()`-aanroepen in `preprocessing/media/`, `preprocessing/media/video/`,
`preprocessing/media/audio/`, `augmentation/video/*` en `augmentation/image/*` door de gedeelde `FileSystem` en
`TempDirectoryStrategy` interfaces (opgebouwd in `orchestration/composition/`).  
□ Alle media geïsoleerde validatiechecks in `preprocessing/` en gerelateerd draaien zonder OpenCV/PyAV/FFmpeg/PIL binaries (via
fakes/adapters).  
□ Bestaande keyframe / audio decode / OCR / frame sampling gedrag equivalent (golden files + cross-backend checks).  
□ Import-boundary checks groen + ADR-0001 niet geschonden (`preprocessing.media` importeert geen `crawler.runtime` of
`orchestration`).  
□ Snapshot Sectie 2 bijgewerkt.

## Architectural Risk

| Onderdeel | Risico                                                                                                                              |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| 063 Media | ⭐⭐ Middel (Beperkt tot preprocessing/media/* + augmentation/video/* + augmentation/image/* . Subtiele decode verschillen mogelijk.) |

**Zie Snapshot**: Sectie 2.

## PR-opdeling (fijn)

**063.1** — Definieer interfaces + eerste adapters (video).  
**063.2** — Video frame sampling + keyframe + metadata.  
**063.3** — Audio decode + transcription.  
**063.4** — Image / PIL + document paths + raw_inputs.  
**063.5** — Filesystem + TempDirectoryStrategy + cleanup.  
**063.6** — Augmentation adapters + checks.

---

# 065 — Dependency Injection voor Storage (inclusief FetchResult & Manifests)

## Scope

**In scope**

✔ Alle `datetime.now` en `uuid.uuid4` in storage writers (`crawler/storage/datasets/writing/*`), fetch results (
`crawler/fetching/results/result.py`), manifests, snapshots (`mmcrawler_datasets/snapshots/*`), datachecker, training
status, augmentation cache
✔ Directe `Path` + `open` + mkdir + tempfile in `crawler/storage/datasets/writing/`,
`crawler/storage/datasets/run_layout/`, `mmcrawler_datasets/snapshots/`, `datachecker/artifacts/manifest_file_writer.py`
✔ `AtomicWriter`, `PathLayout` (uitbreiding `DatasetPathLayout`), `Clock`, `IdGenerator` introductie + injectie vanuit
`orchestration/composition/`

**Out of scope**

✖ Volledige dataset collation en dataloader logica  
✖ Training checkpoint internals (alleen de writers die we raken)

---

## Huidige situatie

- `datetime.now(timezone.utc)` in `crawler/fetching/results/result.py:100`,
  `crawler/storage/datasets/writing/dataset_writer.py:291`, `crawler/storage/datasets/records/failed_task_record_assembler.py:65`,
  `mmcrawler_datasets/snapshots/*`, `training/*`, `augmentation/cache.py`,
  `datachecker/artifacts/manifest_file_writer.py:182,390` etc.
- `uuid.uuid4()` in `raw_payload_writer.py` (voor .tmp bestanden) en video frame sampling.
- Directe `Path`, `.mkdir(parents=True)`, `.open()` in `raw_payload_writer.py`, `dataset_manifest_writer.py`,
  `dataset_path_layout.py` etc.

**Zie Snapshot**: Sectie 1 en Sectie 3.

## Gewenste architectuur (projectspecifiek)

Vervang directe `Path(...)`, `.open()`, `.mkdir(parents=True)`, `tempfile` en `uuid.uuid4()`-aanroepen in:

- `crawler/storage/datasets/writing/raw_payload_writer.py`
- `crawler/storage/datasets/manifests/dataset_manifest_writer.py`
- `crawler/storage/datasets/writing/dataset_writer.py`
- `crawler/storage/datasets/records/failed_task_record_assembler.py`
- `crawler/storage/datasets/run_layout/dataset_path_layout.py`
- `mmcrawler_datasets/snapshots/output_writer.py`
- `augmentation/assembly/outputs.py`
- `datachecker/artifacts/manifest_file_writer.py`
- `crawler/fetching/results/result.py`
- `augmentation/cache.py`

door een gedeelde `AtomicWriter`, `PathLayout` / uitbreiding van `DatasetPathLayout`, `Clock` en `IdGenerator`,
opgebouwd vanuit `orchestration/composition/storage/` (of `orchestration/composition/runtime/dataset_services.py` +
bestaande layout klassen).

Vervang alle `datetime.now(timezone.utc)` en `uuid.uuid4()` die zijn geïnventariseerd in
`docs/architecture/fase-9-current-violations-snapshot.md` Sectie 1 door `Clock` + `IdGenerator` (SystemClock /
FixedClock + SequenceIdGenerator), zodat manifests, snapshots, fetch results en training status deterministisch testbaar
worden.

### Training job status (opgelost)

- Pad: `training/runtime/job_status/` met gescheiden identity-, payload-, persistence- en storemodules (voorheen in `orchestration/runtime/`).
- Boundary: `training` importeert **geen** `crawler.runtime`. Orchestration injecteert
  `now=clock.now` en `generate_id=id_generator.generate` als smalle callables.
- Identity: `TrainingJobIdentity(snapshot_id, attempt_id)` — canonieke snapshot-ID uit
  `WorkflowExecutionPlan.training_snapshot_id`, attempt-ID per run via id-generator.
  Statusbestand: `jobs/{snapshot_id}/{attempt_id}/status.json`.
- Statusschema v1: aparte velden `status`, `training_status`,
  `artifact_persistence_status` (geen samengestelde statusstrings).

## Definition of Done — 065

□ Geen `datetime.now(timezone.utc)`, `uuid.uuid4()`, directe `Path().open/mkdir`, `tempfile` of raw FS mutaties meer in
de storage/fetch result/manifest bestanden genoemd in Gewenste architectuur.  
□ `FetchResult`, manifests (`dataset_manifest_writer.py` in `crawler/storage/datasets/manifests/`), snapshots (`output_writer.py` etc. in `mmcrawler_datasets/snapshots/`) en training
status stores gebruiken `Clock` + `IdGenerator` (FixedClock + SequenceIdGenerator voor validatie).  
□ `AtomicWriter` (write-to-tmp + atomic rename) gebruikt in `crawler/storage/datasets/writing/` en
`datachecker/artifacts/`.  
□ Manifests, snapshots en fetch results zijn reproduceerbaar/deterministisch.  
□ Import-boundary checks + ADR-0001 niet geschonden.  
□ Snapshot Sectie 1 + 3 bijgewerkt.

## Architectural Risk

| Onderdeel   | Risico                                                                                                                                                                                                            |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 065 Storage | ⭐⭐⭐ Hoog (Storage verandert. Raakt manifests, snapshots, fetch resultaten in `crawler/fetching/results/`, `crawler/storage/datasets/writing/`, `mmcrawler_datasets/snapshots/`, `datachecker/artifacts/`, training status.) |

**Zie Snapshot**: Sectie 1 (Directe tijd- en ID-generatie) + Sectie 3 (Filesystem).

## PR-opdeling (fijn)

**065.1** — `Clock` + `IdGenerator` interfaces + `SystemClock` / `FixedClock` + `SequenceIdGenerator`. Vervang aanroepen
in `result.py` en de top 5 storage bestanden uit de snapshot.  
**065.2** — `AtomicWriter` (write-to-tmp + atomic rename).  
**065.3** — `PathLayout` / uitbreiding `DatasetPathLayout` + grootschalige vervangingen in writers + run layout.  
**065.4** — Manifest/snapshot determinism + checks + cleanup.

### Rollback — 065 (samengevat)

- Oude directe `datetime.now()` / `uuid` / `Path` calls blijven tijdelijk bestaan naast de nieuwe `Clock`/`IdGenerator`/
  `AtomicWriter`.
- Oude codepaden pas verwijderen na volledige migratie van alle afhankelijke callers + checks.
- Bij issues: tijdelijk terugschakelen in composition root.

---

# Overige secties (064, 066, 067, 068) — samengevat met dezelfde structuur

(Dezelfde patronen gelden. Elk onderdeel krijgt expliciete packages, DoD, risico en migratiestappen.)

## 064 — Source Registry

**Scope (projectspecifiek)**: Registry file loading + parsing in `crawler/governance/discovery/`, `crawler/curation/`,
`config/source_catalog/registry_settings.py`, `orchestration/composition/governance/`.

**Gewenste architectuur**: Vervang directe `Path`, `open()`, `toml.load` etc. in governance/discovery door
`FilesystemRegistryReader` + `FileSystem` interface, opgebouwd vanuit `orchestration/composition/`.

**Definition of Done — 064**
□ Geen directe `Path` + parsing in `crawler/governance/*` of `crawler/discovery/*`; alleen via
`FilesystemRegistryReader` / `FileSystem`.
□ Registry tests deterministisch zonder echte FS.
□ Import-boundary checks groen + ADR-0001.

**Risk**: ⭐⭐ Middel

**PRs**:

- 064.1 Model + `RegistryReader` interface
- 064.2 Reader implementatie + migratie van `crawler/governance/discovery/*`
- 064.3 Injectie in composition roots + opruimen oude constructors

## 066 — Scheduler State

**Scope (projectspecifiek)**: Checkpoint stores, dead letter queues, hidden caches in `crawler/scheduling/`,
`crawler/worker/`, `orchestration/composition/runtime/build_scheduler_services.py`, `crawler/worker_scaling/`.

**Doel**: Herstel scenario's volledig testbaar met fakes (in-memory stores) zonder volledige runtime of echte
schedulers.

**Definition of Done — 066**
□ Geen directe tijd/FS/ID calls of verborgen state in scheduler/worker domeincode buiten `Clock`/`AtomicWriter`/
`IdGenerator`.
□ Alle checkpoint/dead-letter/recovery geïsoleerde validatiechecks draaien met fakes (geen DB/FS/network).
□ ADR-0001: `crawler` importeert geen `orchestration`.

**Risk**: ⭐⭐⭐ Hoog (raakt recovery paths en worker stability).

**PRs**:

- 066.1 `SchedulerStateStore` / `CheckpointStore` interfaces
- 066.2 In-memory + file adapters + migratie
- 066.3 Injectie + opruimen in `build_scheduler_services.py` + worker

### Rollback — 066

- Oude checkpoint/dead-letter/recovery stores blijven tijdelijk naast de nieuwe `SchedulerStateStore` interfaces (
  parallel support).
- Injectie van fakes/adapters gebeurt stapsgewijs; volledige verwijdering van oude stateful code pas na alle
  worker/scheduler call-sites gemigreerd en herstel-checks geslaagd.
- Bij instabiliteit in recovery: herintroduceer tijdelijk de oude implementatie in de composition root.

## 067 — Dependency Detection

**Huidige situatie**: `orchestration/composition/runtime/optional_dependency_validator.py` + verspreide
`try: import cv2` / `import av` in `crawler/extraction/payloads/*`, `preprocessing/privacy/inspection/*`,
`augmentation/image/*` (alleen `av` resteert nog uitsluitend in `preprocessing/media/adapters/pyav_media.py`).

**Doel**: Eén `DependencyProbeService` (gecached, in orchestration/composition) die door alle lagen gebruikt wordt.

**Definition of Done — 067**
□ Alle native dependency checks lopen via centrale `DependencyProbeService` / `OptionalDependencyValidator`.
□ Geen verspreide `try: import` buiten de probe + adapters.
□ Laag risico; checks groen.

**Risk**: ⭐ Laag (beperkt tot probing).

**PRs**: 067.1 Probe service + composition → 067.2 Vervang try/except → 067.3 Opruimen.

## 068 — Composition Roots vereenvoudigen

**Huidige situatie**:

- `orchestration/composition/runtime/build_fetch_services.py` (`build_fetch_services`)
- `orchestration/composition/runtime/build_task_processor.py`
- `orchestration/composition/runtime/build_scheduler_services.py`
- `orchestration/composition/runtime/build_crawler_runtime_services.py`
- `orchestration/composition/runtime/build_dataset_services.py`
- `orchestration/composition/runtime/build_worker_services.py` etc.
  zijn groot en bevatten validatie + mkdir logica.

**Doel**: Kleine builders (één cluster per builder: fetch, storage, media, scheduler, ...), alleen wiring + object
creatie. Validatie, dir setup en business rules elders (in domain of dedicated services).

**Definition of Done — 068**
□ Geen builder > ~180 regels.
□ Iedere builder unit-testbaar in isolatie (via fakes).
□ Volledige dependency graph expliciet zichtbaar (fetch, storage, media, governance).
□ Validatie + mkdir verplaatst uit builders.
□ ADR-0001 gerespecteerd.

**Risk**: ⭐⭐ Middel

**PRs**:

- 068.1 Richtlijn en reviewcriteria voor modulegrootte
- 068.2 Split fetch builders (bouw voort op 062.5)
- 068.3 Split scheduler + runtime + dataset services
- 068.4 Split media / processing builders
- 068.5 Validatie & dir creation extractie

---

## Fase-brede risico's (samengevat)

| Onderdeel             | Risico    |
|-----------------------|-----------|
| 062 Fetch             | ⭐⭐⭐ Hoog  |
| 063 Media             | ⭐⭐ Middel |
| 064 Registry          | ⭐⭐ Middel |
| 065 Storage           | ⭐⭐⭐ Hoog  |
| 066 Scheduler         | ⭐⭐⭐ Hoog  |
| 067 Dependency probes | ⭐ Laag    |
| 068 Composition roots | ⭐⭐ Middel |

---

## Volgende stappen

1. Gebruik de snapshot (`fase-9-current-violations-snapshot.md`) als bron voor
   exacte locaties, bestanden en lijnen.
2. Begin met 062.1 + 065.1 (hoogste validatiestabiliteit winst + grootste risico). Volg strikt de **Migratiestrategie**.
3. Update na iedere sub-PR de snapshot secties, de DoD checklists en de centrale dependency grafiek.
4. Voeg import-boundary checks en builder-size guards toe waar ze nog ontbreken.
5. Voer na elke PR de geautomatiseerde import-boundary check + relevante geïsoleerde validatiechecks uit (fetch, storage, media).

**Belangrijk**: Houd je aan de migratiestrategie. Nooit interface + volledige vervanging + logica-wijzigingen in
dezelfde PR.

Dit document is nu projectspecifiek (met expliciete package-namen zoals `preprocessing/media/`,
`crawler/fetching/execution/attempt.py`, `orchestration/composition/runtime/build_fetch_services.py`), bevat een centrale
grafiek, Definition of Done per onderdeel, fijne PR-opdeling, risicoklassen en de volledige migratiestrategie.
