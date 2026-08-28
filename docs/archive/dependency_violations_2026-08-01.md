# FASE 9 — Snapshot van huidige Dependency Inversion schendingen

**Status**: Archive / Dated snapshot (2026-08-01)
**Superseded by**: [../architecture/infrastructure_boundaries.md](../architecture/infrastructure_boundaries.md) (current architecture) and [../roadmaps/dependency_injection_phase_9.md](../roadmaps/dependency_injection_phase_9.md) (planning)

**Datum**: 2026-07-04  
**Bron**: Statische zoekopdrachten op de cleaned codebase.
**Laatst geverifieerd tegen source tree**: 2026-08-01

Dit is een **startpunt** voor de concrete stappen in `fase-9-dependency-injection.md`.  
Gebruik dit als input voor PR 062.1, 063.1, 065.1 etc.

**Onderhoud**: Dit document is een momentopname en moet na iedere afgeronde PR worden bijgewerkt. De roadmap (
fase-9-dependency-injection.md) verwijst naar sectienummers uit deze snapshot; wijzigingen aan de snapshot vereisen een
update van de referenties in de roadmap en PR-opdelingen.

**Belangrijk**: De gedetailleerde Current Architecture grafieken, **Centrale Dependency Grafiek**, Scope, Definition of
Done per onderdeel, Mag-vs-Moet, Architectural Risk tabellen (met risicoklassen), Migratiestrategie en projectspecifieke
acties (met exacte package-namen) staan in `fase-9-dependency-injection.md`.  
Verwijs in die documentatie expliciet naar de secties hieronder. Gebruik dit snapshot als input voor PR 062.1, 063.1,
065.1 etc.

---

## 1. Directe tijd- en ID-generatie (relevant voor 065 + 062)

Bestanden met `datetime.now` of `uuid.uuid4` in domein-/storage-code:

- `preprocessing/privacy/artifacts.py:385`
- `training/export/export.py:526`
- `crawler/governance/retention_executor.py:48,100,254` (datetime.now + uuid4 voor staging/audit)
- `crawler/governance/processing_activity.py:39`
- `crawler/governance/training_permission.py:101`
- `crawler/storage/datasets/records/governance.py:85,92`

**Opgelost (niet langer een violation):**

- `training/runtime/job_status/` — `TrainingJobStatusStore` injecteert `now` en
  `generate_id` als callables (geen `crawler.runtime`, geen `datetime.now` /
  `uuid.uuid4`). Statuspad = `jobs/{snapshot_id}/{attempt_id}/status.json` via
  expliciete `TrainingJobIdentity` (geen basename van `training_root`).
- `datachecker/artifacts/manifest_file_writer.py` — idem: callables i.p.v.
  `crawler.runtime.Clock` / `IdGenerator`.

**Actie**: Vervang door `Clock` + `IdGenerator` volgens de Migratiestrategie in fase-9-dependency-injection.md (nieuwe
interface → adapter → composition root → migratie → opruimen).

---

## 2. Media / native libraries (063 + 067)

Directe of lazy imports buiten adapters:

**OpenCV (cv2)**:

- `crawler/extraction/payloads/video_payload_extractor.py`
- `preprocessing/privacy/inspection/local_visual_analysis.py`

**PyAV (av)**:

- Geen resterende violations: enige importeur is de adapter
  `preprocessing/media/adapters/pyav_media.py`.

**PIL**:

- `augmentation/document/document_page_augmenter.py`
- `augmentation/equivalence.py`
- `augmentation/image/image_operations.py`
- `augmentation/image/image_augmentation_validation.py`
- `augmentation/image/image_artifact_writer.py`
- `augmentation/image/image_operation_executor.py`
- `crawler/analysis/enrichment/image/image_metadata_reader.py`
- `crawler/analysis/enrichment/ocr/tesseract_engine.py`
- `crawler/extraction/payloads/image_payload_extractor.py`
- `preprocessing/privacy/remediation/images/mask_sensitive_regions.py`
- `preprocessing/media/image/*`

**Actie**: Verplaats naar `preprocessing/media/adapters/` + `augmentation/.../adapters/`.

---

## 3. Filesystem operaties in domein (064, 065, 063)

Veel `.mkdir(parents=True)`, `.open()`, `Path(...)` mutaties in:

- `crawler/storage/datasets/writing/raw_payload_writer.py`
- `crawler/storage/datasets/manifests/dataset_manifest_writer.py`
- `crawler/storage/datasets/run_layout/dataset_path_layout.py`
- `crawler/storage/datasets/writing/dataset_write_journal.py`
- `crawler/storage/datasets/sync_index/*`
- `crawler/curation/snapshots/...`
- `crawler/analysis/enrichment/video/video_frame_sampler.py` (output_directory + mkdir)
- `augmentation/cache.py` (staging + uuid4 .tmp)
- Verspreid in `crawler/analysis/...`, `crawler/processing/...`

**Actie**: Introduceer `AtomicWriter`, `PathLayout`, `FileSystem` + `TempDirectoryStrategy` interfaces (zie expliciete
paden in fase-9-dependency-injection.md). Volg migratiestrategie.

---

## 4. Fetch / HTTP / retry (062)

- `crawler/fetching/fetcher.py`, `crawler/fetching/execution/attempt.py`,
  `crawler/fetching/feedback/attempt_recorder.py` bevatten retry/rate-limit-afhandeling met tijdafhankelijkheid.
- Veel directe `aiohttp` types in signatures (`Fetcher`, etc.) — let op: `FetchRetryExecutor`, `ResponseBodyReader`,
  `HttpClientSessionProvider` en `HeadPreflightService` zijn *toegestane* infrastructuurimplementaties (mogen concrete
  libs gebruiken), maar mogen geen concrete types lekken naar domeincode of call-sites.
- `build_fetch_services` in `orchestration/composition/runtime/build_fetch_services.py` is erg groot.
- Domeincode (fetcher, preflight call-sites, results) lekt nog of roept direct infra aan.

**Actie**: Volg de extractie in 062.1 t/m 062.5 PRs (fijnmazig: inventarisatie → session provider → retry executor →
response handling → cleanup). Zie Definition of Done — 062 en centrale grafiek (orchestration/composition → crawler
toegestaan; omgekeerd verboden). Maak onderscheid tussen toegestane infra-impls en ongewenste lekken (zie
fase-9-dependency-injection.md).

---

## 5. Composition roots / builders (068)

Grote builders (indicatief):

- `orchestration/composition/runtime/build_fetch_services.py` (`build_fetch_services`)
- `orchestration/composition/runtime/build_task_processor.py`
- `orchestration/composition/runtime/build_scheduler_services.py`
- `orchestration/composition/runtime/build_*_services.py` (meerdere)

Daarnaast nog logica (validatie + mkdir) in sommige builders.

**Meten**: Voeg builder-grootte check toe (zie plan).

---

## 6. Dependency detection (067)

Bestaand:

- `orchestration/composition/runtime/optional_dependency_validator.py`

Verspreid:

- Meerdere `try: import cv2` / `import av` in media en augmentation (zie hierboven).

**Actie**: Centraliseer in `DependencyProbeService` en laat de bestaande validator hierop leunen.

---

## Aanbeveling

1. Begin met een gedetailleerde, per-module inventarisatie en focus op de exacte
   bestanden in secties 1-6.
2. Focus eerst op 065.1 (Storage) + 062.1 (Fetch) → grootste validatiestabiliteit + hoogste risico (⭐⭐⭐).
3. Maak de abstracties (`Clock`, `IdGenerator`, `AtomicWriter`, `HttpClientSessionProvider`, `RetryExecutor`,
   `VideoDecoder`, `FileSystem`, `TempDirectoryStrategy`, ...) in kleine, reviewbare stappen volgens de *
   *Migratiestrategie** (nooit alles in één PR).
4. Na elke sub-PR: update snapshot + voer import-boundary en geïsoleerde validatiechecks uit zonder netwerk/FS/binaries.

Dit snapshot kan na elke PR geüpdatet worden (of automatisch gegenereerd). Koppel altijd terug naar de centrale
dependency grafiek en Definition of Done in het hoofd document.
