"""Load raw crawl manifest records for curation input assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from config.path_resolution.project_paths import ProjectPaths
from crawler.curation.ingest.schema.entry import RawManifestEntry
from crawler.curation.ingest.schema.record import RawManifestRecord
from crawler.governance.deletion_index import ensure_asset_trainable
from crawler.storage.datasets.run_layout.dataset_path_layout import (
    output_subdirectory,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from config.settings.datasets import (
        DatasetPathSettings,
        RawManifestReaderSettings,
    )


class CurationInputLoader:
    """Load typed raw manifest records from the configured raw dataset layer."""

    def __init__(
        self,
        *,
        settings: RawManifestReaderSettings,
        dataset_paths: DatasetPathSettings,
        project_root: Path,
        logger: ProjectLogger,
        minimum_modality_counts: Mapping[str, int] | None = None,
    ) -> None:
        self._settings = settings
        self._dataset_paths = dataset_paths
        self._project_root = project_root
        self._logger = logger
        self._requested_run_ids = frozenset(
            run_id.strip()
            for run_id in self._settings.selected_run_ids
            if run_id.strip()
        )
        self._deletion_index_path = (
            ProjectPaths(project_root=self._project_root).resolve(
                Path(self._dataset_paths.output_directory)
            )
            / self._settings.deletion_index_filename
        )
        self._minimum_modality_counts = {
            str(kind).strip().lower(): max(0, int(minimum))
            for kind, minimum in (minimum_modality_counts or {}).items()
            if str(kind).strip()
        }

    def read_all(self) -> tuple[RawManifestEntry, ...]:
        """Return all available raw manifest entries for the active subdirectory."""

        return tuple(self.iter_all())

    def selected_modality_counts(self) -> dict[str, int]:
        """Return aggregated modality counts for exactly the selected raw runs.

        Uses the same manifest discovery and run selection (all, latest,
        coverage_best, coverage_combined, selected_run_ids) as the raw
        record loader, so validation judges the same input set that
        preprocessing will consume.
        """

        counts: dict[str, int] = {}
        for manifest_path, _ in self._selected_manifest_specs():
            for kind, count in self._count_manifest_kinds(
                manifest_path=manifest_path
            ).items():
                counts[kind] = counts.get(kind, 0) + count
        return counts

    def selected_record_count(self) -> int:
        """Return the total record count across the selected raw runs."""

        return sum(self.selected_modality_counts().values())

    def selected_crawl_evidence(self) -> SelectedCrawlEvidence | None:
        """Aggregate acquisition-health evidence across the selected runs.

        Reads the ``output_readiness`` block of each selected run summary
        and returns summed successful requests, summed object records, and
        an object-weighted mean quality score. Fail-closed: returns
        ``None`` as soon as any selected run lacks valid readiness
        evidence, so partial evidence never passes acquisition gates.
        """

        total_objects = 0
        total_successful_requests = 0
        weighted_quality = 0.0
        observed_runs = 0
        for manifest_path, manifest_name in self._selected_manifest_specs():
            run_directory = self._resolve_run_directory(
                manifest_path=manifest_path,
                manifest_name=manifest_name,
            )
            summary_path = (
                run_directory
                / Path(manifest_name).parent
                / "run_manifest.json"
            )
            evidence = _read_run_output_readiness(summary_path=summary_path)
            if evidence is None:
                return None
            observed_runs += 1
            total_objects += evidence.object_records_total
            total_successful_requests += evidence.successful_requests_total
            weighted_quality += (
                evidence.quality_score * evidence.object_records_total
            )

        if observed_runs == 0:
            return None
        quality_score = (
            weighted_quality / total_objects if total_objects > 0 else 0.0
        )
        return SelectedCrawlEvidence(
            object_records_total=total_objects,
            successful_requests_total=total_successful_requests,
            quality_score=quality_score,
        )

    def _selected_manifest_specs(
        self,
    ) -> tuple[tuple[Path, str], ...]:
        root = self._resolve_manifests_root()
        if not root.exists():
            self._raise_for_missing_requested_runs(found_final=frozenset())
            return ()
        return self._discover_manifest_specs(root=root)

    def iter_all(self) -> Iterator[RawManifestEntry]:
        """Yield raw manifest entries lazily across discovered run manifests."""

        manifests_root = self._resolve_manifests_root()

        if not manifests_root.exists():
            self._logger.warning(
                "raw_manifest_root_missing",
                path=manifests_root.as_posix(),
            )
            self._raise_for_missing_requested_runs(found_final=frozenset())
            return

        loaded_records = 0
        manifest_specs = self._discover_manifest_specs(root=manifests_root)
        selected_run_ids = sorted(
            {
                self._resolve_run_directory(
                    manifest_path=manifest_path,
                    manifest_name=manifest_name,
                ).name
                for manifest_path, manifest_name in manifest_specs
            }
        )

        for manifest_path, manifest_name in manifest_specs:
            run_directory = self._resolve_run_directory(
                manifest_path=manifest_path,
                manifest_name=manifest_name,
            )

            for record in self._iter_manifest_records(
                manifest_path=manifest_path
            ):
                loaded_records += 1
                yield RawManifestEntry(
                    run_directory=run_directory,
                    record=record,
                )

        self._logger.info(
            "raw_manifest_loaded",
            manifests=len(manifest_specs),
            records=loaded_records,
            root=manifests_root.as_posix(),
            selected_runs=selected_run_ids,
            selected_run_count=len(selected_run_ids),
            run_selection_mode=(self._settings.run_selection_mode),
        )

    def _iter_manifest_records(
        self,
        *,
        manifest_path: Path,
    ) -> Iterator[RawManifestRecord]:
        """Yield valid rows while rejecting schema drift fail-closed."""

        try:
            handle = manifest_path.open("r", encoding="utf-8")
        except OSError as exc:
            self._log_row_skipped(manifest_path, 0, exc)
            return

        with handle:
            line_number = 0
            while True:
                try:
                    line = handle.readline()
                except (OSError, UnicodeError) as exc:
                    self._log_row_skipped(
                        manifest_path,
                        line_number + 1,
                        exc,
                    )
                    return
                if line == "":
                    return

                line_number += 1
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise ValueError("manifest row must be a JSON object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    self._log_row_skipped(manifest_path, line_number, exc)
                    continue

                schema_version = str(payload.get("schema_version") or "")
                if (
                    schema_version
                    and schema_version.strip()
                    != self._settings.raw_schema_version.strip()
                ):
                    raise ValueError(
                        "raw manifest schema mismatch: "
                        f"path={manifest_path.as_posix()}, "
                        f"line={line_number}, "
                        f"expected={self._settings.raw_schema_version!r}, "
                        f"observed={schema_version!r}"
                    )

                try:
                    record = RawManifestRecord.from_payload(payload)
                except (KeyError, TypeError, ValueError) as exc:
                    self._log_row_skipped(manifest_path, line_number, exc)
                    continue

                ensure_asset_trainable(
                    deletion_index_path=self._deletion_index_path,
                    object_sha256=record.content_sha256,
                    max_index_bytes=self._settings.deletion_index_max_bytes,
                    max_index_rows=self._settings.deletion_index_max_rows,
                    max_row_bytes=self._settings.deletion_index_max_row_bytes,
                )
                yield record

    def _log_row_skipped(
        self,
        manifest_path: Path,
        line: int,
        exc: Exception,
    ) -> None:
        self._logger.warning(
            "raw_manifest_row_skipped",
            path=manifest_path.as_posix(),
            line=line,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def _discover_manifest_specs(
        self,
        *,
        root: Path,
    ) -> tuple[tuple[Path, str], ...]:
        names = (
            str(
                Path(self._dataset_paths.raw_sync_directory)
                / self._dataset_paths.raw_sync_current_objects_filename
            ),
        )
        seen: set[Path] = set()
        specs: list[tuple[Path, str]] = []

        for manifest_name in names:
            for path in sorted(root.rglob(manifest_name)):
                if path in seen:
                    continue
                seen.add(path)
                specs.append((path, manifest_name))

        return self._filter_manifest_specs(specs=tuple(specs))

    def _resolve_run_directory(
        self,
        *,
        manifest_path: Path,
        manifest_name: str,
    ) -> Path:
        parent = manifest_path.parent
        relative_parent = Path(manifest_name).parent
        for _ in relative_parent.parts:
            parent = parent.parent
        return parent

    def _resolve_manifests_root(self) -> Path:
        resolved_subdirectory = output_subdirectory(
            configured_subdirectory=(self._dataset_paths.output_subdirectory),
        )
        return ProjectPaths(project_root=self._project_root).resolve(
            Path(self._dataset_paths.raw_output_directory)
            / resolved_subdirectory
        )

    def _filter_manifest_specs(
        self,
        *,
        specs: tuple[tuple[Path, str], ...],
    ) -> tuple[tuple[Path, str], ...]:
        specs = tuple(
            spec for spec in specs if self._is_final_run_spec(spec=spec)
        )
        found_final = frozenset(
            self._resolve_run_directory(
                manifest_path=spec[0],
                manifest_name=spec[1],
            ).name
            for spec in specs
        )
        self._raise_for_missing_requested_runs(found_final=found_final)
        if self._requested_run_ids:
            return tuple(
                spec
                for spec in specs
                if self._resolve_run_directory(
                    manifest_path=spec[0],
                    manifest_name=spec[1],
                ).name
                in self._requested_run_ids
            )

        if not specs:
            return specs

        mode = self._settings.run_selection_mode.strip().lower()
        if mode == "all":
            return specs
        if mode in {"coverage_combined", "coverage-aware", "coverage_aware"}:
            return self._coverage_combined_specs(specs=specs)
        if mode == "coverage_best":
            return self._coverage_best_specs(specs=specs)
        if mode == "latest":
            latest_run_id = self._latest_manifest_run_id(specs=specs)
            return tuple(
                spec
                for spec in specs
                if self._resolve_run_directory(
                    manifest_path=spec[0],
                    manifest_name=spec[1],
                ).name
                == latest_run_id
            )
        raise ValueError(f"unknown raw manifest run_selection_mode: {mode}")

    def _latest_manifest_run_id(
        self,
        *,
        specs: tuple[tuple[Path, str], ...],
    ) -> str:
        latest_spec = max(
            specs,
            key=lambda spec: (
                self._manifest_freshness_timestamp_ns(spec=spec),
                self._resolve_run_directory(
                    manifest_path=spec[0],
                    manifest_name=spec[1],
                ).name,
            ),
        )
        return self._resolve_run_directory(
            manifest_path=latest_spec[0],
            manifest_name=latest_spec[1],
        ).name

    def _manifest_freshness_timestamp_ns(
        self,
        *,
        spec: tuple[Path, str],
    ) -> int:
        manifest_path, manifest_name = spec
        run_directory = self._resolve_run_directory(
            manifest_path=manifest_path,
            manifest_name=manifest_name,
        )
        records_directory = run_directory / Path(manifest_name).parent
        candidate_paths = (
            manifest_path,
            records_directory / "run_manifest.json",
            records_directory / "errors.jsonl",
        )
        timestamps: list[int] = []
        for path in candidate_paths:
            try:
                timestamps.append(path.stat().st_mtime_ns)
            except OSError:
                continue
        return max(timestamps, default=0)

    def _coverage_combined_specs(
        self,
        *,
        specs: tuple[tuple[Path, str], ...],
    ) -> tuple[tuple[Path, str], ...]:
        """Select the smallest satisfying run combination within the limit.

        Correctness first: every run's counts are capped at the canonical
        minima and a bounded 0/1 knapsack search finds a combination of at
        most ``coverage_selection_max_runs`` runs that jointly reaches all
        minima, if one exists. Fewer runs are preferred; the weighted
        coverage score is only a tie-breaker between combinations of the
        same size. If no combination satisfies the minima, the best single
        run is returned.
        """

        summaries = self._coverage_summaries(specs=specs)
        if not summaries:
            return specs

        minimum_records = self._coverage_minimum_records()
        if not minimum_records:
            return self._summaries_to_specs(summaries=summaries[:1])

        max_runs = max(1, int(self._settings.coverage_selection_max_runs))
        kinds = tuple(sorted(minimum_records))
        caps = tuple(max(0, int(minimum_records[kind])) for kind in kinds)
        reduced = [
            _capped_coverage_counts(
                counts=summary["counts"],
                kinds=kinds,
                caps=caps,
            )
            for summary in summaries
        ]

        selected = _find_satisfying_combination(
            summaries=summaries,
            reduced=reduced,
            kinds=kinds,
            caps=caps,
            max_runs=max_runs,
        )
        return self._summaries_to_specs(summaries=selected or summaries[:1])

    def _coverage_best_specs(
        self,
        *,
        specs: tuple[tuple[Path, str], ...],
    ) -> tuple[tuple[Path, str], ...]:
        summaries = self._coverage_summaries(specs=specs)
        if not summaries:
            return specs
        return self._summaries_to_specs(summaries=summaries[:1])

    def _coverage_summaries(
        self,
        *,
        specs: tuple[tuple[Path, str], ...],
    ) -> list[CoverageSummary]:
        summaries: list[CoverageSummary] = []
        for spec in specs:
            manifest_path, manifest_name = spec
            run_id = self._resolve_run_directory(
                manifest_path=manifest_path,
                manifest_name=manifest_name,
            ).name
            counts = self._count_manifest_kinds(
                manifest_path=manifest_path,
            )
            summaries.append(
                {
                    "spec": spec,
                    "run_id": run_id,
                    "counts": counts,
                    "score": _coverage_score(counts=counts),
                }
            )
        summaries.sort(
            key=lambda item: (
                -item["score"],
                str(item["run_id"]),
            )
        )
        return summaries

    def _count_manifest_kinds(self, *, manifest_path: Path) -> dict[str, int]:
        """Count the exact valid, trainable records yielded to curation."""
        counts: dict[str, int] = {}
        for record in self._iter_manifest_records(manifest_path=manifest_path):
            kind = record.kind.strip().lower()
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def _raise_for_missing_requested_runs(
        self,
        *,
        found_final: frozenset[str],
    ) -> None:
        missing = sorted(self._requested_run_ids - found_final)
        if missing:
            raise ValueError(
                f"requested raw run IDs are missing or not final: {missing}"
            )

    def _coverage_minimum_records(self) -> dict[str, int]:
        return {
            kind: minimum
            for kind, minimum in sorted(self._minimum_modality_counts.items())
            if minimum > 0
        }

    @staticmethod
    def _summaries_to_specs(
        *,
        summaries: list[CoverageSummary],
    ) -> tuple[tuple[Path, str], ...]:
        return tuple(summary["spec"] for summary in summaries)

    def _is_final_run_spec(
        self,
        *,
        spec: tuple[Path, str],
    ) -> bool:
        manifest_path, manifest_name = spec
        run_directory = self._resolve_run_directory(
            manifest_path=manifest_path,
            manifest_name=manifest_name,
        )
        summary_path = (
            run_directory / Path(manifest_name).parent / "run_manifest.json"
        )
        if not summary_path.is_file():
            return False

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(payload, dict):
            return False

        return (
            str(payload.get("status") or "").strip() == "completed"
            and payload.get("final") is True
        )


@dataclass(frozen=True, slots=True)
class SelectedCrawlEvidence:
    """Aggregated acquisition-health evidence for the selected raw runs."""

    object_records_total: int
    successful_requests_total: int
    quality_score: float


def _read_run_output_readiness(
    *,
    summary_path: Path,
) -> SelectedCrawlEvidence | None:
    """Read the readiness evidence block from one run summary."""

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    readiness = payload.get("output_readiness")
    if not isinstance(readiness, dict):
        return None

    object_records_total = readiness.get("object_records_total")
    successful_requests_total = readiness.get("successful_requests_total")
    quality_score = readiness.get("quality_score")
    if isinstance(object_records_total, bool) or not isinstance(
        object_records_total, int
    ):
        return None
    if isinstance(successful_requests_total, bool) or not isinstance(
        successful_requests_total, int
    ):
        return None
    if isinstance(quality_score, bool) or not isinstance(
        quality_score, (int, float)
    ):
        return None
    if object_records_total < 0 or successful_requests_total < 0:
        return None
    return SelectedCrawlEvidence(
        object_records_total=object_records_total,
        successful_requests_total=successful_requests_total,
        quality_score=float(quality_score),
    )


@dataclass(frozen=True, slots=True)
class CoverageScoringProfile:
    kind_weights: dict[str, float]
    fallback_weight: float = 0.5
    max_count_per_kind: int = 25
    presence_bonus_multiplier: float = 50.0


class CoverageSummary(TypedDict):
    """Typed coverage evidence used while selecting raw runs."""

    spec: tuple[Path, str]
    run_id: str
    counts: dict[str, int]
    score: float


COVERAGE_SCORING_PROFILE = CoverageScoringProfile(
    kind_weights={
        "video": 10.0,
        "audio": 7.0,
        "document": 5.0,
        "image": 4.0,
        "page": 1.0,
        "feed": 1.0,
    }
)


def _coverage_score(*, counts: dict[str, int]) -> float:
    profile = COVERAGE_SCORING_PROFILE
    score = 0.0
    for kind, count in counts.items():
        weight = profile.kind_weights.get(kind, profile.fallback_weight)
        score += weight * min(profile.max_count_per_kind, max(0, count))
        if count > 0 and kind in profile.kind_weights:
            score += weight * profile.presence_bonus_multiplier
    return score


def _capped_coverage_counts(
    *,
    counts: dict[str, int],
    kinds: tuple[str, ...],
    caps: tuple[int, ...],
) -> tuple[int, ...]:
    """Cap a run's counts at the kind minima for combination search."""

    return tuple(
        min(max(0, int(counts.get(kind, 0) or 0)), cap)
        for kind, cap in zip(kinds, caps, strict=True)
    )


def _find_satisfying_combination(
    *,
    summaries: list[CoverageSummary],
    reduced: list[tuple[int, ...]],
    kinds: tuple[str, ...],
    caps: tuple[int, ...],
    max_runs: int,
) -> list[CoverageSummary] | None:
    """Return a minimal-size satisfying combination, or ``None``.

    Layered 0/1 knapsack: ``dp[k]`` maps an aggregated capped-count state
    to the best path reaching it with exactly ``k`` runs, where each run
    is used at most once. The first layer that reaches the capped minima
    state is returned (fewest runs); within a layer the path with the
    highest total coverage score wins.
    """

    zero = tuple(0 for _ in kinds)
    dp: list[dict[tuple[int, ...], tuple[float, tuple[int, ...]]]] = [
        {zero: (0.0, ())}
    ]
    for _ in range(1, max_runs + 1):
        dp.append({})

    for run_index, contribution in enumerate(reduced):
        score = summaries[run_index]["score"]
        for k in range(max_runs, 0, -1):
            source = dp[k - 1]
            target = dp[k]
            for state, (state_score, selected_indices) in tuple(
                source.items()
            ):
                added = tuple(
                    min(state[i] + contribution[i], caps[i])
                    for i in range(len(kinds))
                )
                candidate = state_score + score
                current = target.get(added)
                if current is None or candidate > current[0]:
                    target[added] = (
                        candidate,
                        selected_indices + (run_index,),
                    )

    goal = caps
    for k in range(1, max_runs + 1):
        if goal not in dp[k]:
            continue
        _score, selected_indices = dp[k][goal]
        return [summaries[index] for index in selected_indices]

    return None
