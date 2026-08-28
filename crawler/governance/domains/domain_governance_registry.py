"""Load domain governance by canonical host identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.governance.domains.domain_governance import DomainGovernance
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.source_catalog.catalog_settings import SourceRulesSettings
    from crawler.governance.domains.host_normalizer import HostNormalizer


class DomainGovernanceRegistry:
    """Resolve per-domain governance metadata from typed source settings."""

    def __init__(
        self,
        *,
        entries: tuple[SourceRulesSettings, ...],
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
    ) -> None:
        self._logger = logger
        self._host_normalizer = host_normalizer
        self._domain_governance_by_domain = self._load(entries=entries)

    def get(self, *, domain: str) -> DomainGovernance | None:
        """Return domain governance metadata for a domain, if configured."""

        normalized = self._host_normalizer.normalize(domain)
        if normalized is None:
            return None
        for candidate in self._iter_domain_candidates(normalized):
            governance = self._domain_governance_by_domain.get(candidate)
            if governance is not None:
                return governance
        return None

    def _load(
        self,
        *,
        entries: tuple[SourceRulesSettings, ...],
    ) -> dict[str, DomainGovernance]:
        domain_governance_by_domain: dict[str, DomainGovernance] = {}
        for entry in entries:
            domain_governance = self._to_domain_governance(entry)
            domain_key = self._host_normalizer.require(
                domain_governance.domain
            )
            domain_governance_by_domain[domain_key] = domain_governance

        self._logger.info(
            "source_governance_registry_loaded",
            extra={
                "domains": len(domain_governance_by_domain),
            },
        )
        return domain_governance_by_domain

    @staticmethod
    def _to_domain_governance(
        payload: SourceRulesSettings,
    ) -> DomainGovernance:
        # Map to DomainGovernance
        return DomainGovernance(
            domain=payload.domain,
            license=payload.license.expression if payload.license else None,
            license_url=payload.license.evidence_url
            if payload.license
            else None,
            allow_training=payload.training.allowed,
            governance_note=None,
            allow_collection=payload.collection.allowed,
            robots_status=None,
            terms_source=payload.license.evidence_kind
            if payload.license
            else None,
            usage_rules="allow_training"
            if payload.training.allowed
            else "deny_training",
            allow_boilerplate_image_caption=(
                payload.allow_boilerplate_image_caption
            ),
        )

    @staticmethod
    def _iter_domain_candidates(domain: str) -> tuple[str, ...]:
        parts = [part for part in domain.split(".") if part]
        candidates: list[str] = []
        for index in range(len(parts)):
            candidate = ".".join(parts[index:])
            if candidate:
                candidates.append(candidate)
        return tuple(candidates)
