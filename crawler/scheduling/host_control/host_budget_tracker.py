"""Manage per-host crawl budgets and quality feedback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

from .discovery_signal_scorer import (
    DiscoverySignalScorer,
    coerce_score,
    field_as_float,
    field_as_str,
)

if TYPE_CHECKING:
    from config.collection.discovery import SchedulingSettings
    from config.collection.governance import UrlFilterSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.domains.host_normalizer import HostNormalizer

    from .host_feedback_aggregator import HostFeedback, HostFeedbackAggregator


class HostBudgetTracker:
    """Manage per-host crawl budgets and quality feedback."""

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        url_filter_settings: UrlFilterSettings,
        logger: ProjectLogger,
        signal_scorer: DiscoverySignalScorer,
        feedback_aggregator: HostFeedbackAggregator,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
        seed_urls: tuple[str, ...],
    ) -> None:
        self._settings = settings
        self._url_filter_settings = url_filter_settings
        self._logger = logger
        self._signal_scorer = signal_scorer
        self._feedback_aggregator = feedback_aggregator
        self._host_normalizer = host_normalizer
        self._seed_hosts = self._extract_seed_hosts(
            host_extractor=host_extractor,
            host_normalizer=host_normalizer,
            seed_urls=seed_urls,
        )

    def register_task_outcome(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None = None,
    ) -> None:
        if not self._settings.allow_scheduler_feedback:
            return

        host = self._feedback_aggregator.host_for_url(task.url)
        if not host:
            return

        feedback = self._feedback_aggregator.get_or_create(host)
        feedback.observed_tasks += 1

        ignore_reason = self._ignored_feedback_reason(
            outcome=outcome,
            fields=fields,
        )
        if ignore_reason is not None:
            self._logger.debug(
                "host_feedback_ignored",
                url=task.url,
                host=host,
                outcome=outcome,
                reason=ignore_reason,
                budgeted=feedback.budgeted_tasks,
                observed=feedback.observed_tasks,
            )
            return

        feedback.budgeted_tasks += 1
        info_gain = self._signal_scorer.calculate_info_gain(
            outcome=outcome,
            fields=fields,
        )

        feedback.info_gain_ewma = self._signal_scorer.ewma(
            feedback.info_gain_ewma,
            info_gain,
        )

        if self._settings.record_host_quality:
            self._record_host_quality(
                task=task,
                outcome=outcome,
                fields=fields,
                host=host,
                feedback=feedback,
                info_gain=info_gain,
            )

        self._logger.debug(
            "host_feedback_recorded",
            url=task.url,
            host=host,
            outcome=outcome,
            budgeted=feedback.budgeted_tasks,
            observed=feedback.observed_tasks,
            info_gain=round(feedback.info_gain_ewma, 4),
            quality=round(feedback.quality_ewma, 4),
        )

    def host_budget_exhausted(self, host: str) -> bool:
        if not self._settings.dynamic_crawl_budget_enabled:
            return False

        feedback = self._feedback_aggregator.get(host)
        if feedback is None:
            return False

        if feedback.budgeted_tasks < self._settings.crawl_budget_window:
            return False

        return (
            feedback.info_gain_ewma
            < self._settings.crawl_budget_low_info_threshold
        )

    def host_quality(self, host: str) -> float:
        feedback = self._feedback_aggregator.get(host)
        if feedback is None:
            return self._settings.discovery_feedback.default_host_quality

        return coerce_score(
            feedback.quality_ewma,
            default=self._settings.discovery_feedback.default_host_quality,
        )

    def host_info_gain(self, host: str) -> float:
        feedback = self._feedback_aggregator.get(host)
        if feedback is None:
            return self._settings.discovery_feedback.default_info_gain

        return coerce_score(
            feedback.info_gain_ewma,
            default=self._settings.discovery_feedback.default_info_gain,
        )

    def is_seed_host(self, host: str) -> bool:
        normalized = self._host_normalizer.normalize(host)
        if normalized is None:
            return False
        if normalized in self._seed_hosts:
            return True
        if not self._url_filter_settings.allow_subdomains_of_seed_hosts:
            return False
        return any(
            normalized.endswith(f".{seed_host}")
            for seed_host in self._seed_hosts
        )

    def is_expanded_host(self, source_name: str, host: str) -> bool:
        return self._feedback_aggregator.is_expanded_host(source_name, host)

    def _record_host_quality(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
        host: str,
        feedback: HostFeedback,
        info_gain: float,
    ) -> None:
        topic_score = self._topic_score(
            url=task.url,
            category=field_as_str(fields, "category"),
        )

        relevance_value = field_as_float(fields, "relevance_score")
        if relevance_value is None:
            relevance_value = (
                self._signal_scorer.default_relevance_for_outcome(outcome)
            )

        relevance = coerce_score(relevance_value, default=0.25)
        blended_quality = self._signal_scorer.blend_quality(
            relevance=relevance,
            info_gain=info_gain,
            topic_score=topic_score,
        )

        feedback.quality_ewma = self._signal_scorer.ewma(
            feedback.quality_ewma,
            blended_quality,
        )

        if self._should_expand_host(
            task=task,
            host=host,
            feedback=feedback,
            topic_score=topic_score,
        ):
            self._feedback_aggregator.mark_expanded(
                task.source_name,
                host,
            )
            self._logger.debug(
                "host_feedback_host_expanded",
                host=host,
                quality_score=round(feedback.quality_ewma, 4),
                topic_score=round(topic_score, 4),
            )

    def _should_expand_host(
        self,
        *,
        task: CrawlTask,
        host: str,
        feedback: HostFeedback,
        topic_score: float,
    ) -> bool:
        return (
            self._url_filter_settings.intelligent_domain_expansion_enabled
            and not self.is_seed_host(host)
            and not self._feedback_aggregator.is_expanded_host(
                task.source_name,
                host,
            )
            and self._feedback_aggregator.expanded_host_count
            < self._url_filter_settings.max_expanded_hosts
            and feedback.quality_ewma
            >= self._url_filter_settings.expanded_host_min_quality
            and topic_score > 0.0
        )

    @staticmethod
    def _ignored_feedback_reason(
        *,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> str | None:
        normalized_outcome = (outcome or "").strip().lower()
        del fields

        if normalized_outcome in {"cancelled", "interrupted"}:
            return normalized_outcome

        return None

    def _topic_score(self, *, url: str, category: str | None) -> float:
        normalized_url = url.lower()
        normalized_category = (category or "").lower()
        keywords = self._url_filter_settings.expanded_host_topic_keywords

        if any(keyword in normalized_category for keyword in keywords):
            return 1.0

        if any(keyword in normalized_url for keyword in keywords):
            return 0.8

        return 0.0

    @staticmethod
    def _extract_seed_hosts(
        *,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
        seed_urls: tuple[str, ...],
    ) -> frozenset[str]:
        return frozenset(
            normalized
            for url in seed_urls
            if (
                normalized := host_normalizer.normalize(
                    host_extractor.extract(url),
                )
            )
            is not None
        )
