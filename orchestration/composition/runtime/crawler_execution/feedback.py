"""Execution feedback composition.

Builds host/task feedback and robots-scheduler late binding.
"""

from __future__ import annotations

from crawler.extraction.hosts_extractor import HostExtractor
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from crawler.governance.robots.robots_request_gate import RobotsRequestGate
from crawler.runtime.feedback.crawler_task_feedback import CrawlerTaskFeedback
from crawler.scheduling.url_scheduler import UrlScheduler
from logger.factory import ProjectLoggerFactory


def build_execution_feedback(
    *,
    logger_factory: ProjectLoggerFactory,
    scheduler: UrlScheduler,
    robots_request_gate: RobotsRequestGate,
    host_budget_tracker: object,
    host_media_byte_budget: object,
    host_normalizer: HostNormalizer,
    rate_limiter: RateLimiter,
    host_extractor: HostExtractor,
) -> CrawlerTaskFeedback:
    """Build feedback wiring and robots-scheduler late binding."""
    from crawler.runtime.feedback.crawler_host_feedback import CrawlerHostFeedback

    host_feedback = CrawlerHostFeedback(
        logger=logger_factory.get_logger_for(CrawlerHostFeedback),
        register_host_advice=scheduler.register_host_rules_advice,
        rate_limiter=rate_limiter,
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
    )
    robots_request_gate.set_scheduler_advice_registrar(
        scheduler.register_host_rules_advice
    )
    return CrawlerTaskFeedback(
        scheduler=scheduler,
        host_feedback=host_feedback,
        host_budget_tracker=host_budget_tracker,
        host_media_byte_budget=host_media_byte_budget,
        host_normalizer=host_normalizer,
    )
