from __future__ import annotations

from config.collection.extraction import UrlNormalizerSettings
from crawler.extraction.candidates.url_candidate_resolution import (
    UrlCandidateResolution,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer
from tests.support.logging import TEST_LOGGER


def make_host_normalizer() -> HostNormalizer:
    return HostNormalizer()


def make_url_normalizer(
    *,
    settings: UrlNormalizerSettings | None = None,
    host_normalizer: HostNormalizer | None = None,
) -> UrlNormalizer:
    return UrlNormalizer(
        settings=settings or UrlNormalizerSettings(),
        logger=TEST_LOGGER,
        host_normalizer=host_normalizer or make_host_normalizer(),
    )


def make_url_candidate_resolution(
    *,
    url_normalizer: UrlNormalizer | None = None,
) -> UrlCandidateResolution:
    return UrlCandidateResolution(
        url_normalizer=url_normalizer or make_url_normalizer(),
        logger=TEST_LOGGER,
    )
