"""Public models for config.collection.settings.

Exports: CollectionSettings.
"""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel
from config.collection.autoscaling import AutoscalerSettings
from config.collection.caching import CollectionCacheSettings, MetricsSettings
from config.collection.content_settings import ContentProcessorSettings
from config.collection.discovery import (
    ExtensionDetectorSettings,
    HtmlParserSettings,
    SchedulingSettings,
    UrlPrioritySettings,
    WorkerPoolSettings,
)
from config.collection.extraction import (
    AssetExtractorSettings,
    HostExtractorSettings,
    LinkExtractorSettings,
    UrlExtractorSettings,
    UrlNormalizerSettings,
)
from config.collection.fetching import (
    FetcherSettings,
    ResponseBodyReaderSettings,
    UrlSchemeValidatorSettings,
)
from config.collection.governance import (
    BlacklistManagerSettings,
    RobotsSettings,
    UrlFilterSettings,
)
from config.collection.http_rules import HttpRulesSettings
from config.collection.identity import IdentitySettings
from config.collection.modality_acceptance import (
    ModalityAcceptanceSettingsCatalog,
)
from config.collection.pacing import PacingSettings
from config.collection.processors import ProcessorSettings
from config.collection.training_input_gate import DataCheckerSettings


class CollectionSettings(SettingsModel):
    cache: CollectionCacheSettings = Field(
        default_factory=CollectionCacheSettings
    )
    modality_acceptance: ModalityAcceptanceSettingsCatalog = Field(
        default_factory=ModalityAcceptanceSettingsCatalog
    )
    http_rules: HttpRulesSettings = Field(default_factory=HttpRulesSettings)
    fetcher: FetcherSettings = Field(default_factory=FetcherSettings)
    response_body_reader: ResponseBodyReaderSettings = Field(
        default_factory=ResponseBodyReaderSettings
    )
    pacing: PacingSettings = Field(default_factory=PacingSettings)
    url_scheme_validator: UrlSchemeValidatorSettings = Field(
        default_factory=UrlSchemeValidatorSettings
    )
    url_normalizer: UrlNormalizerSettings = Field(
        default_factory=UrlNormalizerSettings
    )
    link_extractor: LinkExtractorSettings = Field(
        default_factory=LinkExtractorSettings
    )
    asset_extractor: AssetExtractorSettings = Field(
        default_factory=AssetExtractorSettings
    )
    url_extractor: UrlExtractorSettings = Field(
        default_factory=UrlExtractorSettings
    )
    host_extractor: HostExtractorSettings = Field(
        default_factory=HostExtractorSettings
    )
    extension_detector: ExtensionDetectorSettings = Field(
        default_factory=ExtensionDetectorSettings
    )
    url_filter: UrlFilterSettings = Field(default_factory=UrlFilterSettings)
    html_parser: HtmlParserSettings = Field(default_factory=HtmlParserSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    robots: RobotsSettings = Field(default_factory=RobotsSettings)
    scheduling: SchedulingSettings = Field(default_factory=SchedulingSettings)
    url_priority_calculator: UrlPrioritySettings = Field(
        default_factory=UrlPrioritySettings
    )
    blacklist_manager: BlacklistManagerSettings = Field(
        default_factory=BlacklistManagerSettings
    )

    content_processor: ContentProcessorSettings = Field(
        default_factory=ContentProcessorSettings
    )
    datachecker: DataCheckerSettings = Field(
        default_factory=DataCheckerSettings
    )
    processors: ProcessorSettings = Field(default_factory=ProcessorSettings)
    worker_pool: WorkerPoolSettings = Field(default_factory=WorkerPoolSettings)
    autoscaler: AutoscalerSettings = Field(default_factory=AutoscalerSettings)
