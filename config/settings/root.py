"""Root settings for the profile-based configuration layer.

One Settings tree, one loader, one validator.  The runtime-facing shape is
flat at the multimodal boundary: model settings live at ``multimodal`` and
training settings live at ``training``.
"""

from __future__ import annotations

from pydantic import Field

from config.augmentation.augmentation_settings import AugmentationSettings
from config.base.settings_model import SettingsModel
from config.collection.settings import CollectionSettings
from config.coverage.settings import CoverageSettings
from config.media_toolchain import MediaToolchainSettings
from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_settings import TrainingSettings
from config.preprocessing.settings import PreprocessingSettings
from config.profiles import Profile
from config.settings.app import AppSettings
from config.settings.classification import ClassificationSettings
from config.settings.crawler import CrawlerSettings
from config.settings.datasets import DatasetSettings
from config.settings.gate import CrawlOutputGateSettings
from config.settings.logging import LoggingSettings
from config.settings.meta import ConfigMeta
from config.settings.paths import PathSettings
from config.settings.release import ReleaseSettings
from config.settings.sources import SourcesSettings


class Settings(SettingsModel):
    """Validated canonical runtime settings object."""

    profile: Profile
    meta: ConfigMeta | None = None

    application: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    sources: SourcesSettings = Field(default_factory=SourcesSettings)
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    media_toolchain: MediaToolchainSettings = Field(
        default_factory=MediaToolchainSettings
    )
    preprocessing: PreprocessingSettings = Field(
        default_factory=PreprocessingSettings
    )
    datasets: DatasetSettings = Field(default_factory=DatasetSettings)
    augmentation: AugmentationSettings = Field(
        default_factory=AugmentationSettings
    )
    coverage: CoverageSettings = Field(default_factory=CoverageSettings)
    crawler: CrawlerSettings = Field(default_factory=CrawlerSettings)
    crawl_output_gate: CrawlOutputGateSettings = Field(
        default_factory=CrawlOutputGateSettings
    )
    multimodal: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    release: ReleaseSettings = Field(default_factory=ReleaseSettings)
    classification: ClassificationSettings = Field(
        default_factory=ClassificationSettings
    )


Settings.model_rebuild()
