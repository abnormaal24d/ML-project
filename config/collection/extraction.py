"""Collection extraction configuration models."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import normalize_string_tuple


class AssetExtractorSettings(SettingsModel):
    """Configuration for asset extraction."""

    enabled: bool = True
    extract_images: bool = True
    extract_audio: bool = True
    extract_video: bool = True
    extract_documents: bool = True
    include_link_assets: bool = True
    include_icon_link_assets: bool = False
    include_stylesheets_as_documents: bool = False
    include_script_assets: bool = False
    include_font_assets: bool = False
    include_data_urls: bool = False
    max_asset_urls: int = Field(default=5_000, ge=1)


class HostExtractorSettings(SettingsModel):
    """Structural host-extraction admission settings."""

    enabled: bool = True
    allow_ip_hosts: bool = False
    allow_local_hosts: bool = False


class LinkExtractorSettings(SettingsModel):
    """Configuration for hyperlink extraction."""

    enabled: bool = True
    extract_anchor_links: bool = True
    extract_media_links: bool = True
    extract_canonical_links: bool = True
    extract_feed_links: bool = True
    include_nofollow_links: bool = False
    tags: tuple[str, ...] = ("a", "area")
    attribute: str = Field(default="href", min_length=1)
    max_links_per_page: int = Field(default=10_000, ge=1)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> tuple[str, ...]:
        """Normalize HTML tag names to lowercase values."""

        return normalize_string_tuple(value, lowercase=True)

    @model_validator(mode="after")
    def validate_link_extractor(self) -> LinkExtractorSettings:
        """Validate that link extraction has at least one tag."""

        if not self.tags:
            raise ValueError("tags must not be empty")

        return self


class UrlNormalizerSettings(SettingsModel):
    """Configuration for optional URL-equivalence normalization.

    Parsing, scheme/host canonicalization, path dot-segment removal, slash
    normalization, and safe path encoding are structural and always applied.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Enable optional URL-equivalence transforms; structural URL "
            "canonicalization is always applied."
        ),
    )
    strip_fragments: bool = True
    strip_default_ports: bool = True
    sort_query_parameters: bool = True
    remove_tracking_parameters: bool = True
    remove_trailing_slash: bool = True
    allowed_schemes: tuple[str, ...] = ("http", "https")
    upgrade_http_to_https: bool = False
    remove_default_ports: bool = True
    strip_known_media_variant_query_params: bool = True
    media_variant_query_param_names: tuple[str, ...] = (
        "width",
        "height",
        "w",
        "h",
        "format",
        "quality",
        "q",
    )

    @field_validator("allowed_schemes", mode="before")
    @classmethod
    def normalize_allowed_schemes(cls, value: object) -> tuple[str, ...]:
        """Normalize allowed URL schemes to lowercase."""

        return normalize_string_tuple(value, lowercase=True)

    @field_validator("media_variant_query_param_names", mode="before")
    @classmethod
    def normalize_media_variant_query_param_names(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        """Normalize media-variant query parameter names to lowercase."""

        return normalize_string_tuple(value, lowercase=True)

    @model_validator(mode="after")
    def validate_url_normalizer(self) -> UrlNormalizerSettings:
        """Validate URL normalizer scheme and media-variant settings."""

        if not self.allowed_schemes:
            raise ValueError("allowed_schemes must not be empty")

        if (
            self.strip_known_media_variant_query_params
            and not self.media_variant_query_param_names
        ):
            raise ValueError(
                "media_variant_query_param_names must not be empty when "
                "strip_known_media_variant_query_params is true"
            )

        return self


class UrlExtractorSettings(SettingsModel):
    """Configuration for URL extraction."""

    enabled: bool = True
    normalize_urls: bool = True
    deduplicate_urls: bool = True
    extract_from_html: bool = True
    extract_from_text: bool = True
    extract_from_metadata: bool = True
    include_links: bool = True
    include_assets: bool = True
    include_feeds: bool = True
    max_urls_per_document: int = Field(default=10_000, ge=1)
