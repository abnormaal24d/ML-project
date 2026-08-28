"""Crawl task context models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CrawlTaskContext:
    """Compact discovery context carried with one crawl task."""

    tag_name: str | None = None
    source_tag: str | None = None
    source_attribute: str | None = None
    text_hint: str | None = None
    surrounding_text: str | None = None
    mime_hint: str | None = None
    alt_text: str | None = None
    caption_text: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    asset_quality_score: float | None = None
    asset_fetch_mode: str | None = None
    asset_rejection_reason: str | None = None
    embed_url: str | None = None
    embed_host: str | None = None
    poster_url: str | None = None
    parent_text_hash: str | None = None
    parent_text_preview: str | None = None
    parent_title: str | None = None
    html_dom_path: str | None = None
    asset_discovery_stage: str | None = None
    source_page_url: str | None = None
    discovery_reason: str | None = None
    candidate_strength: float | None = None
    selection_reason: str | None = None
    admission_reason: str | None = None
    media_identity: str | None = None
    source_page_depth: int | None = None
    is_boilerplate_asset: bool | None = None
    boilerplate_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly context payload."""
        payload: dict[str, object] = {}
        for key, value in (
            ("tag_name", self.tag_name),
            ("source_tag", self.source_tag),
            ("source_attribute", self.source_attribute),
            ("text_hint", self.text_hint),
            ("surrounding_text", self.surrounding_text),
            ("mime_hint", self.mime_hint),
            ("alt_text", self.alt_text),
            ("caption_text", self.caption_text),
            ("width", self.width),
            ("height", self.height),
            ("duration_seconds", self.duration_seconds),
            ("asset_quality_score", self.asset_quality_score),
            ("asset_fetch_mode", self.asset_fetch_mode),
            ("asset_rejection_reason", self.asset_rejection_reason),
            ("embed_url", self.embed_url),
            ("embed_host", self.embed_host),
            ("poster_url", self.poster_url),
            ("parent_text_hash", self.parent_text_hash),
            ("parent_text_preview", self.parent_text_preview),
            ("parent_title", self.parent_title),
            ("html_dom_path", self.html_dom_path),
            ("asset_discovery_stage", self.asset_discovery_stage),
            ("source_page_url", self.source_page_url),
            ("discovery_reason", self.discovery_reason),
            ("candidate_strength", self.candidate_strength),
            ("selection_reason", self.selection_reason),
            ("admission_reason", self.admission_reason),
            ("media_identity", self.media_identity),
            ("source_page_depth", self.source_page_depth),
            ("is_boilerplate_asset", self.is_boilerplate_asset),
            ("boilerplate_reason", self.boilerplate_reason),
        ):
            if value is not None:
                payload[key] = value
        return payload
