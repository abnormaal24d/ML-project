"""Document reference extraction from a typed page element index."""

from __future__ import annotations

from typing import Any

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import (
    match_extension,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidateDraft,
    build_asset_candidate_drafts,
    clean_string,
    parent_text_metadata,
)
from crawler.extraction.assets.embedded_media.inline_media_urls import (
    iter_inline_media_urls,
    looks_like_inline_media_config,
)
from crawler.extraction.assets.html.html_asset_element_reader import (
    build_element_asset_context,
    context_metadata,
)
from crawler.extraction.assets.html.html_asset_kind_rules import (
    infer_link_kind_from_attributes,
    infer_meta_kind_from_names,
    normalize_rel_values,
)
from crawler.extraction.assets.structured_data.jsonld_media_object_parser import (
    iter_jsonld_media_candidates,
    parse_jsonld_payload,
)
from crawler.extraction.html.html_parser import (
    element_attribute,
    element_raw_text,
    element_string_attribute,
    element_string_attributes,
)
from crawler.extraction.modalities.image_extractor import PageExtractionContext
from crawler.extraction.modalities.page_element_index import PageElementIndex

_TRACK_ATTRIBUTES = ("src",)
_OBJECT_ATTRIBUTES = ("data",)
_EMBED_ATTRIBUTES = ("src", "data-src")
_SOURCE_ATTRIBUTES = ("src", "data-src")


class DocumentReferenceExtractor:
    """Discover document references from a typed page element index."""

    def __init__(
        self,
        *,
        include_stylesheets_as_documents: bool = False,
        include_script_assets: bool = False,
        include_font_assets: bool = False,
        include_icon_link_assets: bool = False,
    ) -> None:
        self._include_stylesheets_as_documents = (
            include_stylesheets_as_documents
        )
        self._include_script_assets = include_script_assets
        self._include_font_assets = include_font_assets
        self._include_icon_link_assets = include_icon_link_assets

    def extract_references(
        self,
        *,
        index: PageElementIndex,
        context: PageExtractionContext,
    ) -> tuple[AssetCandidateDraft, ...]:
        parent_metadata = parent_text_metadata(
            parent_text=context.parent_text,
            parent_title=context.parent_title,
        )
        drafts: list[AssetCandidateDraft] = []

        for element in index.track_elements:
            drafts.extend(
                self._drafts_from_attributes(
                    element=element,
                    tag_name="track",
                    attributes=_TRACK_ATTRIBUTES,
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.object_elements:
            drafts.extend(
                self._drafts_from_attributes(
                    element=element,
                    tag_name="object",
                    attributes=_OBJECT_ATTRIBUTES,
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.embed_elements:
            drafts.extend(
                self._drafts_from_embed_like(
                    element=element,
                    tag_name="embed",
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.iframe_elements:
            drafts.extend(
                self._drafts_from_embed_like(
                    element=element,
                    tag_name="iframe",
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.standalone_source_elements:
            drafts.extend(
                self._drafts_from_document_source(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.metadata_elements:
            drafts.extend(
                self._drafts_from_meta_element(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.resource_link_elements:
            drafts.extend(
                self._drafts_from_link_element(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        for element in index.script_elements:
            drafts.extend(
                self._drafts_from_jsonld_script(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )
            drafts.extend(
                self._drafts_from_inline_script(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        return tuple(drafts)

    def _drafts_from_attributes(
        self,
        *,
        element: Any,
        tag_name: str,
        attributes: tuple[str, ...],
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        element_context = build_element_asset_context(
            element=element,
            tag_name=tag_name,
            parent_text_metadata=parent_text_metadata,
        )
        return build_asset_candidate_drafts(
            references=element_string_attributes(
                element=element,
                names=attributes,
            ),
            kind="document",
            source_tag=tag_name,
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_embed_like(
        self,
        *,
        element: Any,
        tag_name: str,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        references = tuple(
            (attribute, candidate)
            for attribute, candidate in element_string_attributes(
                element=element,
                names=_EMBED_ATTRIBUTES,
            )
            if match_extension(candidate) is MediaKind.DOCUMENT
        )
        if not references:
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name=tag_name,
            parent_text_metadata=parent_text_metadata,
        )
        return build_asset_candidate_drafts(
            references=references,
            kind="document",
            source_tag=tag_name,
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_document_source(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        media_type = element_string_attribute(element=element, name="type")
        typed_document = False

        if media_type is not None:
            lowered = media_type.casefold()
            if lowered.startswith(
                (
                    "application/pdf",
                    "application/msword",
                    "application/vnd.",
                    "text/csv",
                    "text/tab-separated-values",
                    "application/json",
                )
            ) or lowered in {"text/vtt", "application/ttml+xml"}:
                typed_document = True
            elif "/" in lowered and not lowered.startswith(
                ("image/", "audio/", "video/")
            ):
                typed_document = lowered.startswith(
                    "text/"
                ) or lowered.startswith("application/")
            else:
                return ()

        references = tuple(
            (attribute, candidate)
            for attribute, candidate in element_string_attributes(
                element=element,
                names=_SOURCE_ATTRIBUTES,
            )
            if typed_document
            or match_extension(candidate) is MediaKind.DOCUMENT
        )
        if not references:
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="source",
            parent_text_metadata=parent_text_metadata,
        )
        if typed_document and media_type is not None:
            element_context = {
                **element_context,
                "mime_hint": media_type,
            }

        return build_asset_candidate_drafts(
            references=references,
            kind="document",
            source_tag="source",
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_meta_element(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        candidates = {
            value.casefold()
            for key in ("property", "name")
            if (value := element_string_attribute(element=element, name=key))
            is not None
        }
        if infer_meta_kind_from_names(candidates) is not None:
            return ()

        candidate = element_string_attribute(element=element, name="content")
        if (
            candidate is None
            or match_extension(candidate) is not MediaKind.DOCUMENT
        ):
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="meta",
            parent_text_metadata=parent_text_metadata,
        )
        metadata = context_metadata(element_context)
        meta_key = clean_string(
            element_attribute(element=element, name="property")
        ) or clean_string(element_attribute(element=element, name="name"))
        if meta_key is not None:
            metadata["meta_name"] = meta_key

        return (
            AssetCandidateDraft(
                candidate=candidate,
                kind="document",
                source_attribute="content",
                source_tag="meta",
                context=element_context,
                metadata=metadata,
            ),
        )

    def _drafts_from_link_element(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        kind = infer_link_kind_from_attributes(
            rel_values=normalize_rel_values(
                rel=element_attribute(element=element, name="rel")
            ),
            media_type=element_attribute(element=element, name="type"),
            preload_as=element_attribute(element=element, name="as"),
            include_icon_link_assets=self._include_icon_link_assets,
            include_stylesheets_as_documents=(
                self._include_stylesheets_as_documents
            ),
            include_script_assets=self._include_script_assets,
            include_font_assets=self._include_font_assets,
        )
        if kind != "document":
            return ()

        candidate = element_string_attribute(element=element, name="href")
        if candidate is None:
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="link",
            parent_text_metadata=parent_text_metadata,
        )
        return build_asset_candidate_drafts(
            references=(("href", candidate),),
            kind="document",
            source_tag="link",
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_jsonld_script(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        script_type = element_string_attribute(element=element, name="type")
        if (
            script_type is None
            or script_type.casefold() != "application/ld+json"
        ):
            return ()

        payload = parse_jsonld_payload(element_raw_text(element=element))
        if payload is None:
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="script",
            parent_text_metadata=parent_text_metadata,
        )
        drafts: list[AssetCandidateDraft] = []
        for url, kind, metadata in iter_jsonld_media_candidates(payload):
            if kind != "document":
                continue
            drafts.append(
                AssetCandidateDraft(
                    candidate=url,
                    kind="document",
                    source_attribute="jsonld",
                    source_tag="script",
                    context=element_context,
                    metadata={
                        **context_metadata(element_context),
                        **dict(metadata),
                    },
                )
            )
        return tuple(drafts)

    def _drafts_from_inline_script(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        script_type = element_string_attribute(element=element, name="type")
        if (
            script_type is not None
            and script_type.casefold() == "application/ld+json"
        ):
            return ()

        script_text = element_raw_text(element=element)
        if not looks_like_inline_media_config(script_text):
            return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="script",
            parent_text_metadata=parent_text_metadata,
        )
        drafts: list[AssetCandidateDraft] = []
        for url in iter_inline_media_urls(script_text):
            if match_extension(url) is not MediaKind.DOCUMENT:
                continue
            drafts.append(
                AssetCandidateDraft(
                    candidate=url,
                    kind="document",
                    source_attribute="inline_json",
                    source_tag="script",
                    context=element_context,
                    metadata={
                        **context_metadata(element_context),
                        "asset_discovery_stage": "inline_player_config",
                        "discovery_reason": "inline_player_config",
                    },
                )
            )
        return tuple(drafts)
