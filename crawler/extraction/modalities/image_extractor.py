"""Image reference and (later) payload extraction for page discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidateDraft,
    build_asset_candidate_drafts,
    clean_string,
    parent_text_metadata,
)
from crawler.extraction.assets.html.html_asset_element_reader import (
    build_element_asset_context,
    context_metadata,
    srcset_references,
)
from crawler.extraction.assets.html.html_asset_kind_rules import (
    infer_link_kind_from_attributes,
    infer_meta_kind_from_names,
    normalize_rel_values,
)
from crawler.extraction.assets.html.inline_style_image_extractor import (
    extract_background_image_urls,
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
    element_tag_name,
)
from crawler.extraction.modalities.page_element_index import PageElementIndex

_IMG_ATTRIBUTES = (
    "src",
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-image",
    "data-url",
    "data-hi-res-src",
)
_SOURCE_IMAGE_ATTRIBUTES = ("src", "data-src")
_VIDEO_POSTER_ATTRIBUTES = ("poster",)


@dataclass(frozen=True, slots=True)
class PageExtractionContext:
    """Immutable page-level context for reference extraction."""

    page_url: str
    parent_text: str | None = None
    parent_title: str | None = None


class ImageReferenceExtractor:
    """Discover image references from a typed page element index."""

    def __init__(
        self,
        *,
        include_icon_link_assets: bool = False,
    ) -> None:
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

        for element in index.image_elements:
            drafts.extend(
                self._drafts_from_image_element(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        for container in index.picture_containers:
            for element in container.owned_elements:
                tag_name = element_tag_name(element=element)
                if tag_name == "img":
                    drafts.extend(
                        self._drafts_from_image_element(
                            element=element,
                            parent_text_metadata=parent_metadata,
                        )
                    )
                elif tag_name == "source":
                    drafts.extend(
                        self._drafts_from_source_element(
                            element=element,
                            parent_text_metadata=parent_metadata,
                        )
                    )

        for element in index.standalone_source_elements:
            drafts.extend(
                self._drafts_from_source_element(
                    element=element,
                    parent_text_metadata=parent_metadata,
                )
            )

        for container in index.video_containers:
            drafts.extend(
                self._drafts_from_video_poster(
                    element=container.element,
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

        for element in index.styled_elements:
            drafts.extend(
                self._drafts_from_styled_element(
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

        return tuple(drafts)

    def _drafts_from_image_element(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        element_context = build_element_asset_context(
            element=element,
            tag_name="img",
            parent_text_metadata=parent_text_metadata,
        )
        references = (
            *element_string_attributes(
                element=element,
                names=_IMG_ATTRIBUTES,
            ),
            *srcset_references(element=element),
        )
        return build_asset_candidate_drafts(
            references=references,
            kind="image",
            source_tag="img",
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_source_element(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        media_type = element_string_attribute(element=element, name="type")
        if media_type is not None:
            lowered = media_type.casefold()
            if not lowered.startswith("image/") and lowered != "image":
                if "/" in lowered:
                    return ()

        element_context = build_element_asset_context(
            element=element,
            tag_name="source",
            parent_text_metadata=parent_text_metadata,
        )
        references = (
            *element_string_attributes(
                element=element,
                names=_SOURCE_IMAGE_ATTRIBUTES,
            ),
            *srcset_references(element=element),
        )
        return build_asset_candidate_drafts(
            references=references,
            kind="image",
            source_tag="source",
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_video_poster(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        element_context = build_element_asset_context(
            element=element,
            tag_name="video",
            parent_text_metadata=parent_text_metadata,
        )
        return build_asset_candidate_drafts(
            references=element_string_attributes(
                element=element,
                names=_VIDEO_POSTER_ATTRIBUTES,
            ),
            kind="image",
            source_tag="video",
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
        if infer_meta_kind_from_names(candidates) != "image":
            return ()

        candidate = element_string_attribute(element=element, name="content")
        if candidate is None:
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
                kind="image",
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
            include_stylesheets_as_documents=False,
            include_script_assets=False,
            include_font_assets=False,
        )
        if kind != "image":
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
            kind="image",
            source_tag="link",
            context=element_context,
            metadata=context_metadata(element_context),
        )

    def _drafts_from_styled_element(
        self,
        *,
        element: Any,
        parent_text_metadata: dict[str, object],
    ) -> tuple[AssetCandidateDraft, ...]:
        urls = extract_background_image_urls(
            style=element_attribute(element=element, name="style")
        )
        if not urls:
            return ()

        tag_name = element_tag_name(element=element) or "div"
        element_context = build_element_asset_context(
            element=element,
            tag_name=tag_name,
            parent_text_metadata=parent_text_metadata,
        )
        return build_asset_candidate_drafts(
            references=tuple(("style", url) for url in urls),
            kind="image",
            source_tag=tag_name,
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
        if script_type is None or "ld+json" not in script_type.casefold():
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
            if kind != "image":
                continue
            drafts.append(
                AssetCandidateDraft(
                    candidate=url,
                    kind="image",
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
