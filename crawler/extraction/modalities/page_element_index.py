"""Typed page DOM index built with one structural traversal.

The index is a structural input for page metadata and modality reference
extractors. It classifies elements and assigns media descendants to one
nearest structural owner. It does not create candidates, resolve URLs, or
rank preferences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crawler.extraction.html.html_parser import (
    element_has_attribute,
    element_parent,
    element_tag_name,
)

_MEDIA_CONTAINER_TAGS = frozenset({"picture", "audio", "video"})


@dataclass(frozen=True, slots=True)
class IndexedMediaContainer:
    """One media container and the descendants structurally owned by it."""

    element: Any
    owned_elements: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class PageElementIndex:
    """Explicit structural buckets collected from one parsed document."""

    title_elements: tuple[Any, ...]
    metadata_elements: tuple[Any, ...]
    resource_link_elements: tuple[Any, ...]
    link_elements: tuple[Any, ...]

    image_elements: tuple[Any, ...]
    picture_containers: tuple[IndexedMediaContainer, ...]
    audio_containers: tuple[IndexedMediaContainer, ...]
    video_containers: tuple[IndexedMediaContainer, ...]
    standalone_source_elements: tuple[Any, ...]

    iframe_elements: tuple[Any, ...]
    object_elements: tuple[Any, ...]
    embed_elements: tuple[Any, ...]
    track_elements: tuple[Any, ...]
    script_elements: tuple[Any, ...]
    styled_elements: tuple[Any, ...]


class PageElementIndexBuilder:
    """Build a :class:`PageElementIndex` with one document traversal."""

    def build(self, *, document: Any) -> PageElementIndex:
        """Classify tags and assign media descendants to nearest owners."""

        title_elements: list[Any] = []
        metadata_elements: list[Any] = []
        resource_link_elements: list[Any] = []
        link_elements: list[Any] = []

        image_elements: list[Any] = []
        picture_container_elements: list[Any] = []
        picture_members: dict[int, list[Any]] = {}
        audio_container_elements: list[Any] = []
        audio_members: dict[int, list[Any]] = {}
        video_container_elements: list[Any] = []
        video_members: dict[int, list[Any]] = {}
        standalone_source_elements: list[Any] = []

        iframe_elements: list[Any] = []
        object_elements: list[Any] = []
        embed_elements: list[Any] = []
        track_elements: list[Any] = []
        script_elements: list[Any] = []
        styled_elements: list[Any] = []

        descendants = getattr(document, "descendants", None)
        iterable = descendants if descendants is not None else ()

        for element in iterable:
            tag_name = element_tag_name(element=element)
            if not tag_name:
                continue

            if tag_name == "title":
                title_elements.append(element)
            elif tag_name == "meta":
                metadata_elements.append(element)
            elif tag_name == "link":
                resource_link_elements.append(element)
            elif tag_name in {"a", "area"}:
                link_elements.append(element)
            elif tag_name == "picture":
                picture_container_elements.append(element)
                picture_members[id(element)] = []
            elif tag_name == "audio":
                audio_container_elements.append(element)
                audio_members[id(element)] = []
            elif tag_name == "video":
                video_container_elements.append(element)
                video_members[id(element)] = []
            elif tag_name == "img":
                owner = _nearest_media_container(element=element)
                if owner is not None and owner[0] == "picture":
                    picture_members.setdefault(id(owner[1]), []).append(
                        element
                    )
                else:
                    image_elements.append(element)
            elif tag_name == "source":
                owner = _nearest_media_container(element=element)
                if owner is None:
                    standalone_source_elements.append(element)
                else:
                    owner_tag, owner_element = owner
                    members_by_tag = {
                        "picture": picture_members,
                        "audio": audio_members,
                        "video": video_members,
                    }
                    members_by_tag[owner_tag].setdefault(
                        id(owner_element), []
                    ).append(element)
            elif tag_name == "iframe":
                iframe_elements.append(element)
            elif tag_name == "object":
                object_elements.append(element)
            elif tag_name == "embed":
                embed_elements.append(element)
            elif tag_name == "track":
                track_elements.append(element)
            elif tag_name == "script":
                script_elements.append(element)

            if element_has_attribute(element=element, name="style"):
                styled_elements.append(element)

        return PageElementIndex(
            title_elements=tuple(title_elements),
            metadata_elements=tuple(metadata_elements),
            resource_link_elements=tuple(resource_link_elements),
            link_elements=tuple(link_elements),
            image_elements=tuple(image_elements),
            picture_containers=_freeze_containers(
                elements=picture_container_elements,
                members=picture_members,
            ),
            audio_containers=_freeze_containers(
                elements=audio_container_elements,
                members=audio_members,
            ),
            video_containers=_freeze_containers(
                elements=video_container_elements,
                members=video_members,
            ),
            standalone_source_elements=tuple(standalone_source_elements),
            iframe_elements=tuple(iframe_elements),
            object_elements=tuple(object_elements),
            embed_elements=tuple(embed_elements),
            track_elements=tuple(track_elements),
            script_elements=tuple(script_elements),
            styled_elements=tuple(styled_elements),
        )


def _nearest_media_container(*, element: Any) -> tuple[str, Any] | None:
    current = element_parent(element=element)
    visited: set[int] = set()

    while current is not None:
        identity = id(current)
        if identity in visited:
            return None
        visited.add(identity)

        tag_name = element_tag_name(element=current)
        if tag_name in _MEDIA_CONTAINER_TAGS:
            return tag_name, current

        current = element_parent(element=current)

    return None


def _freeze_containers(
    *,
    elements: list[Any],
    members: dict[int, list[Any]],
) -> tuple[IndexedMediaContainer, ...]:
    return tuple(
        IndexedMediaContainer(
            element=element,
            owned_elements=tuple(members.get(id(element), ())),
        )
        for element in elements
    )
