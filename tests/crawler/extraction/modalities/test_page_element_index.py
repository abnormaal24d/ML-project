"""Tests for parser-neutral page indexing and media ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from config.collection.discovery import HtmlParserSettings
from crawler.extraction.html.html_parser import (
    HtmlParser,
    element_attribute,
    element_raw_text,
    element_visible_text,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndex,
    PageElementIndexBuilder,
)


class _CountingDescendants:
    def __init__(self, elements: list[Any]) -> None:
        self._elements = elements
        self.iteration_count = 0

    @property
    def descendants(self) -> Any:
        self.iteration_count += 1
        return iter(self._elements)


class _Element:
    def __init__(
        self,
        name: str,
        *,
        parent: Any | None = None,
        attrs: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.attrs = attrs or {}

    def get(self, name: str, default: object = None) -> object:
        return self.attrs.get(name, default)

    def has_attr(self, name: str) -> bool:
        return name in self.attrs


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


def _html_parser(*, fallback: bool = False) -> HtmlParser:
    return HtmlParser(
        settings=HtmlParserSettings(
            parser="html.parser",
            parser_candidates=("html.parser",),
            prefer_beautiful_soup=not fallback,
            allow_stdlib_fallback=True,
        ),
        logger=_logger(),
    )


@pytest.fixture
def builder() -> PageElementIndexBuilder:
    return PageElementIndexBuilder()


def test_builder_classifies_tags_and_media_ownership(
    builder: PageElementIndexBuilder,
) -> None:
    picture = _Element("picture")
    picture_source = _Element("source", parent=picture)
    picture_image = _Element("img", parent=picture)
    audio = _Element("audio")
    audio_source = _Element("source", parent=audio)
    video = _Element("video")
    video_source = _Element("source", parent=video)
    standalone_source = _Element("source")
    standalone_image = _Element("IMG")

    document = _CountingDescendants(
        [
            "text-node",
            _Element("title"),
            _Element("meta"),
            _Element("link"),
            _Element("a"),
            _Element("area"),
            standalone_image,
            picture,
            picture_source,
            picture_image,
            audio,
            audio_source,
            video,
            video_source,
            standalone_source,
            _Element("iframe"),
            _Element("object"),
            _Element("embed"),
            _Element("track"),
            _Element("script"),
            _Element("div", attrs={"style": "background:url(x.png)"}),
        ]
    )

    index = builder.build(document=document)

    assert isinstance(index, PageElementIndex)
    assert len(index.title_elements) == 1
    assert len(index.metadata_elements) == 1
    assert len(index.resource_link_elements) == 1
    assert len(index.link_elements) == 2
    assert index.image_elements == (standalone_image,)
    assert index.picture_containers[0].element is picture
    assert index.picture_containers[0].owned_elements == (
        picture_source,
        picture_image,
    )
    assert index.audio_containers[0].owned_elements == (audio_source,)
    assert index.video_containers[0].owned_elements == (video_source,)
    assert index.standalone_source_elements == (standalone_source,)
    assert len(index.iframe_elements) == 1
    assert len(index.object_elements) == 1
    assert len(index.embed_elements) == 1
    assert len(index.track_elements) == 1
    assert len(index.script_elements) == 1
    assert len(index.styled_elements) == 1


def test_builder_traverses_descendants_exactly_once(
    builder: PageElementIndexBuilder,
) -> None:
    document = _CountingDescendants(
        [_Element("img"), _Element("a"), _Element("video")],
    )

    builder.build(document=document)

    assert document.iteration_count == 1


def test_nearest_media_container_wins_for_malformed_nesting(
    builder: PageElementIndexBuilder,
) -> None:
    video = _Element("video")
    audio = _Element("audio", parent=video)
    source = _Element("source", parent=audio)

    index = builder.build(
        document=_CountingDescendants([video, audio, source])
    )

    assert index.video_containers[0].owned_elements == ()
    assert index.audio_containers[0].owned_elements == (source,)
    assert index.standalone_source_elements == ()


def test_media_ownership_has_no_depth_limit(
    builder: PageElementIndexBuilder,
) -> None:
    picture = _Element("picture")
    parent: Any = picture
    descendants: list[Any] = [picture]
    for _ in range(12):
        parent = _Element("div", parent=parent)
        descendants.append(parent)
    source = _Element("source", parent=parent)
    descendants.append(source)

    index = builder.build(document=_CountingDescendants(descendants))

    assert index.picture_containers[0].owned_elements == (source,)
    assert index.standalone_source_elements == ()


def test_cyclic_parent_chain_terminates_safely(
    builder: PageElementIndexBuilder,
) -> None:
    first = _Element("div")
    second = _Element("section", parent=first)
    first.parent = second
    source = _Element("source", parent=first)

    index = builder.build(document=_CountingDescendants([source]))

    assert index.standalone_source_elements == (source,)


def test_each_media_element_is_indexed_exactly_once(
    builder: PageElementIndexBuilder,
) -> None:
    picture = _Element("picture")
    picture_source = _Element("source", parent=picture)
    picture_image = _Element("img", parent=picture)
    ordinary_image = _Element("img")
    standalone_source = _Element("source")

    index = builder.build(
        document=_CountingDescendants(
            [
                ordinary_image,
                picture,
                picture_source,
                picture_image,
                standalone_source,
            ]
        )
    )

    indexed = [
        *index.image_elements,
        *index.standalone_source_elements,
        *index.picture_containers[0].owned_elements,
    ]
    assert indexed == [
        ordinary_image,
        standalone_source,
        picture_source,
        picture_image,
    ]
    assert len({id(element) for element in indexed}) == len(indexed)


def test_dom_text_contracts_are_distinct() -> None:
    document = _html_parser().parse(
        body=b"""
        <html><body>
          <div id="visible"><span>Hello</span><b>world</b></div>
          <script type="application/ld+json">{"name":"raw"}</script>
        </body></html>
        """,
        encoding="utf-8",
    )
    visible = document.find("div", id="visible")
    script = document.find("script")

    assert element_visible_text(element=visible) == "Hello world"
    assert element_raw_text(element=script).strip() == '{"name":"raw"}'


def test_element_attribute_falls_back_after_broken_get() -> None:
    class BrokenGetter:
        attrs = {"href": "/fallback"}

        def get(self, name: str) -> object:
            raise TypeError(name)

    assert (
        element_attribute(element=BrokenGetter(), name="href") == "/fallback"
    )


def test_stdlib_fallback_supports_index_and_media_ownership(
    builder: PageElementIndexBuilder,
) -> None:
    document = _html_parser(fallback=True).parse(
        body=b"""
        <html><body>
          <picture><source srcset="/p.jpg"><img src="/fallback.jpg"></picture>
          <audio><source src="/a.ogg"></audio>
          <video><source src="/v.mp4"></video>
          <source src="/standalone.pdf" type="application/pdf">
        </body></html>
        """,
        encoding="utf-8",
    )

    index = builder.build(document=document)

    assert len(document.descendants) > 0
    assert index.picture_containers[0].owned_elements
    assert index.audio_containers[0].owned_elements
    assert index.video_containers[0].owned_elements
    assert len(index.standalone_source_elements) == 1
    picture_source = index.picture_containers[0].owned_elements[0]
    assert picture_source.parent is index.picture_containers[0].element
    assert element_attribute(element=picture_source, name="srcset") == "/p.jpg"


def test_builder_works_with_real_html_parser(
    builder: PageElementIndexBuilder,
) -> None:
    document = _html_parser().parse(
        body=b"""
        <html>
          <head>
            <title>Example</title>
            <meta name="robots" content="index"/>
            <link rel="canonical" href="https://example.test/page"/>
            <meta property="og:image" content="https://cdn.example.test/og.jpg"/>
          </head>
          <body>
            <a href="/next">next</a>
            <img src="/a.jpg" data-src="/lazy.jpg"/>
            <picture><source srcset="/p.jpg"/><img src="/fallback.jpg"/></picture>
            <audio src="/a.mp3"></audio>
            <video poster="/poster.jpg" src="/v.mp4"></video>
            <div style="background-image:url(/bg.png)">x</div>
            <script type="application/ld+json">{}</script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    index = builder.build(document=document)

    assert len(index.title_elements) == 1
    assert len(index.metadata_elements) >= 2
    assert len(index.resource_link_elements) == 1
    assert len(index.link_elements) == 1
    assert len(index.image_elements) == 1
    assert len(index.picture_containers) == 1
    assert len(index.picture_containers[0].owned_elements) == 2
    assert len(index.audio_containers) == 1
    assert len(index.video_containers) == 1
    assert len(index.standalone_source_elements) == 0
    assert len(index.script_elements) == 1
    assert len(index.styled_elements) >= 1
