"""Matrix regressions for the shared HTML asset-kind rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from crawler.classification.media_kind import MediaKind
from crawler.extraction.assets.html.html_asset_kind_rules import (
    infer_link_kind_from_attributes,
    normalize_rel_values,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MODALITY_EXTRACTORS = (
    "crawler/extraction/modalities/image_extractor.py",
    "crawler/extraction/modalities/document_extractor.py",
    "crawler/extraction/modalities/video_extractor.py",
    "crawler/extraction/modalities/audio_extractor.py",
)


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (None, set()),
        ("", set()),
        ("   ", set()),
        ("icon stylesheet", {"icon", "stylesheet"}),
        ("Icon", {"icon"}),
        ("apple-touch-icon preload", {"apple-touch-icon", "preload"}),
        (["icon", " STYLESHEET "], {"icon", "stylesheet"}),
        (("icon", "stylesheet"), {"icon", "stylesheet"}),
        ({"icon", "stylesheet"}, {"icon", "stylesheet"}),
        ([], set()),
        (("icon", "", "stylesheet"), {"icon", "stylesheet"}),
        ({"icon", None, "stylesheet"}, {"icon", "stylesheet"}),
        (42, set()),
        (["icon", 42], {"icon"}),
    ],
)
def test_normalize_rel_values(rel: object, expected: set[str]) -> None:
    assert normalize_rel_values(rel=rel) == expected


@pytest.mark.parametrize(
    (
        "rel",
        "media_type",
        "preload_as",
        "stylesheets_as_documents",
        "script_assets",
        "expected",
    ),
    [
        ({"icon"}, "image/png", None, True, False, MediaKind.IMAGE.value),
        ({"image_src"}, "image/png", None, True, False, MediaKind.IMAGE.value),
        ({"icon"}, None, None, True, False, MediaKind.IMAGE.value),
        (
            {"stylesheet"},
            "text/css",
            None,
            True,
            False,
            MediaKind.DOCUMENT.value,
        ),
        ({"stylesheet"}, "text/css", None, False, False, None),
        ({"preload"}, None, None, True, False, None),
        ({"preload"}, None, "video", True, False, MediaKind.VIDEO.value),
        ({"preload"}, "image/png", None, True, False, MediaKind.IMAGE.value),
        (
            {"alternate"},
            "application/rss+xml",
            None,
            True,
            False,
            MediaKind.FEED.value,
        ),
        ({"alternate"}, "video/mp4", None, True, False, MediaKind.VIDEO.value),
        (
            {"enclosure"},
            "audio/mpeg",
            None,
            True,
            False,
            MediaKind.AUDIO.value,
        ),
        ({"canonical"}, "image/png", None, True, False, None),
        ({"dns-prefetch"}, None, None, True, False, None),
        ({"nofollow"}, "image/png", None, True, False, MediaKind.IMAGE.value),
        ({"nofollow"}, "video/mp4", None, True, False, MediaKind.VIDEO.value),
        ({"nofollow"}, "application/octet-stream", None, True, False, None),
        ({"script"}, None, None, True, False, None),
        ({"script"}, "text/javascript", None, True, False, None),
        (
            {"script"},
            "text/javascript",
            None,
            True,
            True,
            MediaKind.DOCUMENT.value,
        ),
        (
            {"stylesheet"},
            "text/x-less",
            None,
            True,
            False,
            MediaKind.DOCUMENT.value,
        ),
        ({"stylesheet"}, "image/svg+xml", None, True, False, None),
        ({"preload"}, "font/woff2", None, True, False, None),
    ],
)
def test_infer_link_kind_matrix(
    rel: set[str],
    media_type: str | None,
    preload_as: str | None,
    stylesheets_as_documents: bool,
    script_assets: bool,
    expected: str | None,
) -> None:
    actual = infer_link_kind_from_attributes(
        rel_values=rel,
        media_type=media_type,
        preload_as=preload_as,
        include_icon_link_assets=True,
        include_stylesheets_as_documents=stylesheets_as_documents,
        include_script_assets=script_assets,
        include_font_assets=False,
    )
    assert actual == expected


def test_modality_extractors_share_normalize_rel_values() -> None:
    """The rel-normalizer must live in the rules module only."""
    for relative in MODALITY_EXTRACTORS:
        path = PROJECT_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        local_definitions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "_normalize_rel_values" not in local_definitions, (
            f"{relative} still defines a local _normalize_rel_values"
        )
        source = path.read_text(encoding="utf-8")
        assert "normalize_rel_values" in source, (
            f"{relative} no longer references the shared normalizer"
        )
