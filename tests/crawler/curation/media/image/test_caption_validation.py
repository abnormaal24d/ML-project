"""Unit tests for curated caption validation."""

from __future__ import annotations

from crawler.curation.media.image.caption_validation import (
    has_caption_garbage,
    is_boilerplate_caption,
)


def test_missing_caption_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption(None)
    assert not is_boilerplate_caption("")
    assert not is_boilerplate_caption("   ")


def test_missing_caption_has_no_garbage() -> None:
    assert not has_caption_garbage(None)
    assert not has_caption_garbage("")


def test_semantic_use_of_next_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption("The next generation of electric cars")


def test_semantic_use_of_search_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption("Researchers search for water on Mars")


def test_semantic_use_of_share_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption("They share a common ancestor")


def test_ordinary_caption_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption(
        "A bronze statue of a lion stands in the city park at dawn"
    )


def test_short_ui_caption_is_boilerplate() -> None:
    assert is_boilerplate_caption("search results")
    assert is_boilerplate_caption("next")
    assert is_boilerplate_caption("previous")
    assert is_boilerplate_caption("gallery grid")
    assert is_boilerplate_caption("view list")
    assert is_boilerplate_caption("sort by")
    assert is_boilerplate_caption("menu")


def test_punctuated_ui_caption_is_boilerplate() -> None:
    assert is_boilerplate_caption("next.")
    assert is_boilerplate_caption("Next >")
    assert is_boilerplate_caption("menu:")
    assert is_boilerplate_caption("search, results")


def test_hyphenated_semantic_word_is_not_ui_token() -> None:
    assert not is_boilerplate_caption("A next-generation electric vehicle")


def test_multiword_boilerplate_phrases_are_boilerplate() -> None:
    assert is_boilerplate_caption("Click here for more photos")
    assert is_boilerplate_caption("Read more about this story")
    assert is_boilerplate_caption("Skip to main content")


def test_photo_credit_caption_is_boilerplate() -> None:
    assert is_boilerplate_caption("Photo credit: Jane Doe")
    assert is_boilerplate_caption("Image credit: John Smith")
    assert is_boilerplate_caption("credit: stock photo")


def test_long_photo_credit_attribution_is_not_boilerplate() -> None:
    assert not is_boilerplate_caption(
        "Photo credit: A detailed caption describing the historical context "
        "of the photograph and the people depicted in it"
    )


def test_html_markup_is_garbage() -> None:
    assert has_caption_garbage('<meta name="description" content="x">')
    assert has_caption_garbage('<link rel="stylesheet" href="/style.css">')
    assert has_caption_garbage("<script>alert('x')</script>")


def test_html_free_caption_has_no_garbage() -> None:
    assert not has_caption_garbage("A portrait of a woman in a red dress")


def test_garbage_marker_in_ratio_terms() -> None:
    assert has_caption_garbage("plain text <broken> tag")
