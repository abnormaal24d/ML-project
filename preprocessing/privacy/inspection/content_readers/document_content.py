"""Structured document content with page-level extraction evidence."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DocumentPage:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentContent:
    subject_bytes: bytes
    title: str | None = None
    pages: tuple[DocumentPage, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    language: str | None = None
    country: str | None = None
    expected_page_count: int | None = None
