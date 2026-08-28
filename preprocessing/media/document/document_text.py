"""Document text extraction multimodal for crawl acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DocumentText:
    """Accepted extracted text for one document payload."""

    text: str
    source: Literal["native", "ocr"]


class DocumentTextUnavailableError(RuntimeError):
    """Document yielded no usable text under the active extraction rules."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
