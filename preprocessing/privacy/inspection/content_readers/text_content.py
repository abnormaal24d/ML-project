"""Text payload supplied to privacy inspection."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    field_name: str = "text"
    language: str | None = None
    country: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
