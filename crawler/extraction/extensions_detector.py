"""URL extension based crawl-kind detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.media_kind_registry import (
    candidate_suffixes,
    match_extension,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import ExtensionDetectorSettings


class ExtensionDetector:
    """Infer crawl kinds from URL extensions."""

    def __init__(
        self,
        settings: ExtensionDetectorSettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._logger = logger

    def detect_kind(self, url: str) -> str | None:
        """Return a crawl kind inferred from the URL extension."""

        candidate_suffixes_list = self._candidate_suffixes(url)
        if not candidate_suffixes_list:
            return None

        for suffix in candidate_suffixes_list:
            kind = match_extension(
                f"https://extension.invalid/resource{suffix}"
            )
            if kind is not None:
                return kind.value

        if all(
            suffix in self._settings.ignored_unknown_suffixes
            for suffix in candidate_suffixes_list
        ):
            return None

        self._logger.debug(
            "extension_detector_unknown",
            url=url,
            suffixes=candidate_suffixes_list,
        )
        return None

    def _candidate_suffixes(self, url: str) -> tuple[str, ...]:
        """Return normalized candidate suffixes from most specific to least."""

        suffixes = list(candidate_suffixes(url=url))
        if not suffixes:
            return ()

        normalized_filename = str(url).rsplit("/", 1)[-1].lower()
        if self._looks_like_host_token(normalized_filename):
            return ()

        if ExtensionDetector._looks_like_numeric_resource_token(
            normalized_filename
        ):
            return ()

        if ExtensionDetector._looks_like_identifier_suffix_chain(suffixes):
            return ()

        if ExtensionDetector._looks_like_decimal_slug_fragment(suffixes):
            return ()

        return tuple(suffixes)

    def _looks_like_host_token(self, token: str) -> bool:
        parts = [part for part in token.split(".") if part]
        if len(parts) < 2:
            return False

        if parts[-1] not in self._settings.domain_like_tlds:
            return False

        return all(part.replace("-", "").isalnum() for part in parts)

    @staticmethod
    def _looks_like_numeric_resource_token(token: str) -> bool:
        parts = [part for part in token.split(".") if part]
        if len(parts) < 2:
            return False

        return all(part.isdigit() for part in parts[1:])

    @staticmethod
    def _looks_like_identifier_suffix_chain(suffixes: list[str]) -> bool:
        if len(suffixes) < 2:
            return False

        suffix_tokens = [suffix.lstrip(".") for suffix in suffixes]
        return all(
            token.isalnum() and any(char.isdigit() for char in token)
            for token in suffix_tokens
        )

    @staticmethod
    def _looks_like_decimal_slug_fragment(suffixes: list[str]) -> bool:
        if len(suffixes) != 1:
            return False

        token = suffixes[0].lstrip(".")
        if "_" not in token or not token:
            return False

        first = token[0]
        return first.isdigit()
