"""Deterministic embedded video/player URL detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import match_extension


@dataclass(slots=True, frozen=True)
class VideoEmbedDetector:
    """Detect real embedded video/player URLs.

    This detector intentionally does not classify broad domains such as
    nasa.gov, science.nasa.gov, assets.science.nasa.gov, images-assets.nasa.gov
    or usgs.gov as video.

    Direct video files are not embed URLs. They must be classified by the
    modality-specific HTML reference extractors and the media-kind registry.
    """

    _YOUTUBE_HOSTS: ClassVar[frozenset[str]] = frozenset(
        {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }
    )

    _VIMEO_HOSTS: ClassVar[frozenset[str]] = frozenset(
        {
            "vimeo.com",
            "www.vimeo.com",
            "player.vimeo.com",
        }
    )
    _NASA_PLUS_HOSTS: ClassVar[frozenset[str]] = frozenset(
        {
            "plus.nasa.gov",
        }
    )

    def is_video_embed_url(self, url: str) -> bool:
        """Return whether the URL is a known embedded video/player URL."""

        parsed_url = self._split_url(url)
        if parsed_url is None:
            return False

        host, path, query = parsed_url

        if self._has_image_extension(path):
            return False

        return (
            self._is_youtube_embed(
                host=host,
                path=path,
                query=query,
            )
            or self._is_vimeo_embed(
                host=host,
                path=path,
                query=query,
            )
            or self._is_nasa_plus_embed(
                host=host,
                path=path,
                query=query,
            )
        )

    def video_embed_metadata(self, *, url: str) -> dict[str, object]:
        """
        Return metadata for embedded video records that should not be fetched.
        """

        return {
            "asset_fetch_mode": "embed_metadata",
            "embed_url": url,
            "trainable": False,
            "embed_host": self.embed_host(url=url),
        }

    def embed_host(self, *, url: str) -> str | None:
        """Return the normalized embed host."""

        parsed_url = self._split_url(url)
        if parsed_url is None:
            return None
        host, _, _ = parsed_url
        return host

    def _split_url(self, url: str) -> tuple[str, str, str] | None:
        if not url:
            return None

        try:
            parsed = urlsplit(url)
        except ValueError:
            return None

        host = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query

        return host, path, query

    def _has_image_extension(self, path: str) -> bool:
        kind = match_extension(f"https://extension.invalid{path}")
        return kind is MediaKind.IMAGE

    def _is_youtube_embed(
        self,
        *,
        host: str,
        path: str,
        query: str,
    ) -> bool:
        if host not in self._YOUTUBE_HOSTS:
            return False
        if path not in ("/embed", "/embed/", "/watch", "/v/"):
            return False
        if "v" not in parse_qs(query):
            return False
        return True

    def _is_vimeo_embed(
        self,
        *,
        host: str,
        path: str,
        query: str,
    ) -> bool:
        if host not in self._VIMEO_HOSTS:
            return False
        if not path.startswith("/"):
            return False
        if not path[1:].isdigit():
            return False
        return True

    def _is_nasa_plus_embed(
        self,
        *,
        host: str,
        path: str,
        query: str,
    ) -> bool:
        if host not in self._NASA_PLUS_HOSTS:
            return False
        if not path.startswith("/watch/"):
            return False
        return True


__all__ = ["VideoEmbedDetector"]
