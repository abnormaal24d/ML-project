"""Structural URL canonicalization and optional equivalence rules."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import (
    SplitResult,
    parse_qsl,
    quote_from_bytes,
    unquote_to_bytes,
    urlencode,
    urlsplit,
    urlunsplit,
)

from crawler.governance.domains.host_normalizer import HostNormalizer
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.extraction import UrlNormalizerSettings


class UrlNormalizer:
    """Canonicalize URL structure and optionally collapse URL equivalence."""

    _TRACKING_QUERY_PARAMETERS = frozenset(
        {
            "fbclid",
            "gclid",
            "mc_cid",
            "mc_eid",
            "msclkid",
            "utm_campaign",
            "utm_content",
            "utm_medium",
            "utm_source",
            "utm_term",
        }
    )

    _MEDIA_VARIANT_EXTENSIONS = frozenset(
        {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".avif",
            ".bmp",
            ".tif",
            ".tiff",
            ".svg",
            ".mp4",
            ".webm",
            ".mov",
            ".m4v",
        }
    )

    _SIGNED_QUERY_PARAMETERS = frozenset(
        {
            "x-amz-signature",
            "x-amz-credential",
            "x-amz-security-token",
            "x-amz-signedheaders",
            "x-goog-signature",
            "x-goog-credential",
            "signature",
            "sig",
            "token",
            "expires",
            "expires_in",
            "rules",
            "key-pair-id",
        }
    )

    def __init__(
        self,
        *,
        settings: UrlNormalizerSettings,
        logger: ProjectLogger,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._host_normalizer = host_normalizer

    def normalize(self, url: object) -> str:
        """Return a normalized URL string."""

        if not isinstance(url, str):
            self._logger.warning(
                "url_normalization_invalid_input",
                input_type=type(url).__name__,
            )
            return ""

        raw_url = url.strip()
        if not raw_url:
            return ""
        try:
            parsed = urlsplit(raw_url)
            normalized_scheme = parsed.scheme.lower()
            netloc = self._canonical_netloc(
                parsed=parsed,
                raw_url=raw_url,
            )
        except (UnicodeError, ValueError) as exc:
            self._logger.warning(
                "url_normalization_failed",
                original=raw_url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return ""

        if (
            self._settings.enabled
            and self._settings.upgrade_http_to_https
            and normalized_scheme == "http"
        ):
            normalized_scheme = "https"

        netloc = self._apply_optional_port_rules(
            netloc=netloc,
            parsed=parsed,
            scheme=normalized_scheme,
        )
        fragment = (
            ""
            if self._settings.enabled and self._settings.strip_fragments
            else parsed.fragment
        )
        path = self._normalize_path(parsed.path or "/")
        if (
            self._settings.enabled
            and self._settings.remove_trailing_slash
            and path != "/"
        ):
            path = path.rstrip("/") or "/"

        query = (
            self._normalize_query(path=path, query=parsed.query)
            if self._settings.enabled
            else parsed.query
        )

        normalized = urlunsplit(
            (
                normalized_scheme,
                netloc,
                path,
                query,
                fragment,
            )
        )

        self._logger.debug(
            "url_normalized",
            original=url,
            normalized=normalized,
            changed=normalized != raw_url,
        )
        return normalized

    def _normalize_query(self, *, path: str, query: str) -> str:
        """Normalize query ordering and removable noise parameters."""

        if not query:
            return query

        pairs = parse_qsl(
            query,
            keep_blank_values=True,
        )
        if self._looks_like_signed_query(pairs=pairs):
            return query

        removable_names: set[str] = set()

        if self._settings.remove_tracking_parameters:
            removable_names.update(self._TRACKING_QUERY_PARAMETERS)

        if (
            self._settings.strip_known_media_variant_query_params
            and self._looks_like_static_media_asset(path)
        ):
            removable_names.update(
                self._settings.media_variant_query_param_names
            )

        changed = False
        if removable_names:
            retained_pairs = [
                (name, value)
                for name, value in pairs
                if name.strip().lower() not in removable_names
            ]
            changed = retained_pairs != pairs
            pairs = retained_pairs

        if self._settings.sort_query_parameters and self._can_sort_query(
            pairs=pairs
        ):
            sorted_pairs = sorted(pairs)
            changed = changed or sorted_pairs != pairs
            pairs = sorted_pairs

        return urlencode(pairs, doseq=True) if changed else query

    @classmethod
    def _looks_like_signed_query(
        cls,
        *,
        pairs: list[tuple[str, str]],
    ) -> bool:
        """Return whether query order/content likely carries a signature."""
        for name, _value in pairs:
            normalized = name.strip().lower()
            if normalized in cls._SIGNED_QUERY_PARAMETERS:
                return True
            if normalized.startswith(("x-amz-", "x-goog-", "awsaccesskeyid")):
                return True
        return False

    @staticmethod
    def _can_sort_query(*, pairs: list[tuple[str, str]]) -> bool:
        names = [name for name, _value in pairs]
        return len(names) == len(set(names))

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Remove dot segments/slash ambiguity and safely encode path bytes."""

        decoded = unquote_to_bytes(path or "/")
        absolute = decoded.startswith(b"/")
        final_segment = decoded.rsplit(b"/", maxsplit=1)[-1]
        keep_trailing_slash = decoded.endswith(b"/") or final_segment in {
            b".",
            b"..",
        }
        segments: list[bytes] = []

        for segment in decoded.split(b"/"):
            if segment in {b"", b"."}:
                continue
            if segment == b"..":
                if segments:
                    segments.pop()
                continue
            segments.append(segment)

        collapsed = b"/".join(segments)
        if absolute:
            collapsed = b"/" + collapsed
        if not collapsed:
            collapsed = b"/"
        elif keep_trailing_slash and collapsed != b"/":
            collapsed += b"/"

        return quote_from_bytes(
            collapsed,
            safe="/:@-._~!$&'()*+,;=",
        )

    @staticmethod
    def _userinfo(*, parsed: SplitResult) -> str:
        if parsed.username is None:
            return ""
        userinfo = quote_from_bytes(
            unquote_to_bytes(parsed.username),
            safe="-._~!$&'()*+,;=",
        )
        if parsed.password is not None:
            password = quote_from_bytes(
                unquote_to_bytes(parsed.password),
                safe="-._~!$&'()*+,;=",
            )
            return f"{userinfo}:{password}"
        return userinfo

    def _canonical_netloc(
        self,
        *,
        parsed: SplitResult,
        raw_url: str,
    ) -> str:
        """Return canonical authority syntax or raise for an invalid authority."""

        if not parsed.netloc:
            return ""

        hostname = parsed.hostname
        normalized_host = self._host_normalizer.normalize(hostname)
        if normalized_host is None:
            raise ValueError(f"URL contains an invalid host: {raw_url!r}")
        if ":" in normalized_host:
            normalized_host = f"[{normalized_host}]"

        userinfo = self._userinfo(parsed=parsed)
        netloc = (
            f"{userinfo}@{normalized_host}" if userinfo else normalized_host
        )
        port = parsed.port
        if port is not None:
            netloc = f"{netloc}:{port}"
        return netloc

    def _apply_optional_port_rules(
        self,
        *,
        netloc: str,
        parsed: SplitResult,
        scheme: str,
    ) -> str:
        """Remove an equivalent default port only when optional rules are on."""

        port = parsed.port
        original_scheme = parsed.scheme.lower()
        remove_default_port = (
            self._settings.enabled
            and (
                self._settings.remove_default_ports
                or self._settings.strip_default_ports
            )
            and (
                (scheme == "http" and port == 80)
                or (scheme == "https" and port == 443)
                # An explicit HTTP default port has no meaning after an
                # opted-in scheme upgrade.  Dropping it lets the upgraded
                # URL use HTTPS's default port rather than retaining :80.
                or (
                    original_scheme == "http"
                    and scheme == "https"
                    and port == 80
                )
            )
        )
        if not remove_default_port or port is None:
            return netloc

        suffix = f":{port}"
        return netloc.removesuffix(suffix)

    def _looks_like_static_media_asset(self, path: str) -> bool:
        """Return whether the path clearly points to a static media asset."""

        suffix = PurePosixPath((path or "").lower()).suffix
        return suffix in self._MEDIA_VARIANT_EXTENSIONS
