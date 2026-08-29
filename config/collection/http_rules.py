"""HTTP transport rules for redirects, retries, timeouts, and statuses."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from math import isfinite
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import normalize_status_code_tuple

if TYPE_CHECKING:
    from collections.abc import Collection


def _default_retry_class_delay_multipliers() -> dict[str, float]:
    return {
        "body_timeout": 1.25,
        "status_retry": 1.0,
        "transport_error": 1.0,
        "transport_timeout": 1.5,
    }


def _normalize_retry_delay_multipliers(value: object) -> dict[str, float]:
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError("retry delay multipliers must be a mapping")

    normalized: dict[str, float] = {}
    for raw_key, raw_multiplier in value.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue

        try:
            multiplier = float(raw_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "retry delay multipliers must contain numeric values"
            ) from exc

        if not isfinite(multiplier) or multiplier <= 0:
            raise ValueError(
                "retry delay multipliers must be positive finite numbers"
            )

        normalized[key] = multiplier

    return normalized


class RedirectRulesSettings(SettingsModel):
    max_redirects: int = Field(default=3, ge=0, le=10)
    robots_max_redirects: int = Field(default=5, ge=3, le=10)
    cross_host_mode: Literal[
        "deny",
        "source_allowlist",
    ] = "deny"
    block_https_downgrade: bool = True
    max_location_length: int = Field(
        default=4096,
        ge=256,
        le=16384,
    )


class NetworkAccessSettings(SettingsModel):
    """SSRF guardrails applied before requests and after DNS resolution."""

    enforce_on_dns_resolution: bool = True
    block_url_credentials: bool = True
    block_local_hostnames: bool = True
    blocked_hostname_suffixes: tuple[str, ...] = (
        "localhost",
        ".localhost",
        ".local",
        ".localdomain",
        "localhost6",
        "ip6-localhost",
        "ip6-loopback",
        "broadcasthost",
    )
    block_private_ip_ranges: bool = True
    block_loopback_ip_ranges: bool = True
    block_link_local_ip_ranges: bool = True
    block_unspecified_ip_ranges: bool = True
    block_multicast_ip_ranges: bool = True
    block_reserved_ip_ranges: bool = True
    block_site_local_ip_ranges: bool = True
    allowed_ip_literals: tuple[str, ...] = ()
    allowed_http_ports: tuple[int, ...] = (80,)
    allowed_https_ports: tuple[int, ...] = (443,)

    @field_validator("blocked_hostname_suffixes", mode="before")
    @classmethod
    def normalize_blocked_hostname_suffixes(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            candidates = (value,)
        elif isinstance(value, Iterable):
            candidates = tuple(value)
        else:
            raise ValueError("blocked hostname suffixes must be iterable")
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            suffix = str(candidate).strip().lower().rstrip(".")
            if not suffix:
                continue
            if suffix != "localhost" and not suffix.startswith("."):
                suffix = f".{suffix}"
            if suffix in seen:
                continue
            seen.add(suffix)
            normalized.append(suffix)
        return tuple(normalized)

    @field_validator("allowed_ip_literals", mode="before")
    @classmethod
    def normalize_allowed_ip_literals(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            candidates = (value,)
        elif isinstance(value, Iterable):
            candidates = tuple(value)
        else:
            raise ValueError("allowed IP literals must be iterable")
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                address = ipaddress.ip_address(str(candidate).strip("[]"))
            except ValueError as exc:
                raise ValueError(
                    "allowed_ip_literals must contain valid IP addresses"
                ) from exc
            canonical = str(getattr(address, "ipv4_mapped", None) or address)
            if canonical in seen:
                continue
            seen.add(canonical)
            normalized.append(canonical)
        return tuple(normalized)

    @field_validator(
        "allowed_http_ports",
        "allowed_https_ports",
        mode="before",
    )
    @classmethod
    def normalize_allowed_ports(
        cls,
        value: object,
    ) -> tuple[int, ...]:
        if value is None:
            return ()

        if isinstance(value, int):
            candidates = (value,)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            candidates = tuple(value)
        else:
            raise ValueError("allowed ports must be integers or an iterable")

        ports: list[int] = []
        seen: set[int] = set()

        for candidate in candidates:
            try:
                port = int(candidate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "allowed ports must contain valid integers"
                ) from exc

            if not 1 <= port <= 65535:
                raise ValueError("allowed ports must be between 1 and 65535")

            if port not in seen:
                seen.add(port)
                ports.append(port)

        return tuple(ports)


class RetryRulesSettings(SettingsModel):
    invalid_not_modified_retries: int = Field(default=1, ge=0, le=1)
    base_delay_seconds: float = Field(default=0.2, ge=0.0)
    max_delay_seconds: float = Field(default=5.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    jitter_strategy: str = Field(default="additive")
    retry_class_delay_multipliers: dict[str, float] = Field(
        default_factory=_default_retry_class_delay_multipliers,
    )
    retry_error_kind_delay_multipliers: dict[str, float] = Field(
        default_factory=dict,
    )

    @field_validator("jitter_strategy", mode="before")
    @classmethod
    def normalize_jitter_strategy(cls, value: object) -> str:
        strategy = str(value or "additive").strip().lower().replace("-", "_")
        if strategy not in {
            "additive",
            "decorrelated",
            "equal",
            "full",
            "none",
        }:
            raise ValueError(
                "jitter_strategy must be one of additive, decorrelated, "
                "equal, full, or none"
            )
        return strategy

    @field_validator(
        "retry_class_delay_multipliers",
        "retry_error_kind_delay_multipliers",
        mode="before",
    )
    @classmethod
    def normalize_retry_delay_multipliers(
        cls,
        value: object,
    ) -> dict[str, float]:
        return _normalize_retry_delay_multipliers(value)

    @model_validator(mode="after")
    def validate_retry_delays(self) -> RetryRulesSettings:
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be greater than or equal to "
                "base_delay_seconds"
            )
        return self


class HostCircuitBreakerSettings(SettingsModel):
    """Per-host failure threshold and recovery cooldown."""

    failure_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: float = Field(default=30.0, gt=0.0)


class HttpStatusRulesSettings(SettingsModel):
    """Canonical HTTP status-code rules across fetch, retry and feedback."""

    retryable: tuple[int, ...] = (
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    )
    accepted_non_success: tuple[int, ...] = (304,)

    head_preflight_hard_drop: tuple[int, ...] = (404, 410)
    head_preflight_method_not_supported: tuple[int, ...] = (405, 501)
    rate_limiter_feedback: tuple[int, ...] = (
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    )

    @field_validator(
        "retryable",
        "accepted_non_success",
        "head_preflight_hard_drop",
        "head_preflight_method_not_supported",
        "rate_limiter_feedback",
        mode="before",
    )
    @classmethod
    def normalize_status_codes(cls, value: object) -> tuple[int, ...]:
        return normalize_status_code_tuple(
            value,
            field_name="http status rules",
        )

    @model_validator(mode="after")
    def validate_feedback_statuses(self) -> HttpStatusRulesSettings:
        if not self.rate_limiter_feedback:
            raise ValueError("rate_limiter_feedback must not be empty")
        return self

    def is_head_method_not_supported(self, status_code: int) -> bool:
        """Return whether HEAD is unsupported but GET may still work."""

        return int(status_code) in self.head_preflight_method_not_supported


class TimeoutRulesSettings(SettingsModel):
    connect_timeout_seconds: float = Field(default=15.0, gt=0.0)
    request_timeout_seconds: float = Field(default=45.0, gt=0.0)
    large_media_request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    head_preflight_timeout_seconds: float = Field(default=8.0, gt=0.0)
    body_stream_timeout_seconds: float = Field(default=45.0, gt=0.0)
    media_body_stream_timeout_seconds: float = Field(default=120.0, gt=0.0)
    document_body_stream_timeout_seconds: float = Field(default=120.0, gt=0.0)
    first_byte_timeout_seconds: float = Field(default=10.0, gt=0.0)
    read_chunk_timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_idle_seconds: float = Field(default=15.0, gt=0.0)
    max_stream_seconds: float | None = Field(default=None, gt=0.0)
    robots_timeout_seconds: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def validate_timeouts(self) -> TimeoutRulesSettings:
        if (
            self.large_media_request_timeout_seconds
            < self.request_timeout_seconds
        ):
            raise ValueError(
                "large_media_request_timeout_seconds must be greater than or "
                "equal to request_timeout_seconds"
            )
        if self.head_preflight_timeout_seconds > self.request_timeout_seconds:
            raise ValueError(
                "head_preflight_timeout_seconds must be less than or equal to "
                "request_timeout_seconds"
            )
        return self

    def body_stream_timeout_for_content_type(
        self,
        *,
        content_type: str,
        document_content_types: Collection[str],
    ) -> float:
        """Resolve body-stream timeout seconds from normalized content type."""

        if content_type.startswith(("audio/", "video/")):
            return float(self.media_body_stream_timeout_seconds)
        if content_type in document_content_types:
            return float(self.document_body_stream_timeout_seconds)
        return float(self.body_stream_timeout_seconds)


class ConnectionPoolSettings(SettingsModel):
    max_connections: int = Field(default=100, ge=1)
    max_connections_per_host: int = Field(default=10, ge=1)
    media_max_connections: int = Field(default=1, ge=1)
    ttl_dns_cache_seconds: int = Field(default=300, ge=0)
    async_dns_enabled: bool = True
    auto_decompress: bool = True

    @model_validator(mode="after")
    def validate_connection_limits(self) -> ConnectionPoolSettings:
        if self.max_connections_per_host > self.max_connections:
            raise ValueError(
                "max_connections_per_host must be less than or equal to "
                "max_connections"
            )
        return self


class HttpRulesSettings(SettingsModel):
    timeouts: TimeoutRulesSettings = Field(
        default_factory=TimeoutRulesSettings
    )
    redirects: RedirectRulesSettings = Field(
        default_factory=RedirectRulesSettings
    )
    circuit_breaker: HostCircuitBreakerSettings = Field(
        default_factory=HostCircuitBreakerSettings
    )
    network_access: NetworkAccessSettings = Field(
        default_factory=NetworkAccessSettings
    )
    retries: RetryRulesSettings = Field(default_factory=RetryRulesSettings)
    statuses: HttpStatusRulesSettings = Field(
        default_factory=HttpStatusRulesSettings
    )
    connection_pool: ConnectionPoolSettings = Field(
        default_factory=ConnectionPoolSettings
    )
