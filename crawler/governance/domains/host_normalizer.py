"""Canonical host identity normalization for crawler governance."""

from __future__ import annotations

from ipaddress import ip_address


class HostNormalizer:
    """Normalize raw hosts into canonical IP or IDNA DNS keys."""

    def normalize(self, host: str | None) -> str | None:
        """Return a canonical host, or ``None`` for invalid input."""

        if not isinstance(host, str):
            return None

        candidate = host.strip()
        if not candidate:
            return None

        bracketed = candidate.startswith("[") or candidate.endswith("]")
        if bracketed:
            if not (candidate.startswith("[") and candidate.endswith("]")):
                return None
            candidate = candidate[1:-1].strip()
            if not candidate:
                return None

        candidate = candidate.removesuffix(".").lower()
        if not candidate or "%" in candidate:
            return None

        try:
            address = ip_address(candidate)
        except ValueError:
            address = None

        if address is not None:
            if bracketed and address.version != 6:
                return None
            return address.compressed.lower()

        if bracketed or ":" in candidate:
            return None

        try:
            canonical = candidate.encode("idna").decode("ascii").lower()
            canonical.encode("ascii").decode("idna")
        except UnicodeError:
            return None

        if len(canonical) > 253:
            return None

        labels = canonical.split(".")
        if all(label.isdigit() for label in labels):
            return None
        if any(not self._valid_dns_label(label) for label in labels):
            return None

        return canonical

    def require(self, host: str | None) -> str:
        """Return a canonical host or raise for missing/invalid input."""

        normalized = self.normalize(host)
        if normalized is None:
            raise ValueError("A valid host is required")
        return normalized

    @staticmethod
    def _valid_dns_label(label: str) -> bool:
        if not label or len(label) > 63:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        return all(
            character.isalnum() or character == "-" for character in label
        )
