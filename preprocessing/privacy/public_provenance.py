"""Remove private URL components before provenance leaves preprocessing."""

from urllib.parse import urlsplit, urlunsplit


def public_source_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    hostname = parsed.hostname or ""
    if not hostname:
        return ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (parsed.scheme, f"{hostname}{port}", parsed.path, "", "")
    )
