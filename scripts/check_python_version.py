"""Validate the only interpreter currently supported by this project."""

from __future__ import annotations

import sys

SUPPORTED_PYTHON = (3, 12)


def is_supported(version: tuple[int, int]) -> bool:
    """Return whether a major/minor interpreter version is supported."""

    return version == SUPPORTED_PYTHON


def main() -> int:
    """Exit nonzero before dependency installation on an unsupported Python."""

    current = sys.version_info[:2]
    if is_supported(current):
        return 0

    actual = f"Python {current[0]}.{current[1]}"
    print(
        f"Unsupported interpreter: {actual}. "
        "This project currently requires Python 3.12.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
