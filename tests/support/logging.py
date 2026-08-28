"""Explicit logger test double for dependency-injected components."""

from __future__ import annotations


class TestLogger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        pass

    def info(self, *_args: object, **_kwargs: object) -> None:
        pass

    def warning(self, *_args: object, **_kwargs: object) -> None:
        pass

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass

    def critical(self, *_args: object, **_kwargs: object) -> None:
        pass

    def exception(self, *_args: object, **_kwargs: object) -> None:
        pass


TEST_LOGGER = TestLogger()
