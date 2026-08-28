"""
Public models and helpers for
crawler.governance.blacklist.storage.blacklist_connection.

Exports: BlacklistConnection.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from config.collection.governance import BlacklistManagerSettings


class BlacklistConnection:
    """Own SQLite connection management for blacklist storage operations."""

    def __init__(
        self,
        *,
        settings: BlacklistManagerSettings,
    ) -> None:
        self._settings = settings

    def execute(
        self,
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> None:
        """Execute a mutating statement and commit the transaction."""

        with self._managed_connection() as connection:
            connection.execute(statement, parameters)
            connection.commit()

    def fetchone(
        self,
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> sqlite3.Row | tuple[object, ...] | None:
        """Execute a query and return the first row, if any."""

        with self._managed_connection() as connection:
            row = connection.execute(statement, parameters).fetchone()
            if row is None or isinstance(row, (sqlite3.Row, tuple)):
                return row
            raise TypeError(
                f"unsupported sqlite row type: {type(row).__name__}"
            )

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection and always close it afterwards."""

        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._settings.blacklist_database_path)
        )
        connection.execute(
            f"PRAGMA busy_timeout = "
            f"{int(self._settings.blacklist_busy_timeout_ms)};"
        )
        return connection
