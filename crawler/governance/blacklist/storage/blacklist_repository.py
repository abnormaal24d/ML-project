"""
Public models and helpers for
crawler.governance.blacklist.storage.blacklist_repository.

Exports: BlacklistRepository.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blacklist_connection import BlacklistConnection


class BlacklistRepository:
    """Persist blacklist URLs in SQLite."""

    _VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        *,
        database_handler: BlacklistConnection,
        table_name: str,
    ) -> None:
        self._database_handler = database_handler
        self._table_name = self._validate_table_name(table_name)

    def add_url(self, *, url: str) -> None:
        """Insert a URL into the blacklist when missing."""

        statement = (
            f"INSERT OR IGNORE INTO {self._table_name} (url) VALUES (?);"
        )
        self._database_handler.execute(statement, (url,))

    def remove_url(self, *, url: str) -> None:
        """Delete a URL from the blacklist."""

        statement = f"DELETE FROM {self._table_name} WHERE url = ?;"  # nosec B608
        self._database_handler.execute(statement, (url,))

    def update_url(
        self,
        *,
        old_url: str,
        new_url: str,
    ) -> None:
        """Update an existing blacklist URL entry."""

        statement = f"UPDATE {self._table_name} SET url = ? WHERE url = ?;"  # nosec B608
        self._database_handler.execute(statement, (new_url, old_url))

    def clear(self) -> None:
        """Remove all blacklist entries."""

        statement = f"DELETE FROM {self._table_name};"  # nosec B608
        self._database_handler.execute(statement)

    def contains(self, *, url: str) -> bool:
        """Return whether a URL exists in the blacklist."""

        statement = f"SELECT 1 FROM {self._table_name} WHERE url = ? LIMIT 1;"  # nosec B608
        row = self._database_handler.fetchone(statement, (url,))
        return row is not None

    @classmethod
    def _validate_table_name(cls, table_name: str) -> str:
        if not cls._VALID_TABLE_NAME.fullmatch(table_name):
            raise ValueError(
                f"Invalid blacklist table name: {table_name!r}",
            )
        return table_name
