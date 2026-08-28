"""
Public models and helpers for
crawler.governance.blacklist.storage.blacklist_table_schema.

Exports: BlacklistTableSchema.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .blacklist_connection import BlacklistConnection


class BlacklistTableSchema:
    """Manage blacklist database domains initialization."""

    _VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        *,
        database_handler: BlacklistConnection,
        database_path: Path,
        table_name: str,
    ) -> None:
        self._database_handler = database_handler
        self._database_path = database_path
        self._table_name = self._validate_table_name(table_name)

    def initialize(self) -> None:
        """Ensure the blacklist database domains exists."""

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        statement = (
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            "url TEXT PRIMARY KEY"
            ");"
        )
        self._database_handler.execute(statement)

    @classmethod
    def _validate_table_name(cls, table_name: str) -> str:
        if not cls._VALID_TABLE_NAME.fullmatch(table_name):
            raise ValueError(
                f"Invalid blacklist table name: {table_name!r}",
            )
        return table_name
