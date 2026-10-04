"""Metadata filters, applied inside the SQL query (aliases: c=chunks, d=documents, p=pages)."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchFilters:
    bid_id: str | None = None
    document_type: str | None = None
    file_name: str | None = None  # exact match
    document_id: int | None = None
    page_number: int | None = None

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """`" AND ..."` clause text plus bind parameters; empty when no filters are set."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for column, name in (
            ("c.bid_id", "bid_id"),
            ("d.document_type", "document_type"),
            ("d.file_name", "file_name"),
            ("c.document_id", "document_id"),
            ("p.page_number", "page_number"),
        ):
            value = getattr(self, name)
            if value is not None:
                clauses.append(f" AND {column} = :f_{name}")
                params[f"f_{name}"] = value
        return "".join(clauses), params

    @property
    def active(self) -> bool:
        return bool(self.to_sql()[1])
