"""Ingestion tables: documents, their extracted pages/sections, and chunks."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# JSONB on PostgreSQL; plain JSON elsewhere (lets unit tests use SQLite).
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


# pgvector column on PostgreSQL (dimension is fixed by `ensure_embedding_schema`,
# which derives it from the embedding model); plain JSON on SQLite for unit tests.
EMBEDDING_TYPE = Vector().with_variant(JSON(none_as_null=True), "sqlite")


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    bid_id: Mapped[str] = mapped_column(String(100), index=True)
    file_name: Mapped[str] = mapped_column(String(500))
    # Path relative to DATA_DIR, POSIX separators. Identifies the document.
    file_path: Mapped[str] = mapped_column(String(1000), unique=True)
    file_type: Mapped[str] = mapped_column(String(10))
    document_type: Mapped[str] = mapped_column(String(50))
    title: Mapped[str | None] = mapped_column(String(1000))
    page_count: Mapped[int | None]
    # SHA-256 of the file bytes; used to detect unchanged files.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(
        String(20), default=DocumentStatus.PENDING.value, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DocumentPage(Base):
    """A PDF page, or a section of an HTML document (`page_number` is NULL)."""

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # `metadata` is reserved on declarative classes, hence the attribute name.
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON_TYPE, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_id: Mapped[int] = mapped_column(
        ForeignKey("document_pages.id", ondelete="CASCADE"), index=True
    )
    bid_id: Mapped[str] = mapped_column(String(100), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON_TYPE, default=dict)
    # NULL until embedded. New chunks (including re-ingested ones) start as NULL,
    # so an old embedding can never stay attached to changed content.
    embedding: Mapped[list[float] | None] = mapped_column(EMBEDDING_TYPE)
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    embedding_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
