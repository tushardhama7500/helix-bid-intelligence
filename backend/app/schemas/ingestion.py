from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ScanResponse(BaseModel):
    files_found: int
    pdf_files: int
    html_files: int
    ignored_files: int
    bids: dict[str, int]


class FileResultOut(BaseModel):
    file_path: str
    status: str
    document_id: int | None = None
    pages: int = 0
    chunks: int = 0
    message: str | None = None


class RunResponse(BaseModel):
    total: int
    processed: int
    skipped: int
    failed: int
    results: list[FileResultOut]


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    bid_id: str
    file_name: str
    file_path: str
    file_type: str
    document_type: str
    title: str | None
    page_count: int | None
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class PageInfo(BaseModel):
    page_number: int | None
    section: str | None
    characters: int
    chunks: int


class DocumentDetail(DocumentSummary):
    content_hash: str
    stored_pages: int
    chunk_count: int
    total_characters: int
    pages: list[PageInfo]
