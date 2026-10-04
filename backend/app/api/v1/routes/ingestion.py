from collections import Counter
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.ingestion import repository as repo
from app.ingestion.service import IngestionService
from app.schemas.ingestion import (
    DocumentDetail,
    DocumentSummary,
    PageInfo,
    RunResponse,
    ScanResponse,
)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def get_ingestion_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    return IngestionService.from_settings(session, settings)


ServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


@router.post("/scan", response_model=ScanResponse)
async def scan(service: ServiceDep) -> ScanResponse:
    """List supported files under the data directory (nothing is stored)."""
    try:
        found = service.scan()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    types = Counter(f.file_type for f in found.files)
    return ScanResponse(
        files_found=len(found.files),
        pdf_files=types["pdf"],
        html_files=types["html"],
        ignored_files=found.ignored,
        bids=dict(Counter(f.bid_id for f in found.files)),
    )


@router.post("/run", response_model=RunResponse)
async def run(service: ServiceDep, force: bool = False) -> RunResponse:
    """Ingest new or changed documents; unchanged ones are skipped unless `force`."""
    try:
        summary = await service.run(force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RunResponse(**asdict(summary))


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db)], bid_id: str | None = None
) -> list[DocumentSummary]:
    documents = await repo.list_documents(session, bid_id)
    return [DocumentSummary.model_validate(d) for d in documents]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int, session: Annotated[AsyncSession, Depends(get_db)]
) -> DocumentDetail:
    document = await repo.get_document(session, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    pages = await repo.get_page_stats(session, document_id)
    return DocumentDetail(
        **DocumentSummary.model_validate(document).model_dump(),
        content_hash=document.content_hash,
        stored_pages=len(pages),
        chunk_count=sum(p["chunks"] for p in pages),
        total_characters=sum(p["characters"] for p in pages),
        pages=[PageInfo(**p) for p in pages],
    )
