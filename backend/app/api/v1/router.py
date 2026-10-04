from fastapi import APIRouter

from app.api.v1.routes import embeddings, extraction, health, ingestion, pipeline, reconciliation, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(ingestion.router)
api_router.include_router(embeddings.router)
api_router.include_router(search.router)
api_router.include_router(extraction.router)
api_router.include_router(reconciliation.router)
api_router.include_router(pipeline.pipeline_router)
api_router.include_router(pipeline.qa_router)
