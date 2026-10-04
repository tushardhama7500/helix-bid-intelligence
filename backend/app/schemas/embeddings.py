from pydantic import BaseModel


class GenerateResponse(BaseModel):
    total_chunks: int
    embedded: int
    skipped: int
    failed: int
    errors: list[str] = []


class StatusResponse(BaseModel):
    total_chunks: int
    embedded_chunks: int
    pending_chunks: int
    failed_chunks: int
    embedding_model: str
    embedding_dimension: int
