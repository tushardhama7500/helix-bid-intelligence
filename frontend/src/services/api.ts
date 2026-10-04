import axios, { type AxiosError } from "axios";

import type { BidExtraction } from "../types/extraction";
import type { GroqHealthResponse, HealthResponse } from "../types/health";
import type { PipelineResponse, QAAnswer } from "../types/pipeline";
import type { BidReconciliation } from "../types/reconciliation";
import type {
  DocumentSummary,
  EmbeddingStatus,
  EmbeddingSummary,
  RunResponse,
  ScanResponse,
  SearchFilters,
  SearchParams,
  SearchResponse,
} from "../types/ingestion";

const baseURL = `${import.meta.env.VITE_API_URL ?? "http://localhost:8000"}/api/v1`;

// Health endpoints answer 503 with a useful body; treat that as data, not an error.
const healthClient = axios.create({
  baseURL,
  timeout: 10_000,
  validateStatus: (status) => status < 600,
});

// Everything else uses axios defaults: non-2xx responses throw.
const client = axios.create({ baseURL, timeout: 10_000 });
// Ingestion can take a while on the first run.
const longClient = axios.create({ baseURL, timeout: 10 * 60_000 });

export async function fetchHealth(): Promise<HealthResponse> {
  return (await healthClient.get<HealthResponse>("/health")).data;
}

export async function fetchGroqHealth(): Promise<GroqHealthResponse> {
  return (await healthClient.get<GroqHealthResponse>("/health/groq")).data;
}

export async function scanDocuments(): Promise<ScanResponse> {
  return (await client.post<ScanResponse>("/ingestion/scan")).data;
}

export async function runIngestion(): Promise<RunResponse> {
  return (await longClient.post<RunResponse>("/ingestion/run")).data;
}

export async function fetchDocuments(): Promise<DocumentSummary[]> {
  return (await client.get<DocumentSummary[]>("/ingestion/documents")).data;
}

export async function fetchEmbeddingStatus(): Promise<EmbeddingStatus> {
  return (await client.get<EmbeddingStatus>("/embeddings/status")).data;
}

// The first run downloads the model and embeds every chunk, so allow a long timeout.
export async function generateEmbeddings(): Promise<EmbeddingSummary> {
  return (await longClient.post<EmbeddingSummary>("/embeddings/generate")).data;
}

// The first search loads the embedding model, so allow more than the default timeout.
export async function searchChunks(params: SearchParams): Promise<SearchResponse> {
  return (await axios.post<SearchResponse>(`${baseURL}/search`, params, { timeout: 120_000 })).data;
}

// Extraction makes several Groq calls, so it can take a minute or more.
export async function extractBid(bidId: string): Promise<BidExtraction> {
  return (await longClient.post<BidExtraction>(`/extraction/${encodeURIComponent(bidId)}`)).data;
}

export async function reconcileBid(bidId: string): Promise<BidReconciliation> {
  return (await longClient.post<BidReconciliation>(`/reconciliation/${encodeURIComponent(bidId)}`)).data;
}

export async function runPipeline(bidId: string): Promise<PipelineResponse> {
  try {
    return (await longClient.post<PipelineResponse>(`/pipeline/${encodeURIComponent(bidId)}`)).data;
  } catch (err) {
    // A failed pipeline answers 422/500/503 but still carries its structured explanation.
    const body = (err as AxiosError<PipelineResponse>).response?.data;
    if (body && typeof body === "object" && "stages" in body) return body;
    throw err;
  }
}

export async function askQuestion(bidId: string, question: string): Promise<QAAnswer> {
  return (await longClient.post<QAAnswer>(`/qa/${encodeURIComponent(bidId)}`, { question })).data;
}

export async function fetchSearchFilters(): Promise<SearchFilters> {
  return (await client.get<SearchFilters>("/search/filters")).data;
}
