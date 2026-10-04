export interface ScanResponse {
  files_found: number;
  pdf_files: number;
  html_files: number;
  ignored_files: number;
  bids: Record<string, number>;
}

export interface RunResponse {
  total: number;
  processed: number;
  skipped: number;
  failed: number;
}

export interface DocumentSummary {
  id: number;
  bid_id: string;
  file_name: string;
  file_type: string;
  document_type: string;
  title: string | null;
  page_count: number | null;
  status: "pending" | "processing" | "completed" | "failed";
  error_message: string | null;
}

export interface EmbeddingStatus {
  total_chunks: number;
  embedded_chunks: number;
  pending_chunks: number;
  failed_chunks: number;
  embedding_model: string;
  embedding_dimension: number;
}

export interface EmbeddingSummary {
  total_chunks: number;
  embedded: number;
  skipped: number;
  failed: number;
  errors: string[];
}

export interface SearchSource {
  document_id: number;
  bid_id: string;
  file_name: string;
  document_type: string;
  page_number: number | null; // null for HTML
  chunk_index: number;
  metadata: Record<string, unknown>;
}

export interface SearchResult {
  chunk_id: number;
  content: string;
  score: number; // RRF ranking score, not a confidence
  retrieval_methods: ("vector" | "keyword")[];
  scores: {
    rrf: number;
    vector: { rank: number; cosine_distance: number; cosine_similarity: number } | null;
    keyword: { rank: number; ts_rank: number } | null;
  };
  source: SearchSource;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  results: SearchResult[];
}

export interface SearchFilters {
  bids: string[];
  document_types: string[];
  file_names: string[];
}

export interface SearchParams {
  query: string;
  bid_id?: string;
  document_type?: string;
  top_k: number;
}
