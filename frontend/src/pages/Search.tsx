import { FormEvent, useEffect, useState } from "react";

import { fetchSearchFilters, searchChunks } from "../services/api";
import type { SearchFilters, SearchResponse, SearchResult } from "../types/ingestion";

function errorMessage(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "Invalid search request.";
  return "Search failed. Is the backend running?";
}

function methodLabel(methods: SearchResult["retrieval_methods"]): string {
  if (methods.length === 2) return "Both";
  return methods[0] === "vector" ? "Vector" : "Keyword";
}

function Citation({ result }: { result: SearchResult }) {
  const { source } = result;
  const section = typeof source.metadata.section === "string" ? source.metadata.section : null;
  const where =
    source.page_number !== null ? `Page ${source.page_number}` : section ? `Section: ${section}` : "HTML document";
  return (
    <div className="citation">
      Source:
      <strong>{source.file_name}</strong>
      {where} - chunk {source.chunk_index}
    </div>
  );
}

export default function Search() {
  const [filters, setFilters] = useState<SearchFilters | null>(null);
  const [query, setQuery] = useState("");
  const [bid, setBid] = useState("");
  const [docType, setDocType] = useState("");
  const [topK, setTopK] = useState(5);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSearchFilters()
      .then(setFilters)
      .catch(() => setFilters(null));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResponse(
        await searchChunks({
          query,
          bid_id: bid || undefined,
          document_type: docType || undefined,
          top_k: topK,
        }),
      );
    } catch (err) {
      setResponse(null);
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h1>Search</h1>
      <p className="tagline">Hybrid retrieval over ingested documents: evidence only, no generated answers.</p>

      <form className="search-form" onSubmit={submit}>
        <label>
          Query
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. final submission deadline, or WD22TB4"
          />
        </label>
        <label>
          Bid
          <select value={bid} onChange={(e) => setBid(e.target.value)}>
            <option value="">All bids</option>
            {filters?.bids.map((b) => (
              <option key={b}>{b}</option>
            ))}
          </select>
        </label>
        <label>
          Document type
          <select value={docType} onChange={(e) => setDocType(e.target.value)}>
            <option value="">All types</option>
            {filters?.document_types.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <label>
          Top K
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[3, 5, 10, 20].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={busy || !query.trim()}>
          {busy ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <p className="notice">{error}</p>}
      {response && response.results.length === 0 && <p className="hint">No matching evidence found.</p>}

      {response?.results.map((r, i) => (
        <article className="result" key={r.chunk_id}>
          <div className="result-head">
            <h3>Result {i + 1}</h3>
            <span
              className="hint"
              title="Reciprocal Rank Fusion score. A ranking score, not a probability or confidence."
            >
              RRF score {r.score.toFixed(4)}
            </span>
          </div>
          <Citation result={r} />
          <div className="meta">
            <span className="chip">{r.source.bid_id}</span>
            <span className="chip">{r.source.document_type}</span>
            Retrieved by: {methodLabel(r.retrieval_methods)}
            {r.retrieval_methods.map((m) => (
              <span key={m} className={`chip ${m}`}>
                {m}
              </span>
            ))}
            {r.scores.vector && (
              <span>
                vector similarity {r.scores.vector.cosine_similarity.toFixed(3)} (rank {r.scores.vector.rank})
              </span>
            )}
            {r.scores.keyword && (
              <span>
                keyword ts_rank {r.scores.keyword.ts_rank.toFixed(3)} (rank {r.scores.keyword.rank})
              </span>
            )}
          </div>
          <p className="chunk-text">{r.content}</p>
        </article>
      ))}
    </section>
  );
}
