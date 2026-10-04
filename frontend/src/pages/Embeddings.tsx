import { useCallback, useEffect, useState } from "react";

import { fetchEmbeddingStatus, generateEmbeddings } from "../services/api";
import type { EmbeddingStatus, EmbeddingSummary } from "../types/ingestion";

function errorMessage(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
  return detail ?? "Request failed. Is the backend running?";
}

export default function Embeddings() {
  const [status, setStatus] = useState<EmbeddingStatus | null>(null);
  const [summary, setSummary] = useState<EmbeddingSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setStatus(await fetchEmbeddingStatus());
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const generate = async () => {
    setBusy(true);
    setError(null);
    setSummary(null);
    try {
      setSummary(await generateEmbeddings());
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
      await load();
    }
  };

  return (
    <section>
      <h1>Embeddings</h1>
      <p className="tagline">
        Local vectors stored in pgvector
        {status && ` - ${status.embedding_model} (${status.embedding_dimension} dimensions)`}
      </p>

      <div className="grid stats">
        <div className="card"><h3>Total chunks</h3><strong>{status?.total_chunks ?? "-"}</strong></div>
        <div className="card"><h3>Embedded</h3><strong>{status?.embedded_chunks ?? "-"}</strong></div>
        <div className="card"><h3>Pending</h3><strong>{status?.pending_chunks ?? "-"}</strong></div>
        <div className="card"><h3>Failed</h3><strong>{status?.failed_chunks ?? "-"}</strong></div>
      </div>

      <div className="actions">
        <button onClick={generate} disabled={busy}>
          {busy ? "Generating..." : "Generate Embeddings"}
        </button>
      </div>

      {summary && (
        <p className={summary.failed ? "notice" : "info"}>
          {summary.embedded} embedded, {summary.skipped} skipped, {summary.failed} failed
          (of {summary.total_chunks} chunks).
          {summary.errors.length > 0 && ` First error: ${summary.errors[0]}`}
        </p>
      )}
      {error && <p className="notice">{error}</p>}
    </section>
  );
}
