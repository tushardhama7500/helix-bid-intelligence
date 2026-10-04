import { useCallback, useEffect, useState } from "react";

import { fetchDocuments, runIngestion, scanDocuments } from "../services/api";
import type { DocumentSummary } from "../types/ingestion";

type Busy = "scan" | "run" | null;

function errorMessage(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
  return detail ?? "Request failed. Is the backend running?";
}

export default function Documents() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [busy, setBusy] = useState<Busy>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDocuments(await fetchDocuments());
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const act = async (kind: Exclude<Busy, null>) => {
    setBusy(kind);
    setError(null);
    setMessage(null);
    try {
      if (kind === "scan") {
        const s = await scanDocuments();
        setMessage(
          `Found ${s.files_found} files (${s.pdf_files} PDF, ${s.html_files} HTML), ${s.ignored_files} ignored.`,
        );
      } else {
        const r = await runIngestion();
        setMessage(
          `Ingestion done: ${r.processed} processed, ${r.skipped} skipped, ${r.failed} failed.`,
        );
        await load();
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const completed = documents.filter((d) => d.status === "completed").length;
  const failed = documents.filter((d) => d.status === "failed").length;

  return (
    <section>
      <h1>Documents</h1>
      <p className="tagline">Ingested bid documents.</p>

      <div className="grid stats">
        <div className="card"><h3>Total documents</h3><strong>{documents.length}</strong></div>
        <div className="card"><h3>Processed</h3><strong>{completed}</strong></div>
        <div className="card"><h3>Failed</h3><strong>{failed}</strong></div>
      </div>

      <div className="actions">
        <button onClick={() => act("scan")} disabled={busy !== null}>
          {busy === "scan" ? "Scanning..." : "Scan Documents"}
        </button>
        <button onClick={() => act("run")} disabled={busy !== null}>
          {busy === "run" ? "Ingesting..." : "Run Ingestion"}
        </button>
      </div>

      {message && <p className="info">{message}</p>}
      {error && <p className="notice">{error}</p>}

      <table>
        <thead>
          <tr><th>File</th><th>Bid</th><th>Type</th><th>Pages</th><th>Status</th></tr>
        </thead>
        <tbody>
          {documents.map((d) => (
            <tr key={d.id}>
              <td title={d.error_message ?? d.file_name}>{d.file_name}</td>
              <td>{d.bid_id}</td>
              <td>{d.document_type}</td>
              <td>{d.page_count ?? "-"}</td>
              <td>
                <span className={`badge ${d.status === "completed" ? "ok" : d.status === "failed" ? "bad" : "unknown"}`}>
                  {d.status}
                </span>
              </td>
            </tr>
          ))}
          {documents.length === 0 && (
            <tr><td colSpan={5} className="empty">No documents yet. Run ingestion.</td></tr>
          )}
        </tbody>
      </table>
    </section>
  );
}
