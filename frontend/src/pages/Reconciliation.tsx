import { useEffect, useState } from "react";

import { fetchSearchFilters, reconcileBid } from "../services/api";
import type { ExtractionSource } from "../types/extraction";
import type { BidReconciliation, ReconMethod, ReconStatus, ReconciledField } from "../types/reconciliation";

const METHOD: Record<ReconMethod, string> = {
  explicit_addendum_change: "Explicit addendum change",
  ordered_addendum: "Ordered addenda",
  llm_reconciliation: "Groq reconciliation",
  no_change: "No change",
  unresolved: "Unresolved",
};

const STATUS: Record<ReconStatus, { label: string; tone: string }> = {
  resolved: { label: "Resolved", tone: "ok" },
  unchanged: { label: "Unchanged", tone: "unknown" },
  conflict: { label: "Conflict", tone: "warn" },
  insufficient_evidence: { label: "Insufficient evidence", tone: "bad" },
};

function show(value: unknown): string {
  if (value === null || value === undefined) return "-";
  return Array.isArray(value) ? value.join("; ") : typeof value === "string" ? value : JSON.stringify(value);
}

function Sources({ sources }: { sources: ExtractionSource[] }) {
  if (sources.length === 0) return null;
  return (
    <ul className="sources">
      {sources.map((s) => (
        <li key={s.chunk_id} title={`${s.document_type}, chunk ${s.chunk_index}`}>
          {s.file_name} - {s.page_number !== null ? `Page ${s.page_number}` : s.section ? `Section: ${s.section}` : "HTML"}
        </li>
      ))}
    </ul>
  );
}

function FieldCard({ f }: { f: ReconciledField }) {
  const unresolved = f.status === "conflict" || f.status === "insufficient_evidence";
  return (
    <article className="result">
      <div className="result-head">
        <h3>{f.field}</h3>
        <span className={`badge ${STATUS[f.status].tone}`}>{STATUS[f.status].label}</span>
      </div>

      <div className="recon-row">
        <span className="recon-label">Original</span>
        <div>
          {show(f.original_value)}
          <Sources sources={f.original_sources} />
        </div>
      </div>

      {f.changes.map((c, i) => (
        <div className="recon-row" key={i}>
          <span className="recon-label">
            {c.addendum_number !== null ? `Addendum ${c.addendum_number}` : "Addendum (number unknown)"}
            <span className="hint"> {c.change_type}</span>
          </span>
          <div>
            {c.new_value !== null && c.new_value !== undefined ? show(c.new_value) : <span className="hint">no new value</span>}
            <Sources sources={c.sources} />
          </div>
        </div>
      ))}

      {f.status === "conflict" &&
        f.conflicting_values.map((c, i) => (
          <div className="recon-row" key={`c${i}`}>
            <span className="recon-label">Conflicting</span>
            <div>
              {show(c.value)}
              <Sources sources={c.sources} />
            </div>
          </div>
        ))}

      <div className={`recon-row recon-final${unresolved ? " unresolved" : ""}`}>
        <span className="recon-label">Final</span>
        <div>
          {unresolved || f.final_value === null ? (
            <strong>Unable to determine authoritative value</strong>
          ) : (
            <strong>{show(f.final_value)}</strong>
          )}
          {!unresolved && <Sources sources={f.sources} />}
        </div>
      </div>

      <div className="meta">
        <span>Resolution: {METHOD[f.resolution_method]}</span>
        <span>Confidence: {f.confidence.toFixed(2)}</span>
      </div>
      {f.notes && <p className="hint">{f.notes}</p>}
    </article>
  );
}

export default function Reconciliation() {
  const [bids, setBids] = useState<string[]>([]);
  const [bid, setBid] = useState("");
  const [result, setResult] = useState<BidReconciliation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSearchFilters()
      .then((f) => {
        setBids(f.bids);
        setBid((current) => current || f.bids[0] || "");
      })
      .catch(() => setBids([]));
  }, []);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await reconcileBid(bid));
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Reconciliation failed. Is the backend running?");
    } finally {
      setBusy(false);
    }
  };

  const interesting = result?.fields.filter((f) => f.changes.length > 0 || f.original_value !== null || f.status === "conflict") ?? [];
  const quiet = result?.fields.filter((f) => !interesting.includes(f)) ?? [];

  return (
    <section>
      <h1>Reconciliation</h1>
      <p className="tagline">
        How addenda change the original requirements. The original value and every change stay visible; uncertain
        results are never hidden.
      </p>

      <div className="actions">
        <select value={bid} onChange={(e) => setBid(e.target.value)} aria-label="Bid">
          {bids.map((b) => (
            <option key={b}>{b}</option>
          ))}
        </select>
        <button onClick={run} disabled={busy || !bid}>
          {busy ? "Reconciling..." : "Reconcile Bid"}
        </button>
      </div>

      {error && <p className="notice">{error}</p>}
      {result && !result.complete && <p className="notice">Incomplete: {result.errors.join("; ")}</p>}
      {result && result.reconciliation_required.length > 0 && (
        <p className="warning">Unresolved: {result.reconciliation_required.join(", ")}</p>
      )}

      {result && (
        <>
          <p className="hint">
            Timeline:{" "}
            {result.timeline.addenda.length === 0
              ? "no addenda for this bid"
              : result.timeline.addenda
                  .map((a) => `Addendum ${a.addendum_number ?? "?"} (${a.file_name})`)
                  .join(" -> ")}
            {" "}[ordering: {result.timeline.ordering}
            {result.timeline.reliable ? "" : ", not reliable"}]. Confidence is a heuristic, not a calibrated probability.
            {result.used_llm ? ` Groq was used for ambiguous wording (${result.model}).` : " Deterministic rules only."}
          </p>
          {interesting.map((f) => (
            <FieldCard key={f.key} f={f} />
          ))}
          {quiet.length > 0 && (
            <p className="hint">No original value or addendum change found for: {quiet.map((f) => f.field).join(", ")}.</p>
          )}
        </>
      )}
    </section>
  );
}
