import { useEffect, useState } from "react";

import { extractBid, fetchSearchFilters } from "../services/api";
import type { BidExtraction, ExtractedField, ExtractionSource, FieldValue } from "../types/extraction";

function errorMessage(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
  return typeof detail === "string" ? detail : "Extraction failed. Is the backend running?";
}

function sourceLabel(s: ExtractionSource): string {
  const where = s.page_number !== null ? `Page ${s.page_number}` : s.section ? `Section: ${s.section}` : "HTML";
  return `${s.file_name} - ${where}`;
}

function Sources({ sources }: { sources: ExtractionSource[] }) {
  if (sources.length === 0) return <span className="hint">-</span>;
  return (
    <ul className="sources">
      {sources.map((s) => (
        <li key={s.chunk_id} title={`${s.document_type}, chunk ${s.chunk_index}`}>
          {sourceLabel(s)}
        </li>
      ))}
    </ul>
  );
}

function Value({ value }: { value: FieldValue | null }) {
  if (value === null) return <span className="hint">null</span>;
  if (typeof value === "string") return <>{value}</>;
  if (Array.isArray(value)) {
    if (value.length > 0 && typeof value[0] === "object") {
      return (
        <>
          {(value as Record<string, unknown>[]).map((p, i) => (
            <SpecBlock key={i} spec={p} />
          ))}
        </>
      );
    }
    return (
      <ul className="sources">
        {(value as string[]).map((v) => (
          <li key={v}>{v}</li>
        ))}
      </ul>
    );
  }
  return <SpecBlock spec={value} />;
}

function SpecBlock({ spec }: { spec: Record<string, unknown> }) {
  const specs = (spec.specifications ?? {}) as Record<string, string>;
  return (
    <div>
      {typeof spec.product === "string" && <strong>{spec.product}</strong>}
      <ul className="sources">
        {Object.entries(specs).map(([k, v]) => (
          <li key={k}>
            {k}: {String(v)}
          </li>
        ))}
      </ul>
    </div>
  );
}

const TONE: Record<ExtractedField["status"], string> = {
  found: "ok",
  not_found: "unknown",
  conflict: "warn",
  unsupported: "bad",
  error: "bad",
};

function statusLabel(f: ExtractedField): string {
  if (f.status === "conflict") return "Conflict - reconciliation required";
  return { found: "Found", not_found: "Not found", unsupported: "Unsupported", error: "Error" }[f.status];
}

function Row({ name, field }: { name: string; field: ExtractedField }) {
  return (
    <tr>
      <td className="field-name">{name}</td>
      <td>
        <span className={`badge ${TONE[field.status]}`}>{statusLabel(field)}</span>
      </td>
      <td className="field-value">
        {field.status === "conflict" ? (
          <div className="conflict">
            {field.conflicting_values.map((c, i) => (
              <div key={i}>
                <Value value={c.value} />
                <Sources sources={c.sources} />
              </div>
            ))}
          </div>
        ) : (
          <Value value={field.value} />
        )}
        {field.notes && <div className="hint">{field.notes}</div>}
      </td>
      <td>{field.status === "found" ? field.confidence.toFixed(2) : "-"}</td>
      <td>{field.status === "conflict" ? <span className="hint">see values</span> : <Sources sources={field.sources} />}</td>
    </tr>
  );
}

export default function Extraction() {
  const [bids, setBids] = useState<string[]>([]);
  const [bid, setBid] = useState("");
  const [result, setResult] = useState<BidExtraction | null>(null);
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
      setResult(await extractBid(bid));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h1>Extraction</h1>
      <p className="tagline">
        Structured fields extracted by Groq from retrieved evidence. Every value is cited; conflicts are reported, not
        resolved.
      </p>

      <div className="actions">
        <select value={bid} onChange={(e) => setBid(e.target.value)} aria-label="Bid">
          {bids.map((b) => (
            <option key={b}>{b}</option>
          ))}
        </select>
        <button onClick={run} disabled={busy || !bid}>
          {busy ? "Extracting..." : "Extract Bid"}
        </button>
      </div>

      {error && <p className="notice">{error}</p>}
      {result && !result.complete && (
        <p className="notice">Some fields could not be extracted: {result.errors.join("; ")}</p>
      )}
      {result && result.reconciliation_required.length > 0 && (
        <p className="warning">
          Conflicting values found for: {result.reconciliation_required.join(", ")}. Addendum reconciliation is required
          before a final value can be stated.
        </p>
      )}

      {result && (
        <>
          <p className="hint">
            Confidence is an extraction-quality heuristic (0-1), not a calibrated probability.
            {result.model && ` Model: ${result.model}.`}
          </p>
          <table className="extraction">
            <thead>
              <tr>
                <th>Field</th>
                <th>Status</th>
                <th>Value</th>
                <th>Confidence</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.fields).map(([name, field]) => (
                <Row key={name} name={name} field={field} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
