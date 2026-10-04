import { FormEvent, useEffect, useState } from "react";

import { askQuestion, fetchSearchFilters, runPipeline } from "../services/api";
import type { ExtractionSource } from "../types/extraction";
import type { PipelineResponse, QAAnswer, ReportField, StageState } from "../types/pipeline";

const STAGES = ["ingestion", "retrieval", "extraction", "reconciliation", "validation"];
const MARK: Record<StageState, { symbol: string; tone: string }> = {
  complete: { symbol: "✓", tone: "ok" },
  partial: { symbol: "~", tone: "warn" },
  failed: { symbol: "✗", tone: "bad" },
  skipped: { symbol: "-", tone: "unknown" },
};
const FAILURE_TEXT: Record<string, string> = {
  infrastructure: "A required service (database or embedding model) is unavailable.",
  no_documents: "None of this bid's documents could be ingested.",
  no_evidence: "No searchable evidence exists for this bid yet.",
  validation: "The analysis failed its own consistency checks, so no results are shown.",
  internal: "An unexpected error occurred.",
};
const BANNER: Record<PipelineResponse["status"], { cls: string; text: string }> = {
  complete: { cls: "info", text: "Analysis complete. Every stage finished and all citations were validated." },
  partial: {
    cls: "warning",
    text: "Analysis partially completed. Some stages were skipped or incomplete (see the stage list); only validated results are shown.",
  },
  failed: { cls: "notice", text: "Analysis failed. No results are shown." },
};
const REFUSAL = "I could not find sufficient evidence to answer this question.";

function detail(err: unknown, fallback: string): string {
  const d = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
  return typeof d === "string" ? d : fallback;
}

function where(s: { file_name?: string; file?: string; page_number?: number | null; page?: number | null; section?: string | null }) {
  const page = s.page_number ?? s.page ?? null;
  return `${s.file_name ?? s.file} - ${page !== null ? `Page ${page}` : s.section ? `Section: ${s.section}` : "HTML"}`;
}

function SourceList({ sources }: { sources: ExtractionSource[] }) {
  return (
    <ul className="sources">
      {sources.map((s) => (
        <li key={s.chunk_id}>{where(s)}</li>
      ))}
    </ul>
  );
}

function isSpec(v: unknown): v is { product?: unknown; specifications?: Record<string, unknown> } {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function show(v: unknown): string {
  return Array.isArray(v) ? v.map(String).join("; ") : typeof v === "string" ? v : JSON.stringify(v);
}

// Strings stay inline; product specifications (one object or a list of them) become a readable block per product.
function ValueView({ value }: { value: unknown }) {
  const specs = Array.isArray(value) ? value.filter(isSpec) : isSpec(value) ? [value] : [];
  if (specs.length === 0 || (Array.isArray(value) && specs.length !== value.length)) return <>{show(value)}</>;
  return (
    <>
      {specs.map((p, i) => (
        <div key={i}>
          {typeof p.product === "string" && <strong>{p.product}</strong>}
          <ul className="sources">
            {Object.entries(p.specifications ?? {}).map(([k, v]) => (
              <li key={k}>
                {k}: {String(v)}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </>
  );
}

function FieldRow({ f }: { f: ReportField }) {
  const unresolved = f.status === "unresolved";
  return (
    <tr>
      <td className="field-name">{f.field}</td>
      <td>
        {unresolved ? (
          <span className="badge warn">Unresolved</span>
        ) : f.value === null || f.value === undefined ? (
          <span className="hint">{f.status.replace("_", " ")}</span>
        ) : (
          <>
            <ValueView value={f.value} />
            {f.changed_by_addendum && <div className="hint">changed by addendum; original: {show(f.original_value)}</div>}
          </>
        )}
      </td>
      <td>{f.value !== null && f.value !== undefined ? f.confidence.toFixed(2) : "-"}</td>
      <td>
        {f.value !== null && f.value !== undefined && <SourceList sources={f.sources} />}
        {f.changed_by_addendum && <SourceList sources={f.original_sources} />}
      </td>
    </tr>
  );
}

export default function Analysis() {
  const [bids, setBids] = useState<string[]>([]);
  const [bid, setBid] = useState("");
  const [run, setRun] = useState<PipelineResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [runError, setRunError] = useState<string | null>(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QAAnswer | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    fetchSearchFilters()
      .then((f) => {
        setBids(f.bids);
        setBid((c) => c || f.bids[0] || "");
      })
      .catch(() => setBids([]));
  }, []);

  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [running]);

  const analyse = async () => {
    setRunning(true);
    setRunError(null);
    setRun(null);
    try {
      setRun(await runPipeline(bid));
    } catch (err) {
      setRunError(detail(err, "Analysis failed. Is the backend running?"));
    } finally {
      setRunning(false);
    }
  };

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    setAsking(true);
    setAskError(null);
    setAnswer(null);
    try {
      setAnswer(await askQuestion(bid, question));
    } catch (err) {
      setAskError(detail(err, "Could not get an answer. Is the backend running?"));
    } finally {
      setAsking(false);
    }
  };

  return (
    <section>
      <h1>Bid Analysis</h1>
      <p className="tagline">
        Runs ingestion, retrieval, extraction, reconciliation and validation, then answers questions from the cited
        evidence.
      </p>

      <div className="actions">
        <select value={bid} onChange={(e) => setBid(e.target.value)} aria-label="Bid">
          {bids.map((b) => (
            <option key={b}>{b}</option>
          ))}
        </select>
        <button onClick={analyse} disabled={running || !bid}>
          {running ? "Analysing..." : "Run Analysis"}
        </button>
      </div>

      {running && (
        <p className="info" role="status">
          Analysing {bid}... {elapsed}s elapsed.{" "}
          {elapsed > 20 && "Extraction calls a language model and can take a minute or more; please keep this page open."}
        </p>
      )}
      {runError && <p className="notice">{runError}</p>}

      {run && (
        <>
          <p className={BANNER[run.status].cls}>
            <strong>{BANNER[run.status].text}</strong>
            {run.failure && ` ${FAILURE_TEXT[run.failure] ?? run.failure}`}
            {run.run_id && <span className="hint"> Run {run.run_id}</span>}
          </p>
          <ul className="stage-list">
            {STAGES.map((s) => {
              const state = run.stages[s] ?? "skipped";
              return (
                <li key={s}>
                  <span className={`badge ${MARK[state].tone}`}>{MARK[state].symbol}</span> {s} <span className="hint">{state}</span>
                </li>
              );
            })}
          </ul>
          {run.errors.map((e) => (
            <p className="notice" key={e}>{e}</p>
          ))}
          {run.warnings.map((w) => (
            <p className="warning" key={w}>{w}</p>
          ))}
          {run.status === "failed" && run.validation && !run.validation.valid && (
            <p className="notice">Validation failed, so no results are shown: {run.validation.issues.filter((i) => i.severity === "error").map((i) => i.code).join(", ")}</p>
          )}
          {run.status !== "failed" && !run.result && <p className="notice">No results were produced for this bid.</p>}
          {run.result && (
            <>
              {run.result.unresolved.length > 0 && (
                <p className="warning">Unable to determine an authoritative value for: {run.result.unresolved.join(", ")}</p>
              )}
              <table className="extraction">
                <thead>
                  <tr><th>Field</th><th>Value</th><th>Confidence</th><th>Source</th></tr>
                </thead>
                <tbody>
                  {run.result.fields.map((f) => (
                    <FieldRow key={f.field} f={f} />
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}

      <h2 className="subhead">Ask a Question</h2>
      <form className="qa-form" onSubmit={ask}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is the final submission deadline?"
          aria-label="Question"
        />
        <button type="submit" disabled={asking || !bid || !question.trim()}>
          {asking ? "Asking..." : "Ask"}
        </button>
      </form>
      {askError && <p className="notice">{askError}</p>}
      {answer && (
        <article className="result">
          <h3>Answer</h3>
          {answer.answerable ? (
            <>
              <p className="answer">{answer.answer}</p>
              <div className="citation">
                Sources:
                <ul className="sources">
                  {answer.sources.map((s) => (
                    <li key={s.chunk_id}>{where(s)}</li>
                  ))}
                </ul>
              </div>
              <p className="hint">
                Confidence {answer.confidence.toFixed(2)} (heuristic, not a probability)
                {answer.used_reconciliation && " - uses the validated addendum reconciliation"}
              </p>
            </>
          ) : (
            <p className="answer">{REFUSAL}</p>
          )}
        </article>
      )}
    </section>
  );
}
