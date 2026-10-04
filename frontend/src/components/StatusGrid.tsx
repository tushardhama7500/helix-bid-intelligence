import type { SystemStatus } from "../services/useSystemStatus";
import StatusCard, { type Tone } from "./StatusCard";

export default function StatusGrid({ status }: { status: SystemStatus }) {
  const { loading, apiDown, health, groq } = status;

  const pendingTone: Tone = loading ? "unknown" : "bad";
  const pendingLabel = loading ? "Checking..." : "Unavailable";

  const apiTone: Tone = health ? "ok" : pendingTone;
  const apiLabel = health ? "Online" : pendingLabel;

  let dbTone: Tone = pendingTone;
  let dbLabel = pendingLabel;
  let dbDetail: string | undefined;
  if (health) {
    dbTone = health.database ? (health.pgvector ? "ok" : "warn") : "bad";
    dbLabel = health.database ? "Connected" : "Disconnected";
    dbDetail = health.database
      ? `pgvector: ${health.pgvector ? "installed" : "missing"}`
      : undefined;
  }

  let groqTone: Tone = pendingTone;
  let groqLabel = pendingLabel;
  let groqDetail: string | undefined;
  if (groq) {
    groqTone = groq.status === "ok" ? "ok" : groq.status === "error" ? "bad" : "warn";
    groqLabel = groq.configured ? "Configured" : "Not configured";
    groqDetail = groq.model ?? groq.detail ?? undefined;
  }

  return (
    <>
      {apiDown && !loading && (
        <p className="notice">Cannot reach the Helix API. Is the backend running?</p>
      )}
      <div className="grid">
        <StatusCard title="API Status" tone={apiTone} label={apiLabel} />
        <StatusCard title="Database Status" tone={dbTone} label={dbLabel} detail={dbDetail} />
        <StatusCard title="Groq Status" tone={groqTone} label={groqLabel} detail={groqDetail} />
      </div>
    </>
  );
}
