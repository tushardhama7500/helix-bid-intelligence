export type Tone = "ok" | "bad" | "warn" | "unknown";

interface Props {
  title: string;
  tone: Tone;
  label: string;
  detail?: string;
}

export default function StatusCard({ title, tone, label, detail }: Props) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <span className={`badge ${tone}`}>{label}</span>
      {detail && <p className="detail">{detail}</p>}
    </div>
  );
}
