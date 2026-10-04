import StatusGrid from "../components/StatusGrid";
import { useSystemStatus } from "../services/useSystemStatus";

export default function SystemHealth() {
  const status = useSystemStatus();
  return (
    <section>
      <h1>System Health</h1>
      <p className="tagline">Live status of Helix services.</p>
      <StatusGrid status={status} />
      <button onClick={status.refresh} disabled={status.loading}>
        {status.loading ? "Checking..." : "Refresh"}
      </button>
    </section>
  );
}
