import StatusGrid from "../components/StatusGrid";
import { useSystemStatus } from "../services/useSystemStatus";

export default function Dashboard() {
  const status = useSystemStatus();
  return (
    <section>
      <h1>Helix</h1>
      <p className="tagline">AI-Powered Bid Intelligence</p>
      <StatusGrid status={status} />
    </section>
  );
}
