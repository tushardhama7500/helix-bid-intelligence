import { useCallback, useEffect, useState } from "react";

import type { GroqHealthResponse, HealthResponse } from "../types/health";
import { fetchGroqHealth, fetchHealth } from "./api";

export interface SystemStatus {
  loading: boolean;
  /** True when the API itself could not be reached. */
  apiDown: boolean;
  health: HealthResponse | null;
  groq: GroqHealthResponse | null;
  refresh: () => void;
}

export function useSystemStatus(): SystemStatus {
  const [loading, setLoading] = useState(true);
  const [apiDown, setApiDown] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [groq, setGroq] = useState<GroqHealthResponse | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    Promise.all([fetchHealth(), fetchGroqHealth()])
      .then(([h, g]) => {
        setHealth(h);
        setGroq(g);
        setApiDown(false);
      })
      .catch(() => {
        setHealth(null);
        setGroq(null);
        setApiDown(true);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(refresh, [refresh]);

  return { loading, apiDown, health, groq, refresh };
}
