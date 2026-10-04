export interface HealthResponse {
  status: "ok" | "degraded";
  database: boolean;
  pgvector: boolean;
}

export interface GroqHealthResponse {
  status: "ok" | "unconfigured" | "error";
  configured: boolean;
  model: string | null;
  reachable: boolean | null;
  detail: string | null;
}
