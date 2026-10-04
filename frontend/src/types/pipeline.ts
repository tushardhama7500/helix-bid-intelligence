import type { ExtractionSource } from "./extraction";

export type StageState = "complete" | "partial" | "failed" | "skipped";

export interface ValidationIssue {
  severity: "error" | "warning";
  stage: string;
  field: string | null;
  code: string;
  message: string;
}

export interface ReportField {
  field: string;
  value: unknown;
  status: string;
  origin: "reconciliation" | "extraction" | "none";
  sources: ExtractionSource[];
  confidence: number; // heuristic, not a calibrated probability
  original_value: unknown;
  original_sources: ExtractionSource[];
  changed_by_addendum: boolean;
  notes: string | null;
}

export interface PipelineResponse {
  bid_id: string;
  run_id?: string;
  status: "complete" | "partial" | "failed";
  failure?: "infrastructure" | "no_documents" | "no_evidence" | "validation" | "internal" | null;
  stages: Record<string, StageState>;
  errors: string[];
  warnings: string[];
  validation: { valid: boolean; issues: ValidationIssue[] } | null;
  result: { bid_id: string; fields: ReportField[]; unresolved: string[] } | null;
}

export interface QASource {
  file: string;
  page: number | null;
  chunk_id: number;
  document_type: string;
  chunk_index: number;
  section: string | null;
}

export interface QAAnswer {
  bid_id: string;
  question: string;
  answer: string;
  answerable: boolean;
  sources: QASource[];
  confidence: number;
  used_reconciliation: boolean;
  notes: string | null;
}
