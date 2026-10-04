import type { ConflictingValue, ExtractionSource } from "./extraction";

export type ReconStatus = "resolved" | "unchanged" | "conflict" | "insufficient_evidence";
export type ReconMethod =
  | "explicit_addendum_change"
  | "ordered_addendum"
  | "llm_reconciliation"
  | "no_change"
  | "unresolved";

export interface ReconciliationChange {
  addendum_number: number | null;
  addendum_file: string | null;
  change_type: "replacement" | "clarification" | "addition" | "removal" | "unknown";
  previous_value: unknown;
  new_value: unknown;
  explanation: string;
  sources: ExtractionSource[];
  method: "rule" | "llm";
}

export interface ReconciledField {
  field: string;
  key: string;
  original_value: unknown;
  original_sources: ExtractionSource[];
  changes: ReconciliationChange[];
  final_value: unknown;
  sources: ExtractionSource[];
  conflicting_values: ConflictingValue[];
  status: ReconStatus;
  resolution_method: ReconMethod;
  confidence: number; // heuristic, not a calibrated probability
  notes: string | null;
}

export interface AddendumEvent {
  document_id: number;
  file_name: string;
  addendum_number: number | null;
  number_source: "content" | "filename" | null;
  date: string | null;
  order: number | null;
}

export interface BidReconciliation {
  bid_id: string;
  timeline: { addenda: AddendumEvent[]; ordering: string; reliable: boolean; notes: string[]; original_documents: string[] };
  fields: ReconciledField[];
  reconciliation_required: string[];
  skipped_fields: Record<string, string>;
  errors: string[];
  complete: boolean;
  used_llm: boolean;
  model: string | null;
}
