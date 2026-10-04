export interface ExtractionSource {
  chunk_id: number;
  file_name: string;
  document_type: string;
  page_number: number | null; // null for HTML
  chunk_index: number;
  section: string | null;
}

// A string, a list of strings, one product specification object, or a list of them (several products).
export type FieldValue = string | string[] | Record<string, unknown> | Record<string, unknown>[];

export interface ConflictingValue {
  value: FieldValue;
  sources: ExtractionSource[];
}

export interface ExtractedField {
  value: FieldValue | null;
  sources: ExtractionSource[];
  confidence: number; // an extraction-quality heuristic, not a calibrated probability
  notes: string | null;
  status: "found" | "not_found" | "conflict" | "unsupported" | "error";
  conflicting_values: ConflictingValue[];
}

export interface BidExtraction {
  bid_id: string;
  fields: Record<string, ExtractedField>;
  reconciliation_required: string[];
  errors: string[];
  complete: boolean;
  model: string | null;
}
