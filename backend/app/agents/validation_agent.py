"""Validation agent: checks the pipeline's output against the database. Reports, never repairs."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schema import ValidationIssue, ValidationReport
from app.agents.tracing import agent_span
from app.extraction.fields import FIELDS
from app.extraction.schema import BidExtraction, ExtractedField, Source
from app.extraction.validation import comparison_key, flatten, grounding
from app.reconciliation.fields import RECONCILABLE_BY_KEY
from app.reconciliation.schema import BidReconciliation, ReconciledField
from app.retrieval.schema import Candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    bid_id: str
    file_name: str
    document_type: str
    page_number: int | None
    chunk_index: int
    content: str

    def as_candidate(self) -> Candidate:
        return Candidate(
            chunk_id=self.chunk_id, content=self.content, document_id=0, bid_id=self.bid_id, file_name=self.file_name,
            document_type=self.document_type, page_number=self.page_number, chunk_index=self.chunk_index,
            metadata={}, score=0.0,
        )  # fmt: skip


ChunkLookup = Callable[[Sequence[int]], Awaitable[dict[int, ChunkRecord]]]

_LOOKUP_SQL = text(
    "SELECT c.id, c.bid_id, d.file_name, d.document_type, p.page_number, c.chunk_index, c.content "
    "FROM document_chunks c JOIN documents d ON d.id = c.document_id JOIN document_pages p ON p.id = c.page_id "
    "WHERE c.id IN :ids"
).bindparams(bindparam("ids", expanding=True))


def db_chunk_lookup(session: AsyncSession) -> ChunkLookup:
    async def lookup(ids: Sequence[int]) -> dict[int, ChunkRecord]:
        if not ids:
            return {}
        rows = (await session.execute(_LOOKUP_SQL, {"ids": list(ids)})).all()
        return {r[0]: ChunkRecord(*r) for r in rows}

    return lookup


class _Issues:
    def __init__(self) -> None:
        self.items: list[ValidationIssue] = []

    def error(self, stage: str, field: str | None, code: str, message: str) -> None:
        self.items.append(ValidationIssue(severity="error", stage=stage, field=field, code=code, message=message))  # type: ignore[arg-type]

    def warn(self, stage: str, field: str | None, code: str, message: str) -> None:
        self.items.append(ValidationIssue(severity="warning", stage=stage, field=field, code=code, message=message))  # type: ignore[arg-type]


class ValidationAgent:
    """Verifies schema, citation existence and provenance, grounding, and cross-stage consistency."""

    def __init__(self, lookup: ChunkLookup) -> None:
        self.lookup = lookup

    async def validate(
        self, bid_id: str, extraction: BidExtraction | None, reconciliation: BidReconciliation | None
    ) -> ValidationReport:
        async with agent_span("validation", bid_id, "validate") as span:
            issues = _Issues()
            sources = _collect_sources(extraction, reconciliation)
            records = await self.lookup(sorted({s.chunk_id for _, _, s in sources}))
            self._check_sources(bid_id, sources, records, issues)
            if extraction is not None:
                self._check_extraction(extraction, records, issues)
            if reconciliation is not None:
                self._check_reconciliation(reconciliation, records, issues)

            report = ValidationReport(
                valid=not any(i.severity == "error" for i in issues.items),
                issues=issues.items,
                checked={
                    "citations": len(sources),
                    "unique_chunks": len(records),
                    "extraction_fields": len(extraction.fields) if extraction else 0,
                    "reconciled_fields": len(reconciliation.fields) if reconciliation else 0,
                },
            )
            span["note"] = f"valid={report.valid} errors={len(report.errors)} warnings={len(report.warnings)}"
            return report

    # ------------------------------------------------------------ citations
    def _check_sources(self, bid_id: str, sources: list, records: dict[int, ChunkRecord], issues: _Issues) -> None:
        for stage, field, s in sources:
            rec = records.get(s.chunk_id)
            if rec is None:
                issues.error(stage, field, "CITATION_NOT_FOUND", f"Cited chunk {s.chunk_id} does not exist.")
            elif rec.bid_id != bid_id:
                issues.error(stage, field, "CITATION_WRONG_BID", f"Cited chunk {s.chunk_id} belongs to {rec.bid_id}, not {bid_id}.")
            elif (rec.file_name, rec.document_type, rec.page_number, rec.chunk_index) != (
                s.file_name, s.document_type, s.page_number, s.chunk_index
            ):  # fmt: skip
                issues.error(stage, field, "CITATION_METADATA_MISMATCH", f"Cited chunk {s.chunk_id} does not match the stored file/page/chunk.")

    # ----------------------------------------------------------- extraction
    def _check_extraction(self, extraction: BidExtraction, records: dict[int, ChunkRecord], issues: _Issues) -> None:
        for spec in FIELDS:
            if spec.name not in extraction.fields:
                issues.error("extraction", spec.name, "FIELD_MISSING", "Required field is absent from the extraction.")
        for name, f in extraction.fields.items():
            if not _roundtrips(ExtractedField, f):
                issues.error("extraction", name, "SCHEMA_INVALID", "Field does not satisfy the extraction schema.")
                continue
            if f.status == "found":
                if f.value is None or not f.sources:
                    issues.error("extraction", name, "FOUND_WITHOUT_VALUE_OR_CITATION", "A found field needs a value and a citation.")
                    continue
                kind = next((s.kind for s in FIELDS if s.name == name), "scalar")
                cited = [records[s.chunk_id].as_candidate() for s in f.sources if s.chunk_id in records]
                if cited and grounding(f.value, cited, kind) is None:
                    issues.error("extraction", name, "VALUE_NOT_GROUNDED", "The value is not supported by the cited evidence.")
            elif f.status == "conflict":
                if f.value is not None or f.confidence != 0:
                    issues.error("extraction", name, "CONFLICT_HAS_VALUE", "A conflicting field must have value=null and confidence 0.")
            else:  # not_found / unsupported / error
                if f.value is not None or f.sources or f.confidence != 0:
                    issues.error("extraction", name, "MISSING_VALUE_INCONSISTENT", "A missing field must have value=null, no sources and confidence 0.")
            if f.status in ("unsupported", "error"):
                issues.warn("extraction", name, f"EXTRACTION_{f.status.upper()}", f.notes or f.status)

    # -------------------------------------------------------- reconciliation
    def _check_reconciliation(self, rec: BidReconciliation, records: dict[int, ChunkRecord], issues: _Issues) -> None:
        addenda_files = {e.file_name for e in rec.timeline.addenda}
        for f in rec.fields:
            if f.status in ("conflict", "insufficient_evidence") and (f.final_value is not None or f.confidence != 0):
                issues.error("reconciliation", f.field, "UNRESOLVED_HAS_FINAL_VALUE", f"Status '{f.status}' requires final_value=null and confidence 0.")
            if not _roundtrips(ReconciledField, f):
                issues.error("reconciliation", f.field, "SCHEMA_INVALID", "Field does not satisfy the reconciliation schema.")
                continue
            kind = RECONCILABLE_BY_KEY[f.key].kind if f.key in RECONCILABLE_BY_KEY else "scalar"
            if f.status in ("conflict", "insufficient_evidence"):
                # "No data for this field" is normal; only real conflicts / uninterpretable wording are flagged.
                if f.status == "conflict" or any(c.change_type == "unknown" for c in f.changes):
                    issues.warn("reconciliation", f.field, f"UNRESOLVED_{f.status.upper()}", f.notes or f.status)
                continue

            if f.final_value is not None:
                cited = [records[s.chunk_id].as_candidate() for s in f.sources if s.chunk_id in records]
                if not f.sources:
                    issues.error("reconciliation", f.field, "FINAL_VALUE_UNCITED", "The final value has no citation.")
                elif cited and grounding(f.final_value, cited, "list" if kind == "list" else "scalar") is None:
                    issues.error("reconciliation", f.field, "FINAL_VALUE_NOT_GROUNDED", "The final value is not supported by the cited evidence.")

            if f.status == "resolved":
                self._check_resolved(f, kind, addenda_files, issues)
            elif f.status == "unchanged" and kind != "list" and f.original_value is not None:
                if comparison_key(f.final_value) != comparison_key(f.original_value):
                    issues.error("reconciliation", f.field, "UNCHANGED_VALUE_DIFFERS", "An unchanged field's final value differs from its original value.")

    def _check_resolved(self, f: ReconciledField, kind: str, addenda_files: set[str], issues: _Issues) -> None:
        value_changes = [c for c in f.changes if c.change_type in ("replacement", "removal")]
        if not value_changes:
            issues.error("reconciliation", f.field, "RESOLVED_WITHOUT_CHANGE", "A resolved field has no value-changing addendum.")
            return
        last = value_changes[-1]
        expected = None if last.change_type == "removal" else ([last.new_value] if kind == "list" else last.new_value)
        if comparison_key(expected) != comparison_key(f.final_value):
            issues.error("reconciliation", f.field, "FINAL_DIFFERS_FROM_LAST_CHANGE", "The final value is not the value the last addendum change set.")
        if last.change_type == "replacement":
            cited_ids = {s.chunk_id for s in f.sources}
            if not {s.chunk_id for s in last.sources} <= cited_ids:
                issues.error("reconciliation", f.field, "FINAL_NOT_CITING_CHANGE", "The final value does not cite the addendum that set it.")
        if last.addendum_file is not None and addenda_files and last.addendum_file not in addenda_files:
            issues.error("reconciliation", f.field, "CHANGE_FROM_UNKNOWN_ADDENDUM", "A change cites a document that is not in this bid's addendum timeline.")
        if any(s.file_name != last.addendum_file for s in last.sources if last.addendum_file):
            issues.error("reconciliation", f.field, "CHANGE_SOURCE_NOT_ADDENDUM", "A change cites text from a document other than its addendum.")
        if f.original_value is not None:
            first = value_changes[0]
            if comparison_key(first.previous_value) != comparison_key(f.original_value):
                issues.error("reconciliation", f.field, "ORIGINAL_NOT_PRESERVED", "The change history does not start from the original value.")
            if not f.original_sources:
                issues.error("reconciliation", f.field, "ORIGINAL_UNCITED", "The original value has no citation.")


def _roundtrips(model: type, instance: Any) -> bool:
    try:
        model.model_validate(instance.model_dump())
        return True
    except (ValidationError, ValueError):
        return False


def _collect_sources(extraction: BidExtraction | None, rec: BidReconciliation | None) -> list[tuple[str, str, Source]]:
    out: list[tuple[str, str, Source]] = []
    if extraction is not None:
        for name, f in extraction.fields.items():
            out += [("extraction", name, s) for s in f.sources]
            out += [("extraction", name, s) for c in f.conflicting_values for s in c.sources]
    if rec is not None:
        for f in rec.fields:
            for s in [*f.original_sources, *f.sources, *(x for c in f.changes for x in c.sources),
                      *(x for c in f.conflicting_values for x in c.sources)]:  # fmt: skip
                out.append(("reconciliation", f.field, s))
    return out
