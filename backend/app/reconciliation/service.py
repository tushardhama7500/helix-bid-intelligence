"""Addendum reconciliation: gather original + addendum evidence per field, then decide."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings
from app.extraction.fields import FIELDS_BY_KEY
from app.extraction.schema import BidExtraction, Source
from app.extraction.service import LLMClient, UnknownBidError
from app.extraction.validation import build_source, grounding
from app.reconciliation.addenda import load_timeline
from app.reconciliation.detect import detect_changes, has_change_language, split_sentences, term_regex
from app.reconciliation.fields import RECONCILABLE, SKIPPED_FIELDS, ReconField
from app.reconciliation.llm import interpret_with_llm
from app.reconciliation.reconcile import ChangeRecord, OriginalValue, reconcile_field
from app.reconciliation.schema import AddendumEvent, BidReconciliation, ReconciledField, Timeline
from app.retrieval.schema import Candidate
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

TimelineLoader = Callable[[str], Awaitable[Timeline]]


class ReconciliationService:
    def __init__(
        self,
        retrieval: RetrievalService,
        settings: Settings,
        llm: LLMClient | None = None,
        timeline_loader: TimelineLoader | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.retrieval = retrieval
        self.settings = settings
        self.llm = llm
        self._timeline_loader = timeline_loader or (lambda bid_id: load_timeline(retrieval.session, bid_id))
        self._sleep = sleep

    async def reconcile_bid(
        self, bid_id: str, extraction: BidExtraction | None = None, keys: set[str] | None = None
    ) -> BidReconciliation:
        """Reconcile every reconcilable field, or only those in `keys` (e.g. for a single question)."""
        bids = (await self.retrieval.available_filters())["bids"]
        if bid_id not in bids:
            raise UnknownBidError(f"Unknown bid_id {bid_id!r}. Available: {', '.join(bids) or 'none'}")

        timeline = await self._timeline_loader(bid_id)
        errors: list[str] = []
        fields: list[ReconciledField] = []
        used_llm = False
        for rf in (r for r in RECONCILABLE if keys is None or r.key in keys):
            result, llm_used, error = await self._reconcile_one(bid_id, rf, timeline, extraction)
            fields.append(result)
            used_llm = used_llm or llm_used
            if error:
                errors.append(error)
            logger.info(
                "Reconciled bid=%s field=%s status=%s method=%s changes=%d",
                bid_id, rf.key, result.status, result.resolution_method, len(result.changes),
            )  # fmt: skip

        needs_attention = [
            f.field for f in fields
            if f.status == "conflict" or (f.status == "insufficient_evidence" and any(c.change_type == "unknown" for c in f.changes))
        ]  # fmt: skip
        return BidReconciliation(
            bid_id=bid_id,
            timeline=timeline,
            fields=fields,
            reconciliation_required=needs_attention,
            skipped_fields=dict(SKIPPED_FIELDS),
            errors=errors,
            complete=not errors,
            used_llm=used_llm,
            model=getattr(self.llm, "model", None),
        )

    # ------------------------------------------------------------------ one field
    async def _reconcile_one(
        self, bid_id: str, rf: ReconField, timeline: Timeline, extraction: BidExtraction | None
    ) -> tuple[ReconciledField, bool, str | None]:
        spec = FIELDS_BY_KEY[rf.key]
        original_chunks = await self._original_evidence(bid_id, spec.queries)
        originals = self._find_originals(rf, spec.name, original_chunks, extraction)

        by_addendum = await self._addendum_evidence(bid_id, spec.queries, timeline)
        label = term_regex(rf.terms)
        changes: list[ChangeRecord] = []
        ambiguous: dict[int, tuple[AddendumEvent, list[Candidate]]] = {}
        strong: set[int] = set()  # addenda where the rules saw explicit change wording they could not parse
        for event, chunks in by_addendum.values():
            for chunk in chunks:
                if not re.search(label, chunk.content, re.I):
                    continue
                detected = detect_changes(chunk.content, rf.terms)
                records = [self._to_record(d, event, chunk, rf.kind) for d in detected]
                changes.extend(records)
                # Ambiguous = change-like wording about this field that the rules could not turn into a value.
                unclear = any(r.change_type == "unknown" for r in records)
                if unclear:
                    strong.add(event.document_id)
                if not detected and any(
                    has_change_language(s) for s in split_sentences(chunk.content) if re.search(label, s, re.I)
                ):
                    unclear = True
                    changes.append(self._unknown_record(event, chunk))
                if unclear:
                    ambiguous.setdefault(event.document_id, (event, []))[1].append(chunk)

        llm_used, error = False, None
        if ambiguous:
            if self.llm is None:
                error = f"{spec.name}: addendum wording could not be interpreted deterministically and Groq is not configured"
            else:
                llm_used = True
                llm_changes, decided, llm_error = await interpret_with_llm(
                    self.llm,
                    field_name=spec.name, description=spec.description, kind=rf.kind,
                    original=original_chunks[:3],
                    addendum_chunks={k: (e, c[: self.settings.reconciliation_llm_max_chunks]) for k, (e, c) in ambiguous.items()},
                    char_limit=self.settings.extraction_chunk_char_limit,
                    max_tokens=self.settings.extraction_max_output_tokens,
                    sleep=self._sleep, strong_docs=strong,
                )  # fmt: skip
                if llm_error:
                    error = f"{spec.name}: {llm_error}"
                else:
                    # The model's reading replaces the unresolved 'unknown' records of those addenda.
                    changes = [c for c in changes if not (c.change_type == "unknown" and c.document_id in decided)]
                    changes.extend(llm_changes)

        result = reconcile_field(rf.key, spec.name, originals, changes, timeline, rf.kind)
        return result, llm_used, error

    # ------------------------------------------------------------------ evidence
    async def _original_evidence(self, bid_id: str, queries: tuple[str, ...]) -> list[Candidate]:
        seen: dict[int, Candidate] = {}
        for query in queries[:2]:
            found = await self.retrieval.retrieve(query, bid_id=bid_id, top_k=self.settings.extraction_evidence_per_field + 2)
            for r in found.results:
                c = r.candidate
                if c.bid_id == bid_id and c.document_type != "addendum":
                    seen.setdefault(c.chunk_id, c)
        return list(seen.values())

    async def _addendum_evidence(
        self, bid_id: str, queries: tuple[str, ...], timeline: Timeline
    ) -> dict[int, tuple[AddendumEvent, list[Candidate]]]:
        """Per addendum, scoped to this bid and that document: evidence cannot cross bids."""
        evidence: dict[int, tuple[AddendumEvent, list[Candidate]]] = {}
        for event in timeline.addenda:
            seen: dict[int, Candidate] = {}
            for query in queries[:2]:
                found = await self.retrieval.retrieve(
                    query, bid_id=bid_id, document_id=event.document_id,
                    top_k=self.settings.reconciliation_chunks_per_addendum,
                )  # fmt: skip
                for r in found.results:
                    c = r.candidate
                    if c.bid_id == bid_id and c.document_id == event.document_id:  # defence in depth
                        seen.setdefault(c.chunk_id, c)
            evidence[event.document_id] = (event, sorted(seen.values(), key=lambda c: c.chunk_index))
        return evidence

    # ----------------------------------------------------------------- originals
    def _find_originals(
        self, rf: ReconField, field_name: str, chunks: list[Candidate], extraction: BidExtraction | None
    ) -> list[OriginalValue]:
        found: list[OriginalValue] = []
        for chunk in chunks:
            if not chunk.file_name.lower().endswith(".pdf"):
                continue  # listing pages mirror later changes, so they are not "the original"
            text = " ".join(chunk.content.split())
            for pattern in rf.locators:
                m = re.search(pattern, text, re.I)
                if m:
                    found.append(OriginalValue(m.group("v").strip(), [build_source(chunk)]))
        if found:
            return found

        if extraction is None:
            return []
        f = extraction.fields.get(field_name)
        if f is None:
            return []
        if f.status == "found" and any(s.document_type != "addendum" for s in f.sources):
            return [OriginalValue(f.value, f.sources, f.confidence)]
        if f.status == "conflict":
            return [OriginalValue(c.value, c.sources) for c in f.conflicting_values if any(s.document_type != "addendum" for s in c.sources)]
        return []

    # ------------------------------------------------------------------- records
    @staticmethod
    def _unknown_record(event: AddendumEvent, chunk: Candidate) -> ChangeRecord:
        return ChangeRecord(
            change_type="unknown",
            explanation="The addendum contains wording that may modify this field, but its effect could not be determined.",
            sources=[build_source(chunk)], addendum_number=event.addendum_number, addendum_file=event.file_name,
            document_id=event.document_id, order=event.order,
        )  # fmt: skip

    @staticmethod
    def _to_record(detected: Any, event: AddendumEvent, chunk: Candidate, kind: str) -> ChangeRecord:
        change_type, new_value, explanation = detected.change_type, detected.new_value, detected.sentence
        if new_value and change_type in ("replacement", "addition") and grounding(new_value, [chunk], "scalar") is None:
            change_type, new_value = "unknown", None  # never accept a value the text does not contain
            explanation = "Detected value is not contained in the cited text: " + explanation
        if change_type == "unknown":
            explanation = "Ambiguous wording, not applied: " + detected.sentence
        label = {"replacement": "Replacement", "removal": "Removal", "addition": "Addition", "clarification": "Clarification"}.get(change_type)
        if label:
            explanation = f"{label} stated by the addendum: \"{detected.sentence}\""
        return ChangeRecord(
            change_type=change_type, explanation=explanation, sources=[build_source(chunk)],
            addendum_number=event.addendum_number, addendum_file=event.file_name, document_id=event.document_id,
            order=event.order, previous_value=detected.previous_value, new_value=new_value,
        )  # fmt: skip
