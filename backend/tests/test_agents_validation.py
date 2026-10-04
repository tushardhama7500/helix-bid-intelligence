"""Validation agent: detects bad output and never repairs it."""

import pytest

from app.agents.validation_agent import ChunkRecord, ValidationAgent
from app.extraction.schema import BidExtraction, ConflictingValue, ExtractedField, Source
from app.extraction.fields import FIELDS
from app.reconciliation.reconcile import ChangeRecord, OriginalValue, reconcile_field
from app.reconciliation.schema import BidReconciliation, ReconciledField, ReconciliationChange, Timeline
from app.reconciliation.addenda import build_event, build_timeline

RFP = ChunkRecord(1, "Bid1", "RFP.pdf", "bid_document", 2, 1, "Solicitation Due 27-JUN-2024 14:00:00")
ADD = ChunkRecord(2, "Bid1", "Addendum 2.pdf", "addendum", 1, 0, "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.")
OTHER = ChunkRecord(9, "Bid2", "PORFP.pdf", "rfp", 1, 0, "Delivery within 45 days of Award. Mercury Affidavit.")
STORE = {r.chunk_id: r for r in (RFP, ADD, OTHER)}


async def lookup(ids):
    return {i: STORE[i] for i in ids if i in STORE}


def src(rec: ChunkRecord, **override) -> Source:
    data = dict(chunk_id=rec.chunk_id, file_name=rec.file_name, document_type=rec.document_type, page_number=rec.page_number, chunk_index=rec.chunk_index)
    return Source(**{**data, **override})


def timeline() -> Timeline:
    ev = build_event(document_id=2, file_name="Addendum 2.pdf", document_type="addendum", bid_id="Bid1", first_page="ADDENDUM No. 2", full_text="")
    return build_timeline("Bid1", [ev])


def resolved_due_date() -> ReconciledField:
    changes = [ChangeRecord(
        change_type="replacement", explanation="x", sources=[src(ADD)], addendum_number=2, addendum_file="Addendum 2.pdf",
        document_id=2, order=1, new_value="July 9, 2024 at 2:00 PM CST",
    )]  # fmt: skip
    return reconcile_field("due_date", "Due Date", [OriginalValue("27-JUN-2024 14:00:00", [src(RFP)])], changes, timeline())


def recon(*fields) -> BidReconciliation:
    # model_construct: some tests deliberately build output that violates the schema.
    return BidReconciliation.model_construct(bid_id="Bid1", timeline=timeline(), fields=list(fields))


def codes(report) -> set[str]:
    return {i.code for i in report.issues if i.severity == "error"}


async def validate(extraction=None, reconciliation=None, bid="Bid1"):
    return await ValidationAgent(lookup).validate(bid, extraction, reconciliation)


async def test_valid_reconciliation_passes() -> None:
    report = await validate(reconciliation=recon(resolved_due_date()))
    assert report.valid and report.issues == []
    assert report.checked["citations"] == 3 and report.checked["unique_chunks"] == 2


async def test_missing_citation_chunk_is_an_error() -> None:
    f = resolved_due_date()
    f.sources = [Source(chunk_id=999, file_name="ghost.pdf", document_type="addendum", page_number=1, chunk_index=0)]
    report = await validate(reconciliation=recon(f))
    assert not report.valid and "CITATION_NOT_FOUND" in codes(report)


async def test_final_value_without_any_citation_is_an_error() -> None:
    f = ReconciledField.model_construct(
        field="Due Date", key="due_date", original_value="27-JUN-2024", original_sources=[], changes=[], final_value="July 9",
        sources=[], conflicting_values=[], status="unchanged", resolution_method="no_change", confidence=0.7, notes=None,
    )  # fmt: skip
    report = await validate(reconciliation=recon(f))
    assert not report.valid and {"SCHEMA_INVALID"} <= codes(report)


async def test_citation_from_another_bid_is_an_error() -> None:
    f = resolved_due_date()
    f.sources = [src(OTHER)]
    report = await validate(reconciliation=recon(f))
    assert not report.valid and "CITATION_WRONG_BID" in codes(report)


async def test_citation_metadata_must_match_the_stored_chunk() -> None:
    f = resolved_due_date()
    f.original_sources = [src(RFP, page_number=7)]
    assert "CITATION_METADATA_MISMATCH" in codes(await validate(reconciliation=recon(f)))
    f.original_sources = [src(RFP, file_name="Other.pdf")]
    assert "CITATION_METADATA_MISMATCH" in codes(await validate(reconciliation=recon(f)))


async def test_unsupported_final_value_is_an_error_and_is_not_repaired() -> None:
    f = resolved_due_date()
    f.final_value = "July 10, 2024 at 2:00 PM CST"  # the cited text says July 9
    f.changes[0].new_value = "July 10, 2024 at 2:00 PM CST"
    report = await validate(reconciliation=recon(f))

    assert not report.valid and "FINAL_VALUE_NOT_GROUNDED" in codes(report)
    assert f.final_value == "July 10, 2024 at 2:00 PM CST"  # reported, never altered


async def test_final_value_that_differs_from_the_last_change_is_an_error() -> None:
    f = resolved_due_date()
    f.final_value = "27-JUN-2024 14:00:00"
    f.sources = [src(RFP)]
    report = await validate(reconciliation=recon(f))
    assert "FINAL_DIFFERS_FROM_LAST_CHANGE" in codes(report) and "FINAL_NOT_CITING_CHANGE" in codes(report)


async def test_history_must_start_from_the_original_value() -> None:
    f = resolved_due_date()
    f.original_value = "30-JUN-2024 12:00:00"
    assert "ORIGINAL_NOT_PRESERVED" in codes(await validate(reconciliation=recon(f)))
    f = resolved_due_date()
    f.original_sources = []
    assert "ORIGINAL_UNCITED" in codes(await validate(reconciliation=recon(f)))


async def test_change_must_come_from_a_timeline_addendum() -> None:
    f = resolved_due_date()
    f.changes[0].addendum_file = "Addendum 9.pdf"
    assert "CHANGE_FROM_UNKNOWN_ADDENDUM" in codes(await validate(reconciliation=recon(f)))


async def test_conflict_with_a_final_value_is_an_error() -> None:
    f = ReconciledField.model_construct(
        field="Due Date", key="due_date", original_value=None, original_sources=[], changes=[], final_value="July 9",
        sources=[src(ADD)], conflicting_values=[], status="conflict", resolution_method="unresolved", confidence=0.0, notes=None,
    )  # fmt: skip
    codes_ = codes(await validate(reconciliation=recon(f)))
    assert "UNRESOLVED_HAS_FINAL_VALUE" in codes_


async def test_genuine_unresolved_conflict_is_valid_but_warned() -> None:
    f = reconcile_field("due_date", "Due Date", [OriginalValue("A", [src(RFP)]), OriginalValue("B", [src(ADD)])], [], timeline())
    report = await validate(reconciliation=recon(f))
    assert report.valid and [w.code for w in report.warnings] == ["UNRESOLVED_CONFLICT"]


async def test_no_data_fields_do_not_produce_warnings() -> None:
    f = reconcile_field("bid_bond", "Bid Bond Requirement", [], [], timeline())
    report = await validate(reconciliation=recon(f))
    assert report.valid and report.issues == []


# ------------------------------------------------------------------ extraction
def extraction(**overrides) -> BidExtraction:
    fields = {s.name: ExtractedField.missing() for s in FIELDS}
    fields.update(overrides)
    return BidExtraction.model_construct(bid_id="Bid1", fields=fields)


async def test_valid_extraction_passes() -> None:
    ex = extraction(**{"Due Date": ExtractedField(value="27-JUN-2024 14:00:00", sources=[src(RFP)], confidence=0.9, status="found")})
    assert (await validate(extraction=ex)).valid


async def test_extraction_missing_fields_and_ungrounded_values_are_errors() -> None:
    ex = BidExtraction.model_construct(bid_id="Bid1", fields={"Due Date": ExtractedField(value="1-JAN-2030", sources=[src(RFP)], confidence=0.9, status="found")})
    c = codes(await validate(extraction=ex))
    assert "FIELD_MISSING" in c and "VALUE_NOT_GROUNDED" in c


async def test_extraction_missing_value_consistency() -> None:
    bad = ExtractedField.model_construct(value="x", sources=[src(RFP)], confidence=0.4, notes=None, status="not_found", conflicting_values=[])
    c = codes(await validate(extraction=extraction(**{"Title": bad})))
    assert "SCHEMA_INVALID" in c or "MISSING_VALUE_INCONSISTENT" in c


async def test_extraction_conflict_with_value_is_an_error() -> None:
    bad = ExtractedField.model_construct(
        value="July 9", sources=[src(ADD)], confidence=0.5, notes=None, status="conflict",
        conflicting_values=[ConflictingValue(value="a", sources=[src(RFP)]), ConflictingValue(value="b", sources=[src(ADD)])],
    )  # fmt: skip
    assert "SCHEMA_INVALID" in codes(await validate(extraction=extraction(**{"Due Date": bad})))


async def test_extraction_citation_from_another_bid() -> None:
    ex = extraction(**{"Delivery Date": ExtractedField(value="Delivery within 45 days of Award", sources=[src(OTHER)], confidence=0.9, status="found")})
    assert "CITATION_WRONG_BID" in codes(await validate(extraction=ex))
