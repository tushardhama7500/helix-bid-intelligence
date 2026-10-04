"""Targeted extraction fixes (Bid1 report review): bid-number scope, verbatim specs, multi-product specs.

Scripted model only; checks grounding and shape, never one model's wording.
"""

import pytest

from app.core.config import Settings
from app.extraction.fields import FIELDS_BY_KEY
from app.extraction.prompts import SYSTEM_PROMPT
from app.extraction.validation import MAX_SPEC_PRODUCTS, normalize_value
from tests.extraction_fakes import FakeLLM, StubRetrieval, cand
from tests.test_extraction_service import service

RFP = cand(1, "JA-207652 FINAL.pdf", "Request For Proposal 168884 JA-207652 Student and Staff Computing Devices. Solicitation Due 27-JUN-2024.", page=2)
LISTING = cand(2, "Student and Staff Computing Devices - BidNet.html", "Reference Number 00004079100 Issuing Organization Dallas ISD Solicitation Number JA-207652 SOURCING #168884", doc_type="bid_document", page=None)
WIN_LAPTOP = cand(
    3, "JA-207652 FINAL.pdf",
    "Tier 3 Student Windows Laptop Minimum Requirements Wireless: 802.11AX WiFi-6E and Bluetooth 5.1 "
    "Power Supply: Type C, 65W Ports, Minimum 2 USB 3.0; 1 Type C; 65HDMI; 1 headphone/microphone combo "
    "Storage: 256 GB NVMe SSD Windows 11 Pro Warranty: 3-year minimum",
    page=5,
)
CHROMEBOOK = cand(
    4, "JA-207652 FINAL.pdf",
    "Tier 1 Small Student Chromebook Laptop Minimum Requirements Display: Minimum 11.6 inch "
    "Memory: Minimum 4GB RAM Storage: 32 GB eMMC Chrome O/S License Required Warranty: 1-year minimum",
    page=5,
)
CORPUS = {"Bid1": [RFP, LISTING, WIN_LAPTOP, CHROMEBOOK]}


def extract(answers):
    return service(llm=FakeLLM(answers=answers), retrieval=StubRetrieval(CORPUS)).extract_bid("Bid1")


def spec(product, **attrs):
    return {"product": product, "specifications": attrs}


# ------------------------------------------------------------------ 1. bid number
def test_bid_number_description_separates_solicitation_number_from_other_identifiers() -> None:
    text = FIELDS_BY_KEY["bid_number"].description
    assert "solicitation's own number" in text and "Sourcing #" in text and "Reference Number" in text
    assert "notes" in text  # other identifiers are mentioned there, not in the value


async def test_bid_number_keeps_only_the_cited_solicitation_number_and_the_note() -> None:
    answers = {"bid_number": ([{"value": ["JA-207652"], "chunk_ids": [1, 2]}], 0.95)}
    llm = FakeLLM(answers=answers)
    result = await service(llm=llm, retrieval=StubRetrieval(CORPUS)).extract_bid("Bid1")
    field = result.fields["Bid Number"]
    assert field.status == "found" and field.value == ["JA-207652"]
    assert {s.chunk_id for s in field.sources} == {1, 2}


async def test_other_identifier_is_rejected_when_cited_text_does_not_contain_it() -> None:
    answers = {"bid_number": ([{"value": ["JA-207652", "BPM044557"], "chunk_ids": [1]}], 0.9)}
    result = await extract(answers)
    # A single candidate whose value is not entirely supported is refused rather than partly kept.
    assert result.fields["Bid Number"].value is None
    assert result.fields["Bid Number"].status == "unsupported"


async def test_reference_and_sourcing_numbers_are_still_groundable_if_the_model_returns_them() -> None:
    """Scoping is a prompt/field-definition rule; grounding still guarantees nothing unsupported appears."""
    answers = {"bid_number": ([{"value": ["00004079100"], "chunk_ids": [2]}], 0.9)}
    field = (await extract(answers)).fields["Bid Number"]
    assert field.value == ["00004079100"] and field.sources[0].chunk_id == 2  # grounded, cited


# ------------------------------------------------------------------ 2. verbatim specs
def test_source_really_contains_65hdmi_and_normalisation_preserves_it() -> None:
    assert "65HDMI" in WIN_LAPTOP.content  # as in the real FINAL.pdf (checked in the regression cases)
    value = normalize_value({"product": "Student Laptop", "specifications": {"ports": "Minimum 2 USB 3.0; 1 Type C; 65HDMI; 1 headphone/microphone combo"}}, "spec")
    assert "65HDMI" in value["specifications"]["ports"]


async def test_spec_value_65hdmi_is_kept_verbatim_with_its_citation() -> None:
    ports = "Minimum 2 USB 3.0; 1 Type C; 65HDMI; 1 headphone/microphone combo"
    answers = {"product_specification": ([{"value": spec("Student Windows Laptop", ports=ports, storage="256 GB NVMe SSD"), "chunk_ids": [3]}], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert field.status == "found" and field.value["specifications"]["ports"] == ports
    assert [s.chunk_id for s in field.sources] == [3]


def test_prompt_forbids_correcting_apparent_typos() -> None:
    assert "Never correct" in SYSTEM_PROMPT and "typo" in SYSTEM_PROMPT
    assert "exact wording" in FIELDS_BY_KEY["product_specification"].description


# ------------------------------------------------------------------ 3. multiple products
async def test_several_products_become_a_list_each_grounded_in_its_own_chunk() -> None:
    answers = {"product_specification": ([
        {"value": spec("Tier 3 Student Windows Laptop", storage="256 GB NVMe SSD", warranty="3-year minimum"), "chunk_ids": [3]},
        {"value": spec("Tier 1 Small Student Chromebook Laptop", memory="Minimum 4GB RAM", storage="32 GB eMMC"), "chunk_ids": [4]},
    ], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert field.status == "found" and isinstance(field.value, list) and len(field.value) == 2
    assert [p["product"] for p in field.value] == ["Tier 3 Student Windows Laptop", "Tier 1 Small Student Chromebook Laptop"]
    assert {s.chunk_id for s in field.sources} == {3, 4}
    assert field.conflicting_values == []  # different products are not a conflict


async def test_one_products_figures_cannot_be_borrowed_from_another_products_chunk() -> None:
    """The Chromebook's '4GB' is in chunk 4 only; citing the Windows laptop chunk for it must fail."""
    answers = {"product_specification": ([
        {"value": spec("Tier 3 Student Windows Laptop", storage="256 GB NVMe SSD"), "chunk_ids": [3]},
        {"value": spec("Tier 1 Small Student Chromebook Laptop", memory="Minimum 4GB RAM"), "chunk_ids": [3]},
    ], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert isinstance(field.value, dict) and field.value["product"] == "Tier 3 Student Windows Laptop"
    assert "4GB" not in str(field.value)  # the unsupported product was dropped, nothing fabricated


async def test_a_single_product_keeps_the_original_single_object_shape() -> None:
    answers = {"product_specification": ([{"value": spec("Student Laptop", storage="256 GB NVMe SSD"), "chunk_ids": [3]}], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert isinstance(field.value, dict) and field.value["specifications"]["storage"] == "256 GB NVMe SSD"


async def test_unsupported_spec_is_refused_and_invented_spec_never_appears() -> None:
    answers = {"product_specification": ([{"value": spec("Student Laptop", memory="64GB RAM", gpu="RTX 4090"), "chunk_ids": [3]}], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert field.value is None and field.status == "unsupported" and field.sources == [] and field.confidence == 0.0


async def test_spec_without_valid_citation_is_refused() -> None:
    answers = {"product_specification": ([{"value": spec("Student Laptop", storage="256 GB NVMe SSD"), "chunk_ids": [999]}], 0.9)}
    field = (await extract(answers)).fields["Product Specification"]
    assert field.value is None and field.status == "unsupported"


async def test_spec_list_is_bounded() -> None:
    chunks = [cand(100 + i, "JA-207652 FINAL.pdf", f"Model {i} Minimum Requirements Storage: {i + 1} GB SSD", page=6) for i in range(MAX_SPEC_PRODUCTS + 4)]
    candidates = [{"value": spec(f"Model {i}", storage=f"{i + 1} GB SSD"), "chunk_ids": [100 + i]} for i in range(len(chunks))]
    llm = FakeLLM(answers={"product_specification": (candidates, 0.9)})
    wide = Settings(_env_file=None, database_url="postgresql+psycopg://x/x", extraction_evidence_per_field=20, extraction_max_chunks_per_call=20)
    result = await service(llm=llm, retrieval=StubRetrieval({"Bid1": chunks}), settings=wide).extract_bid("Bid1")
    assert len(result.fields["Product Specification"].value) == MAX_SPEC_PRODUCTS


# ------------------------------------------------------------------ 4. missing values, 5. confidence
MISSING = ["Delivery Date", "MFG for Registration", "Model_no", "Part_no", "Contract or Cooperative to use"]


async def test_fields_without_evidence_stay_missing_with_no_sources_and_zero_confidence() -> None:
    result = await extract({"bid_number": ([{"value": ["JA-207652"], "chunk_ids": [1]}], 0.9)})
    for name in MISSING:
        f = result.fields[name]
        assert (f.value, f.sources, f.confidence, f.status) == (None, [], 0.0, "not_found"), name
        assert f.notes


async def test_confidence_is_never_raised_above_the_models_estimate() -> None:
    bond = cand(5, "JA-207652 FINAL.pdf", "A bid bond is required of each Offeror.", page=9)
    answers = {"bid_bond": ([{"value": "Bid bond required", "chunk_ids": [5]}], 0.63)}
    result = await service(llm=FakeLLM(answers=answers), retrieval=StubRetrieval({"Bid1": [bond]})).extract_bid("Bid1")
    assert result.fields["Bid Bond Requirement"].confidence == pytest.approx(0.63)


# ------------------------------------------------------------------ evidence depth for multi-product specs
def test_deeper_evidence_for_one_field_is_not_crowded_out_by_the_others() -> None:
    from app.extraction.evidence import select_group_evidence

    others = [cand(i, "a.pdf", f"other {i}", page=1) for i in range(1, 11)]
    specs = [cand(100 + i, "a.pdf", f"spec {i}", page=2) for i in range(8)]
    evidence = {"model_no": others[:5], "part_no": others[5:], "product_specification": specs}

    plain, allowed_plain = select_group_evidence(evidence, 10)
    widened, allowed = select_group_evidence(evidence, 10, {"product_specification": 4})
    assert len(allowed["product_specification"]) == len(allowed_plain["product_specification"]) + 4
    assert len(widened) == len(plain) + 4  # the group grows only by the extra the field asked for
    assert set(allowed["model_no"]) == set(allowed_plain["model_no"])  # other fields are unaffected


def test_only_the_spec_field_asks_for_deeper_evidence() -> None:
    from app.extraction.fields import FIELDS

    assert {f.key: f.evidence_limit for f in FIELDS if f.evidence_limit} == {"product_specification": 8}
