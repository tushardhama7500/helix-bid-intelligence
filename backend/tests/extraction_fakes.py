"""Test doubles for extraction: a scripted Groq client and an in-memory retrieval service."""

import json
import re
from dataclasses import dataclass

from app.retrieval.schema import Candidate, FusedResult


def cand(chunk_id, file_name, content, doc_type="bid_document", page=1, index=0, bid="Bid1", section=None):
    return Candidate(
        chunk_id=chunk_id, content=content, document_id=chunk_id, bid_id=bid, file_name=file_name,
        document_type=doc_type, page_number=page, chunk_index=index,
        metadata={"section": section} if section else {}, score=0.1,
    )  # fmt: skip


BID1 = [
    cand(1, "JA-207652 FINAL.pdf", "Request For Proposal 168884 JA-207652 Student and Staff Computing Devices. "
         "Solicitation Due 27-JUN-2024 14:00:00. Pre-Proposal Meeting 10-JUN-2024 14:00:00.", page=2, index=1),
    cand(2, "Addendum 2 RFP JA-207652.pdf", "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.",
         doc_type="addendum", page=1, index=0),
    cand(3, "Addendum 1 RFP JA-207652.pdf", "USB 3.1 is a minimum requirement for the display monitor.",
         doc_type="addendum", page=1, index=0),
]
BID2 = [
    cand(11, "PORFP_Dell_Laptop.pdf", "PORFP Number: #E20P4600040. Manufacturer Name Dell. Delivery within 45 days of Award.",
         page=1, index=0, bid="Bid2"),
    cand(12, "Dell_Laptop_Specs.pdf", "Dell Latitude 5550 Intel Core Ultra 5 125U processor 16 GB DDR5 256 GB SSD",
         doc_type="specification", page=1, index=0, bid="Bid2"),
]


class StubRetrieval:
    """Stands in for RetrievalService: returns the bid's corpus, honouring bid/type filters."""

    def __init__(self, corpora=None, error=None):
        self.corpora = corpora or {"Bid1": BID1, "Bid2": BID2}
        self.error = error
        self.calls: list[dict] = []

    async def available_filters(self):
        if self.error:
            raise self.error
        return {"bids": sorted(self.corpora), "document_types": [], "file_names": []}

    async def retrieve(self, query, bid_id=None, document_type=None, top_k=5, **_):
        if self.error:
            raise self.error
        self.calls.append({"query": query, "bid_id": bid_id, "document_type": document_type, "top_k": top_k})
        chunks = [c for c in self.corpora.get(bid_id, []) if document_type in (None, c.document_type)]
        results = [FusedResult(candidate=c, rrf_score=1 / (60 + i)) for i, c in enumerate(chunks[:top_k], 1)]
        return Found(results)


@dataclass
class Found:
    results: list[FusedResult]


def llm_json(prompt: str, answers: dict | None = None) -> str:
    """A well-formed reply for every field key in `prompt`; `answers` maps key -> (candidates, confidence)."""
    answers = answers or {}
    fields = []
    for key in re.findall(r'key="(\w+)"', prompt):
        candidates, confidence = answers.get(key, ([], 0.0))
        fields.append({"key": key, "candidates": candidates, "confidence": confidence, "notes": None})
    return json.dumps({"fields": fields})


class FakeLLM:
    """Scripted Groq client. `script` items are replies (str), exceptions to raise, or callables."""

    model = "test-model"

    def __init__(self, script=None, answers=None):
        self.script = list(script or [])
        self.answers = answers or {}
        self.calls: list[list[dict]] = []

    async def generate_json(self, messages, **params):
        self.calls.append([dict(m) for m in messages])
        prompt = messages[1]["content"]
        step = self.script.pop(0) if self.script else (lambda p: llm_json(p, self.answers))
        if isinstance(step, Exception):
            raise step
        return step(prompt) if callable(step) else step
