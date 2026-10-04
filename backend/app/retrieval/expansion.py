"""Deterministic query expansion for questions whose wording does not match the documents'.

Why: a generic question such as "What product is being procured?" shares no vocabulary with
the text that answers it ("student and staff computing devices, including laptops..."). In the
real Bid1 corpus the answering chunks ranked 31st and 45th by vector similarity and were not
found by keyword search at all, while compliance boilerplate about "produced equipment" ranked
first. A variant phrased in the *structure* of a solicitation ("what goods or services does this
request for proposal solicit") finds them (vector rank 2, keyword rank 4).

The expansion is generic (it depends on the question's intent, not on any particular question),
bounded (at most MAX_VARIANTS extra queries), deterministic (no LLM) and only used when the
first retrieval looks weak. Variants go through the same bid-scoped retrieval as the question.
"""

import re
from collections.abc import Sequence

MAX_VARIANTS = 2

# (intent pattern, variant phrased in solicitation vocabulary). Only intents whose variant was
# checked against real retrieval belong here.
_INTENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(product|products|item|items|goods|equipment|supplies|devices?|commodit(?:y|ies)"
            r"|(?:being|is|are|to be)\s+(?:procured|purchased|bought|acquired|requested|solicited|sourced)"
            r"|procur\w*|purchas\w*)\b",
            re.I,
        ),
        "what goods or services does this request for proposal solicit",
    ),
)


def query_variants(question: str) -> list[str]:
    """Up to MAX_VARIANTS alternative phrasings for `question` (empty when no intent matches)."""
    variants: list[str] = []
    for pattern, variant in _INTENTS:
        if pattern.search(question) and variant.lower() != question.lower() and variant not in variants:
            variants.append(variant)
    return variants[:MAX_VARIANTS]


def is_weak(methods_per_result: Sequence[Sequence[str]]) -> bool:
    """Retrieval looks weak when no top result is supported by BOTH vector and keyword search.

    When the question's words really occur in the answering text, both channels agree on it;
    when only one channel finds anything, the question and the documents are speaking
    different vocabularies.
    """
    return not any(len(set(methods)) >= 2 for methods in methods_per_result)
