"""Deterministic detection of explicit change language in addendum text.

A sentence only counts as a change when it names the field *and* uses explicit
change wording ("the new due date is X", "changed from X to Y", "replace X with Y").
Merely mentioning a field, or "please refer to the original due date", is not a change.
Anything that sounds like a change but gives no usable new value is reported as
`unknown` so the caller can hand it to Groq instead of guessing.
"""

import re
from dataclasses import dataclass, replace
from typing import Literal

ChangeType = Literal["replacement", "clarification", "addition", "removal", "unknown"]

MAX_VALUE_WORDS = 25  # a "value" longer than this is prose, not a value

_CHANGE_VERBS = r"(?:chang(?:e|es|ed)|revis(?:e|es|ed)|amend(?:s|ed)?|updat(?:e|es|ed)|modif(?:y|ies|ied)|extend(?:s|ed)?|mov(?:e|es|ed)|increas(?:e|es|ed)|decreas(?:e|es|ed)|reduc(?:e|es|ed))"
_CLARIFY = re.compile(r"\bclarif(?:y|ies|ied|ication|ications)\b|\bto clarify\b|\bfor clarity\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(])|\n+")
_GENERIC_CHANGE = re.compile(
    rf"\bnew\s+[\w\- ]{{0,25}}(?:date|deadline|requirement|period|amount|term)\b|\b{_CHANGE_VERBS}\s+(?:to|from)\b|"
    r"\breplace[sd]?\b|\bsupersede[sd]?\b|\bin lieu of\b|\binstead of\b|\bis\s+extended\b|\bextend the\b",
    re.I,
)
_GENERIC_CLARIFY = _CLARIFY.pattern + r"|\bAnswer\s*:|\bQuestion\s*:"


@dataclass(frozen=True)
class DetectedChange:
    change_type: ChangeType
    previous_value: str | None
    new_value: str | None
    sentence: str
    rule: str


def has_change_language(text: str) -> bool:
    return bool(_GENERIC_CHANGE.search(text))


def has_clarification_language(text: str) -> bool:
    return bool(re.search(_GENERIC_CLARIFY, text, re.I))


def term_regex(terms: tuple[str, ...] | list[str]) -> str:
    """Alternation for field names; spaces/hyphens are interchangeable ('pre-bid' == 'pre bid')."""
    parts = []
    for term in sorted(terms, key=len, reverse=True):
        words = re.split(r"[\s\-]+", term.strip())
        parts.append(r"[\s\-]*".join(re.escape(w) for w in words))
    return "(?:" + "|".join(parts) + ")"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("\"“”'").strip()
    value = re.sub(r"[\s.;,:]+$", "", value) if not re.search(r"\b[A-Za-z]\.[A-Za-z]\.$", value) else value
    value = value.strip()
    return value or None


_PROSE_START = frozenset("to allow be include provide ensure permit make enable give all any each some this that those these".split())


def _usable(value: str | None) -> bool:
    """A usable value is short, not a pointer ('the following'), and not a clause of prose."""
    if not value:
        return False
    words = value.split()
    if len(words) > MAX_VALUE_WORDS:
        return False
    if re.match(r"(?:the\s+)?following\b|as\s+(?:follows|shown|described|stated)\b|attached\b", value, re.I):
        return False
    if any(ch.isdigit() for ch in value):
        return True
    return len(words) <= 6 and words[0].lower() not in _PROSE_START


def detect_changes(text: str, terms: tuple[str, ...] | list[str]) -> list[DetectedChange]:
    """Find explicit changes to the field named by `terms` in `text`, in document order."""
    label = term_regex(terms)
    found: list[DetectedChange] = []
    for sentence in split_sentences(text):
        if not re.search(label, sentence, re.I):
            continue
        change = _classify_sentence(sentence, label)
        if change is not None:
            prev = change.previous_value
            if prev and re.fullmatch(rf"(?:the\s+)?{label}(?:\s+\w+){{0,2}}", prev, re.I):
                change = replace(change, previous_value=None)  # "the due date" names the field, it is not a value
            found.append(change)
    return found


def _classify_sentence(s: str, label: str) -> DetectedChange | None:
    flags = re.I

    # Removal: "the bid bond requirement is deleted / waived / no longer required"
    if re.search(rf"{label}[^.;]{{0,40}}\b(?:is|are|has been|have been)?\s*(?:deleted|removed|waived|eliminated|no longer required)\b", s, flags) or \
            re.search(rf"\b(?:delete|remove|waive)\s+(?:the\s+)?(?:requirement\s+(?:for|of)\s+)?(?:{label})", s, flags):
        return DetectedChange("removal", None, None, s, "removal")

    # "<verb> ... from OLD to NEW"  (changed / extended / revised ... from X to Y)
    m = re.search(rf"\b{_CHANGE_VERBS}\b[^.;]*?\bfrom\s+(?P<old>.+?)\s+to\s+(?P<new>.+)$", s, flags)
    if m:
        return _replacement(s, m.group("old"), m.group("new"), "from_to")

    # "<field> is changed / revised / extended ... to NEW"
    m = re.search(rf"\b{_CHANGE_VERBS}\b(?:\s+\w+){{0,3}}?\s+to\s+(?P<new>.+)$", s, flags)
    if m and re.search(label, s[: m.start()] + s[m.start():m.end()], flags):
        return _replacement(s, None, m.group("new"), "changed_to")

    # "The new due date (for this RFP) will be / is NEW"
    m = re.search(rf"\bnew\s+{label}(?:\s+for\s+(?:this|the)\s+[\w\- ]{{1,30}}?)?\s+(?:will\s+be|shall\s+be|is|are)\s+(?:now\s+)?(?P<new>.+)$", s, flags)
    if m:
        return _replacement(s, None, m.group("new"), "new_field_is")

    # "Replace OLD with NEW" / "OLD is replaced by NEW" / "delete OLD and replace with NEW"
    for pattern, rule in (
        (r"\bdelete\s+(?P<old>.+?)\s+and\s+replace(?:\s+it)?\s+(?:with|by)\s+(?P<new>.+)$", "delete_replace"),
        (r"(?P<old>.+?)\s+(?:is|are)\s+(?:hereby\s+)?(?:replaced|superseded)\s+(?:with|by)\s+(?P<new>.+)$", "replaced_by"),
        (r"\breplace(?:d)?\s+(?P<old>.+?)\s+(?:with|by)\s+(?P<new>.+)$", "replace_with"),
        (r"(?P<new>.+?)\s+(?:supersedes|supersede)\s+(?P<old>.+)$", "supersedes"),
        (r"(?P<new>.+?)\s+(?:instead of|in lieu of)\s+(?P<old>.+)$", "instead_of"),
    ):
        m = re.search(pattern, s, flags)
        if m:
            return _replacement(s, m.group("old"), m.group("new"), rule)

    # Addition: "vendors must also submit Form X", "in addition, ..."
    m = re.search(r"\b(?:must|shall|will|are required to)\s+also\s+(?:submit|provide|include|furnish)\s+(?P<new>.+)$", s, flags)
    if m:
        return DetectedChange("addition", None, _clean(m.group("new")), s, "also_submit")
    m = re.search(r"\b(?:add|added)\s+(?:the\s+following\s+)?(?:requirement|document|form)s?\s*[:\-]?\s*(?P<new>.+)$", s, flags)
    if m:
        return DetectedChange("addition", None, _clean(m.group("new")), s, "added_requirement")
    if re.search(r"\bin addition\b|\badditionally\b|\bis hereby added\b|\bnew requirement\b", s, flags):
        return DetectedChange("addition", None, None, s, "addition_wording")

    if _CLARIFY.search(s):
        return DetectedChange("clarification", None, None, s, "clarification")

    # Sounds like a change but no value we can read: let the caller decide (Groq).
    if re.search(r"\breplace[sd]?\b|\bsupersede[sd]?\b|\bshall be changed\b|\bthe following (?:is|are) changed\b|\bdelete and replace\b|\b(?:is|are) (?:hereby )?(?:amended|revised|modified)\b|\bextended?\b", s, flags):
        return DetectedChange("unknown", None, None, s, "unparsed_change")
    return None


def _replacement(sentence: str, old: str | None, new: str | None, rule: str) -> DetectedChange | None:
    new_value, old_value = _clean(new), _clean(old)
    if not _usable(new_value):
        return DetectedChange("unknown", None, None, sentence, f"{rule}:no_usable_value")
    if old_value is not None and not _usable(old_value):
        old_value = None
    return DetectedChange("replacement", old_value, new_value, sentence, rule)
