"""Structured bid extraction: field-oriented retrieval -> Groq -> validation."""

import asyncio
import json
import logging
from collections.abc import Callable, Awaitable, Sequence
from typing import Any, Protocol

import groq
from pydantic import ValidationError

from app.core.config import Settings
from app.extraction.evidence import gather_evidence, select_group_evidence
from app.extraction.fields import FIELDS, FIELDS_BY_KEY, GROUPS, FieldSpec
from app.extraction.prompts import SYSTEM_PROMPT, build_user_prompt, citation_retry_prompt, retry_prompt
from app.extraction.schema import BidExtraction, ExtractedField, LLMField, LLMGroupOutput
from app.extraction.validation import resolve_field
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # initial request + one retry after malformed / invalid output
MAX_RATE_LIMIT_WAIT = 20.0
DEFAULT_MAX_CHUNKS_PER_CALL = 14  # EXTRACTION_MAX_CHUNKS_PER_CALL default; deeper-evidence fields widen only at or above it


class ExtractionError(Exception):
    """Base class for extraction failures."""


class ProviderJSONError(ValueError):
    """Groq's JSON mode could not produce valid JSON (HTTP 400 `json_validate_failed`).

    Observed with real models; it is a malformed-output problem, so callers retry it once like any
    other invalid reply instead of treating it as a provider outage.
    """


def _is_json_failure(exc: groq.BadRequestError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    code = (body.get("error") or {}).get("code") if isinstance(body.get("error"), dict) else None
    return code == "json_validate_failed" or "json_validate_failed" in str(exc)


class UnknownBidError(ExtractionError):
    pass


class LLMProviderError(ExtractionError):
    """Groq could not be reached or refused the request (not a malformed answer)."""


class LLMClient(Protocol):
    model: str

    async def generate_json(self, messages: Sequence[dict[str, str]], **params: Any) -> str: ...


def parse_group_output(raw: str) -> LLMGroupOutput:
    """JSON text -> validated model. Raises ValueError (incl. pydantic ValidationError)."""
    return LLMGroupOutput.model_validate(json.loads(raw))


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"schema error at {'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return f"{type(exc).__name__}: {str(exc)[:120]}"


class ExtractionService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        settings: Settings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.settings = settings
        self._sleep = sleep
        self._schema_retries = 0
        self._citation_retries = 0

    async def extract_bid(self, bid_id: str, evidence: dict[str, list] | None = None) -> BidExtraction:
        """Extract all fields. `evidence` (field key -> chunks) lets a caller that already ran
        retrieval, such as the pipeline's retrieval stage, avoid retrieving twice."""
        bids = (await self.retrieval.available_filters())["bids"]
        if bid_id not in bids:
            raise UnknownBidError(f"Unknown bid_id {bid_id!r}. Available: {', '.join(bids) or 'none'}")

        if evidence is None:
            evidence = await gather_evidence(self.retrieval, bid_id, self.settings)
        results: dict[str, ExtractedField] = {}
        errors: list[str] = []
        self._schema_retries = self._citation_retries = 0

        for group in GROUPS:
            specs = [f for f in FIELDS if f.group == group]
            group_evidence = {s.key: evidence.get(s.key, []) for s in specs}
            # A field that asks for deeper evidence (several product lines) gets that many extra chunks of its own.
            # Not applied when the operator lowered the per-call cap below its default: a small cap stays a hard limit.
            extra = {s.key: s.evidence_limit - self.settings.extraction_evidence_per_field for s in specs
                     if s.evidence_limit and s.evidence_limit > self.settings.extraction_evidence_per_field
                     and self.settings.extraction_max_chunks_per_call >= DEFAULT_MAX_CHUNKS_PER_CALL}  # fmt: skip
            chunks, allowed = select_group_evidence(group_evidence, self.settings.extraction_max_chunks_per_call, extra)
            if not chunks:
                for s in specs:
                    results[s.key] = ExtractedField.missing()
                continue

            llm_fields, error, transcript = await self._run_group(bid_id, specs, chunks, allowed)
            # A citation is valid if it names any chunk that was sent in this call. The value must
            # still be grounded in what it cites, so widening from the field's own list loses no safety.
            sent = {c.chunk_id: c for c in chunks}
            allowed = dict.fromkeys(allowed, sent)
            if error:
                errors.append(f"group '{group}': {error}")
                for s in specs:
                    results[s.key] = ExtractedField.missing("error", f"Extraction failed: {error}")
                continue
            for s in specs:
                results[s.key] = resolve_field(s, llm_fields.get(s.key), allowed[s.key])
            await self._retry_unsupported(bid_id, specs, results, allowed, transcript)
            for s in specs:
                logger.info(
                    "Extracted bid=%s field=%s status=%s evidence=%d",
                    bid_id, s.key, results[s.key].status, len(allowed[s.key]),
                )  # fmt: skip

        by_name = {FIELDS_BY_KEY[k].name: v for k, v in results.items()}
        ordered = {f.name: by_name[f.name] for f in FIELDS}
        return BidExtraction(
            bid_id=bid_id,
            fields=ordered,
            reconciliation_required=[n for n, f in ordered.items() if f.status == "conflict"],
            errors=errors,
            complete=not errors,
            model=self.llm.model,
            schema_retries=self._schema_retries,
            citation_retries=self._citation_retries,
        )

    async def _run_group(
        self, bid_id: str, specs: list[FieldSpec], chunks, allowed
    ) -> tuple[dict[str, LLMField], str | None, list[dict[str, str]]]:
        """One LLM call (plus at most one retry). Returns (fields by key, error message, transcript)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(specs, chunks, allowed, self.settings.extraction_chunk_char_limit)},
        ]
        error = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = await self._complete(messages)
            except ProviderJSONError as exc:
                error = _short_error(exc)
                self._schema_retries += 1
                logger.warning("Provider rejected the model's JSON bid=%s fields=%d attempt=%d", bid_id, len(specs), attempt)
                messages = [*messages, {"role": "user", "content": retry_prompt(error)}]
                continue
            try:
                output = parse_group_output(raw)
            except ValueError as exc:  # JSONDecodeError and pydantic ValidationError
                error = _short_error(exc)
                self._schema_retries += 1
                logger.warning("Invalid LLM output bid=%s fields=%d attempt=%d: %s", bid_id, len(specs), attempt, error)
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": retry_prompt(error)},
                ]
                continue
            return {f.key: f for f in output.fields}, None, [*messages, {"role": "assistant", "content": raw[:6000]}]
        return {}, f"model output invalid after {MAX_ATTEMPTS} attempts ({error})", messages

    async def _retry_unsupported(
        self, bid_id: str, specs: list[FieldSpec], results: dict[str, ExtractedField], allowed, transcript
    ) -> None:
        """One bounded follow-up per group for fields rejected as unsupported.

        In real runs the model sometimes gives a correct value but cites a chunk that does not
        contain it. The follow-up names the problem and asks for a correct citation (or no
        answer). The reply is validated exactly like the first one; a field only improves if the
        new answer passes citation and grounding checks, and it is never repaired by the backend.
        """
        failed = [s for s in specs if results[s.key].status == "unsupported"]
        if not failed:
            return
        self._citation_retries += 1
        feedback = citation_retry_prompt([(s.key, results[s.key].notes or "unsupported") for s in failed])
        messages = [*transcript, {"role": "user", "content": feedback}]
        try:
            output = parse_group_output(await self._complete(messages))
        except ValueError as exc:
            logger.warning("Citation retry for bid=%s was not valid JSON (%s); fields stay unsupported", bid_id, _short_error(exc))
            return
        by_key = {f.key: f for f in output.fields}
        for s in failed:
            if s.key in by_key:
                retried = resolve_field(s, by_key[s.key], allowed[s.key])
                if retried.status != "unsupported":
                    results[s.key] = retried

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        return await call_llm(
            self.llm, messages, max_tokens=self.settings.extraction_max_output_tokens, sleep=self._sleep
        )


def _retry_after(exc: groq.RateLimitError) -> float:
    try:
        return min(float(exc.response.headers.get("retry-after", 5)), MAX_RATE_LIMIT_WAIT)
    except (AttributeError, TypeError, ValueError):
        return 5.0


async def call_llm(
    llm: LLMClient,
    messages: Sequence[dict[str, str]],
    *,
    max_tokens: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Call Groq in JSON mode. A rate limit is waited out once; any other provider error is fatal."""
    for attempt in (1, 2):
        try:
            return await llm.generate_json(messages, max_tokens=max_tokens)
        except groq.RateLimitError as exc:
            if attempt == 2:
                raise LLMProviderError("Groq rate limit exceeded") from exc
            wait = _retry_after(exc)
            logger.warning("Groq rate limited; waiting %.0fs", wait)
            await sleep(wait)
        except groq.BadRequestError as exc:
            if _is_json_failure(exc):
                raise ProviderJSONError("the provider could not produce valid JSON") from exc
            logger.error("Groq request failed: %s", type(exc).__name__)
            raise LLMProviderError(f"Groq request failed ({type(exc).__name__})") from exc
        except groq.APIError as exc:
            logger.error("Groq request failed: %s", type(exc).__name__)
            raise LLMProviderError(f"Groq request failed ({type(exc).__name__})") from exc
    raise LLMProviderError("Groq request failed")  # pragma: no cover


async def generate_validated(
    llm: LLMClient,
    messages: list[dict[str, str]],
    parse: Callable[[str], Any],
    *,
    max_tokens: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[Any | None, str | None]:
    """Ask for JSON, validate it with `parse`, and retry once with the error on failure.

    Returns (parsed, None) or (None, error). Provider errors raise LLMProviderError.
    """
    error = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = await call_llm(llm, messages, max_tokens=max_tokens, sleep=sleep)
        except ProviderJSONError as exc:  # no reply text to echo back; just ask again
            error = _short_error(exc)
            logger.warning("Provider rejected the model's JSON attempt=%d", attempt)
            messages = [*messages, {"role": "user", "content": retry_prompt(error)}]
            continue
        try:
            return parse(raw), None
        except ValueError as exc:  # JSONDecodeError and pydantic ValidationError
            error = _short_error(exc)
            logger.warning("Invalid LLM output attempt=%d: %s", attempt, error)
            messages = [
                *messages,
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content": retry_prompt(error)},
            ]
    return None, f"model output invalid after {MAX_ATTEMPTS} attempts ({error})"
