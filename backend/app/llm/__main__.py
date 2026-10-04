"""CLI: `python -m app.llm --smoke-test` checks the Groq configuration with a tiny request.

No bid documents are sent. The API key is never printed.
"""

import argparse
import asyncio
import json
import sys

from groq import APIError

from app.core.config import ConfigurationError, get_settings
from app.llm.groq_client import GroqClient


async def smoke_test() -> int:
    settings = get_settings()
    try:
        client = GroqClient.from_settings(settings)
    except ConfigurationError as exc:
        print(f"Groq configuration: MISSING ({exc})")
        return 2
    print("Groq configuration: OK")
    print("Model: configured")

    messages = [
        {"role": "system", "content": "Reply with a JSON object only."},
        {"role": "user", "content": 'Return exactly {"status": "ok", "number": 7} as JSON.'},
    ]
    try:
        raw = await client.generate_json(messages, max_tokens=500)
    except APIError as exc:
        print(f"JSON generation: FAILED ({type(exc).__name__})")  # the type only: never request details
        return 1
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("JSON generation: FAILED (reply was not valid JSON)")
        return 1
    if parsed.get("status") != "ok":
        print("JSON generation: FAILED (unexpected content)")
        return 1
    call = client.calls[-1]
    print("JSON generation: OK")
    print(f"(1 request, {call.seconds:.1f}s, finish_reason={call.finish_reason}, "
          f"tokens in/out={call.prompt_tokens}/{call.completion_tokens})")  # fmt: skip
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", required=True)
    parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # document text can contain non-cp1252 characters
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(smoke_test()))
