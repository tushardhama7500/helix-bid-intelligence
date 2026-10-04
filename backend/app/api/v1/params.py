"""Shared, strictly validated request parameters."""

from typing import Annotated

from fastapi import Path

# Bid ids are folder names under DATA_DIR: letters, digits, '_' and '-' only (no dots or slashes),
# so a bid id can never be used to build a path outside the data directory.
BID_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$"

BidId = Annotated[str, Path(pattern=BID_ID_PATTERN, max_length=64, description="Bid folder name, e.g. Bid1")]
