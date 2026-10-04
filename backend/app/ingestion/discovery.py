"""Find supported bid documents under the data directory."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

FILE_TYPES: dict[str, str] = {".pdf": "pdf", ".html": "html", ".htm": "html"}


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    rel_path: str  # relative to the data dir, POSIX separators
    bid_id: str  # first folder under the data dir, e.g. "Bid1"
    file_type: str  # "pdf" | "html"

    @property
    def file_name(self) -> str:
        return self.path.name


@dataclass
class DiscoveryResult:
    files: list[DiscoveredFile] = field(default_factory=list)
    ignored: int = 0
    # Supported-looking files that were refused, with the reason (never silently dropped).
    refused: list[tuple[str, str]] = field(default_factory=list)


def is_within(root: Path, path: Path) -> bool:
    """True if `path` really lives under `root` after resolving symlinks and `..`."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def discover_files(data_dir: Path, max_bytes: int | None = None) -> DiscoveryResult:
    """Recursively list supported files; each top-level folder is a bid.

    Refuses (and reports) files that resolve outside `data_dir` (symlinks) and files larger than
    `max_bytes`. Raises FileNotFoundError if `data_dir` does not exist.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    result = DiscoveryResult()
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(data_dir)
        file_type = FILE_TYPES.get(path.suffix.lower())
        if file_type is None or len(rel.parts) < 2:
            if file_type is not None:
                logger.warning("Ignoring %s: not inside a bid folder", rel.as_posix())
            result.ignored += 1
            continue
        if not is_within(data_dir, path):
            logger.warning("Refusing %s: resolves outside the data directory", rel.as_posix())
            result.refused.append((rel.as_posix(), "outside data directory"))
            result.ignored += 1
            continue
        if max_bytes is not None and path.stat().st_size > max_bytes:
            logger.warning("Refusing %s: larger than %d bytes", rel.as_posix(), max_bytes)
            result.refused.append((rel.as_posix(), "file too large"))
            result.ignored += 1
            continue
        logger.info("Discovered document %s", rel.as_posix())
        result.files.append(
            DiscoveredFile(path, rel.as_posix(), rel.parts[0], file_type)
        )
    return result
