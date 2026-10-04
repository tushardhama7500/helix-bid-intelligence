"""Per-run logging context: a correlation id and the current pipeline stage."""

from contextvars import ContextVar

run_id_var: ContextVar[str] = ContextVar("helix_run_id", default="-")
stage_var: ContextVar[str] = ContextVar("helix_stage", default="-")
