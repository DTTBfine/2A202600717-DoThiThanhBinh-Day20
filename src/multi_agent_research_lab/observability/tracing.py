"""Tracing hooks — structured JSON spans + optional LangSmith integration."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight in-process span — always available
# ---------------------------------------------------------------------------


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that measures wall-clock duration and logs the span."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug("SPAN %s", json.dumps(span, default=str))


# ---------------------------------------------------------------------------
# Run-level trace accumulator
# ---------------------------------------------------------------------------


class RunTrace:
    """Accumulates all spans for one workflow run."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.spans: list[dict[str, Any]] = []
        self._start = perf_counter()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
        with trace_span(name, dict(attributes)) as s:
            yield s
        self.spans.append(s)

    @property
    def total_duration(self) -> float:
        return round(perf_counter() - self._start, 4)

    @property
    def total_input_tokens(self) -> int:
        return sum(int(s["attributes"].get("input_tokens", 0)) for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(int(s["attributes"].get("output_tokens", 0)) for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(float(s["attributes"].get("cost_usd", 0.0)) for s in self.spans), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "total_duration_seconds": self.total_duration,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "spans": self.spans,
        }

    def export_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
