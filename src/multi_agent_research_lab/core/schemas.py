"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(BaseModel):
    agent: AgentName
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    title: str
    url: str | None = None
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    """Per-run benchmark metrics — overall + per-dimension breakdown."""

    run_name: str
    latency_seconds: float
    estimated_cost_usd: float | None = None

    # ── Overall (weighted average of sub-scores below) ──────────────────────
    quality_score: float | None = Field(default=None, ge=0, le=10)

    # ── Dimension scores (0–10 each) ────────────────────────────────────────
    factual_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Ratio of VERIFIED claims in critic review (0–10)",
    )
    citation_coverage: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="What fraction of sources are cited inline (0–10)",
    )
    structure_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Markdown quality: headings, bullets, conclusion, length (0–10)",
    )
    completeness_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Query keyword coverage and answer length adequacy (0–10)",
    )
    critic_quality_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Numeric score extracted directly from critic's review",
    )

    # ── Categorical / counts ─────────────────────────────────────────────────
    hallucination_risk: str | None = Field(
        default=None,
        description="LOW / MEDIUM / HIGH from critic review",
    )
    answer_word_count: int | None = None
    total_tokens: int | None = None
    iterations_used: int | None = None
    failure_count: int = 0

    notes: str = ""
