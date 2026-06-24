"""FastAPI REST API — wraps the CLI runners for HTTP access.

Default port: 8080 (never conflicts with common 8000/3000 ports).

Endpoints:
  GET  /health
  POST /research/baseline
  POST /research/multi-agent
  GET  /research/config
"""

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

configure_logging(get_settings().log_level)

api = FastAPI(
    title="Multi-Agent Research Lab API",
    description="REST interface for single-agent baseline and multi-agent research workflow.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances (initialised once at startup)
_llm: LLMClient | None = None
_search: SearchClient | None = None
_workflow: MultiAgentWorkflow | None = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def _get_search() -> SearchClient:
    global _search
    if _search is None:
        _search = SearchClient()
    return _search


def _get_workflow() -> MultiAgentWorkflow:
    global _workflow
    if _workflow is None:
        _workflow = MultiAgentWorkflow(llm=_get_llm(), search=_get_search())
    return _workflow


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Research question")
    max_sources: int = Field(default=5, ge=1, le=10)
    audience: str = Field(default="technical learners")


class SourceOut(BaseModel):
    title: str
    url: str | None
    snippet: str


class ResearchResponse(BaseModel):
    query: str
    run_type: str
    final_answer: str | None
    critic_notes: str | None = None
    sources: list[SourceOut]
    route_history: list[str]
    iteration: int
    errors: list[str]
    latency_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": (
            settings.anthropic_model
            if settings.llm_provider == "anthropic"
            else settings.openai_model
        ),
    }


@api.get("/research/config", tags=["meta"])
def config() -> dict[str, Any]:
    s = get_settings()
    return {
        "llm_provider": s.llm_provider,
        "max_iterations": s.max_iterations,
        "timeout_seconds": s.timeout_seconds,
        "api_port": s.api_port,
    }


@api.post("/research/baseline", response_model=ResearchResponse, tags=["research"])
def run_baseline(request: ResearchRequest) -> ResearchResponse:
    """Single-agent baseline — one LLM call covers everything."""
    t0 = time.perf_counter()
    try:
        llm = _get_llm()
        search = _get_search()
        req = ResearchQuery(
            query=request.query,
            max_sources=request.max_sources,
            audience=request.audience,
        )
        state = ResearchState(request=req)
        sources = search.search(request.query, max_results=request.max_sources)
        state.sources = sources
        context = "\n".join(
            f"[S{i+1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
        )
        system = (
            "You are a research assistant. Answer the query thoroughly using the provided sources. "
            "Include [S#] citations. Format with markdown headings. End with a Conclusion."
        )
        user = f"Query: {request.query}\n\nSources:\n{context}"
        resp = llm.complete(system, user)
        state.final_answer = resp.content
        state.research_notes = resp.content

        return ResearchResponse(
            query=request.query,
            run_type="single-agent-baseline",
            final_answer=state.final_answer,
            sources=[SourceOut(title=s.title, url=s.url, snippet=s.snippet) for s in state.sources],
            route_history=state.route_history,
            iteration=state.iteration,
            errors=state.errors,
            latency_seconds=round(time.perf_counter() - t0, 3),
            metadata={
                "model": resp.model,
                "tokens_in": resp.input_tokens,
                "tokens_out": resp.output_tokens,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/research/multi-agent", response_model=ResearchResponse, tags=["research"])
def run_multi_agent(request: ResearchRequest) -> ResearchResponse:
    """Full multi-agent workflow: supervisor → researcher → analyst → writer → critic."""
    t0 = time.perf_counter()
    try:
        wf = _get_workflow()
        req = ResearchQuery(
            query=request.query,
            max_sources=request.max_sources,
            audience=request.audience,
        )
        state = ResearchState(request=req)
        result = wf.run(state)

        total_cost = sum(
            float(r.metadata.get("cost_usd") or 0) for r in result.agent_results
        )

        return ResearchResponse(
            query=request.query,
            run_type="multi-agent",
            final_answer=result.final_answer,
            critic_notes=result.critic_notes,
            sources=[
                SourceOut(title=s.title, url=s.url, snippet=s.snippet) for s in result.sources
            ],
            route_history=result.route_history,
            iteration=result.iteration,
            errors=result.errors,
            latency_seconds=round(time.perf_counter() - t0, 3),
            metadata={
                "total_cost_usd": total_cost,
                "agents_used": [r.agent for r in result.agent_results],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
