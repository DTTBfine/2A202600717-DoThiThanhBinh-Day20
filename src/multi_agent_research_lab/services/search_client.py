"""Search client abstraction for ResearcherAgent.

Provider selection: Tavily (if key set) → Mock corpus.
"""

import logging
import re
from dataclasses import dataclass, field

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock corpus — covers the benchmark queries out of the box
# ---------------------------------------------------------------------------

_MOCK_CORPUS: list[dict[str, str]] = [
    {
        "title": "GraphRAG: Unlocking LLM discovery on narrative private data (Microsoft, 2024)",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "GraphRAG uses an LLM to build a knowledge graph from source text, then performs "
            "community detection and hierarchical summarization. Achieves 25-40% improvement on "
            "multi-hop QA benchmarks vs naive RAG. Open-sourced at microsoft/graphrag."
        ),
        "tags": ["graphrag", "rag", "graph", "knowledge graph", "retrieval"],
    },
    {
        "title": "Building Effective Agents — Anthropic Engineering, 2024",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Key principles: start simple, compose workflows not monoliths, invest in tool "
            "quality, "
            "add guardrails (max iterations, timeout, human escalation). Multi-agent only when "
            "task complexity justifies orchestration overhead."
        ),
        "tags": ["agents", "multi-agent", "guardrails", "anthropic", "best practices"],
    },
    {
        "title": "LangGraph: Stateful multi-actor applications with LLMs (LangChain, 2024)",
        "url": "https://langchain-ai.github.io/langgraph/",
        "snippet": (
            "LangGraph models agent workflows as directed graphs. StateGraph manages shared state "
            "across nodes. Supports cycles, conditional edges, and human-in-the-loop. "
            "Production deployments at scale use checkpointing and streaming."
        ),
        "tags": ["langgraph", "graph", "workflow", "state", "multi-agent"],
    },
    {
        "title": "Survey: Multi-Agent LLM Architectures for Complex Tasks (arXiv 2024)",
        "url": "https://arxiv.org/abs/2402.01680",
        "snippet": (
            "Comprehensive survey of 50+ multi-agent systems. Supervisor-worker pattern dominates "
            "production. Shared memory and structured handoff are critical for quality. "
            "Benchmark: multi-agent achieves 8.1/10 vs single-agent 6.2/10 on complex queries."
        ),
        "tags": ["survey", "multi-agent", "benchmark", "supervisor", "architecture"],
    },
    {
        "title": "Production Guardrails for LLM Agents (ML Engineering Blog, 2025)",
        "url": "https://mlops.community/production-guardrails-llm-agents/",
        "snippet": (
            "Essential guardrails: (1) max_iterations to prevent infinite loops, (2) timeout per "
            "step, (3) retry with exponential backoff, (4) output schema validation with Pydantic, "
            "(5) cost budget limits. Reduced failure rate from 18% to 3% in production."
        ),
        "tags": ["guardrails", "production", "safety", "retry", "validation", "agents"],
    },
    {
        "title": "Single-Agent vs Multi-Agent: When to use which? (2025)",
        "url": "https://blog.langchain.dev/single-vs-multi-agent/",
        "snippet": (
            "Single-agent: best for well-scoped, latency-sensitive tasks (p95 2s vs 8s). "
            "Multi-agent: best for complex reasoning, multiple knowledge domains, parallel work. "
            "Cost trade-off: multi-agent is 3x more expensive but 30% better quality."
        ),
        "tags": ["comparison", "single-agent", "multi-agent", "latency", "cost", "quality"],
    },
    {
        "title": "OpenAI Agents SDK: Orchestration and Handoffs (2025)",
        "url": "https://developers.openai.com/api/docs/guides/agents/orchestration",
        "snippet": (
            "Agents SDK supports handoffs between specialized agents, built-in tool use, "
            "and structured outputs. Supervisor pattern: orchestrator decides which agent runs "
            "next "
            "based on conversation context."
        ),
        "tags": ["openai", "agents", "handoff", "orchestration", "sdk"],
    },
    {
        "title": "LangSmith Tracing for Multi-Agent Systems (2024)",
        "url": "https://docs.smith.langchain.com/",
        "snippet": (
            "LangSmith provides per-step latency, token usage, and cost tracing for "
            "LangChain/LangGraph "
            "pipelines. Critical for debugging agent failures and optimizing routing decisions."
        ),
        "tags": ["tracing", "langsmith", "observability", "debugging", "cost"],
    },
]


class _MockSearchClient:
    """Keyword-based mock search over a curated corpus."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        q_words = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[int, dict[str, str]]] = []
        for doc in _MOCK_CORPUS:
            tags = set(doc["tags"])
            title_words = set(re.findall(r"\w+", doc["title"].lower()))
            snippet_words = set(re.findall(r"\w+", doc["snippet"].lower()))
            score = (
                len(q_words & tags) * 3
                + len(q_words & title_words) * 2
                + len(q_words & snippet_words)
            )
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]
        if not top:
            top = [(0, d) for d in _MOCK_CORPUS[:max_results]]

        return [
            SourceDocument(
                title=d["title"],
                url=d["url"],
                snippet=d["snippet"],
                metadata={"score": s},
            )
            for s, d in top
        ]


class _TavilySearchClient:
    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        resp = self._client.search(query=query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", ""),
                url=r.get("url"),
                snippet=r.get("content", ""),
                metadata={"score": r.get("score", 0.0)},
            )
            for r in resp.get("results", [])
        ]


@dataclass
class SearchClient:
    """Provider-agnostic search client. Auto-selects backend from Settings."""

    _settings: Settings = field(default_factory=get_settings)
    _backend: object = field(init=False)

    def __post_init__(self) -> None:
        if self._settings.tavily_api_key:
            self._backend = _TavilySearchClient(self._settings.tavily_api_key)
            logger.info("Search backend: Tavily")
        else:
            self._backend = _MockSearchClient()
            logger.info("Search backend: Mock corpus (no TAVILY_API_KEY)")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return relevant SourceDocuments for the query."""
        return self._backend.search(query=query, max_results=max_results)  # type: ignore[union-attr]
