"""Researcher agent — collects sources then synthesises research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a researcher agent. Your job is to:
1. Review the provided search results.
2. Extract key facts, numbers, and findings relevant to the query.
3. Note which claims come from which source using [S1], [S2], ... markers.
4. Keep notes concise but complete — the analyst and writer will rely on them.

Format:
## Research Notes
**Topic**: <query>
### Key Findings
- ...
### Sources
- [S1] ...
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        query = state.request.query
        max_sources = state.request.max_sources

        try:
            with trace_span("researcher.search", {"query": query}) as span:
                sources = self._search.search(query, max_results=max_sources)
                span["attributes"]["num_sources"] = len(sources)

            state.sources = sources
            logger.info("Researcher: found %d sources", len(sources))

            # Build context from sources
            context_parts = []
            for i, src in enumerate(sources, 1):
                context_parts.append(
                    f"[S{i}] {src.title}\n  URL: {src.url or 'N/A'}\n  {src.snippet}"
                )
            context = "\n\n".join(context_parts)

            user_prompt = (
                f"Query: {query}\n\n"
                f"Target audience: {state.request.audience}\n\n"
                f"Search results:\n{context}"
            )

            with trace_span("researcher.llm") as span:
                resp = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                span["attributes"].update(
                    {
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "model": resp.model,
                    }
                )

            state.research_notes = resp.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=resp.content,
                    metadata={
                        "num_sources": len(sources),
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    },
                )
            )
            state.add_trace_event("researcher_done", {"notes_length": len(resp.content)})
            logger.info("Researcher: research notes ready (%d chars)", len(resp.content))

        except Exception as exc:
            msg = f"ResearcherAgent failed: {exc}"
            logger.error(msg)
            state.errors.append(msg)
            # Provide minimal fallback so downstream agents can still run
            state.research_notes = f"[Researcher fallback] Unable to complete search: {exc}"

        return state
