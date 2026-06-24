"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an analyst agent. Your job is to:
1. Read the research notes provided.
2. Extract and rank key claims by evidence strength (Strong / Moderate / Weak).
3. Compare viewpoints and identify conflicts or gaps.
4. If a comparison (e.g., single-agent vs multi-agent) is relevant, add a table.
5. Flag any hallucination risks or unsupported assumptions.
6. Conclude with a recommendation for the writer.

Format:
## Analysis Notes
### Key Claims
- [STRONG/MODERATE/WEAK] <claim> — source hint
### Comparison (if applicable)
| Dimension | Option A | Option B |
### Risks / Gaps
- ...
### Recommendation for Writer
...
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.errors.append("AnalystAgent: no research_notes to analyse")
            state.analysis_notes = "[Analyst fallback] No research notes available."
            return state

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}"
        )

        try:
            with trace_span("analyst.llm") as span:
                resp = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                span["attributes"].update(
                    {
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "model": resp.model,
                    }
                )

            state.analysis_notes = resp.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst_done", {"analysis_length": len(resp.content)})
            logger.info("Analyst: analysis notes ready (%d chars)", len(resp.content))

        except Exception as exc:
            msg = f"AnalystAgent failed: {exc}"
            logger.error(msg)
            state.errors.append(msg)
            state.analysis_notes = f"[Analyst fallback] Analysis unavailable: {exc}"

        return state
