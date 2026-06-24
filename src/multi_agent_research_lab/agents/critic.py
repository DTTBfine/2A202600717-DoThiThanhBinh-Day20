"""Critic agent — fact-checks final answer and measures citation coverage."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critic agent responsible for quality control. Your job is to:
1. Fact-check the final answer against the research and analysis notes.
2. Mark each major claim as [VERIFIED], [UNCERTAIN], or [UNSUPPORTED].
3. Count citation coverage: how many factual claims have a [S#] reference?
4. Flag any hallucinations or inconsistencies.
5. Assign an overall quality score 0-10.
6. List any recommended edits.

Format:
## Critic Review
### Fact-Check Results
- [VERIFIED/UNCERTAIN/UNSUPPORTED] <claim>
### Citation Coverage: X/Y claims cited (Z%)
### Hallucination Risk: LOW / MEDIUM / HIGH
### Quality Score: X/10
### Recommended Edits
- ...
"""


def _count_citations(text: str) -> int:
    """Count how many [S#] citation markers appear in the text."""
    return len(re.findall(r"\[S\d+\]", text))


class CriticAgent(BaseAgent):
    """Fact-checking and quality-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            state.errors.append("CriticAgent: no final_answer to review")
            state.critic_notes = "[Critic skipped] No answer to review."
            return state

        citation_count = _count_citations(state.final_answer)

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Final answer to review:\n{state.final_answer}\n\n"
            f"Citations found in answer: {citation_count} [S#] markers."
        )

        try:
            with trace_span("critic.llm") as span:
                resp = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                span["attributes"].update(
                    {
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "citation_count": citation_count,
                    }
                )

            state.critic_notes = resp.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "citation_count": citation_count,
                    },
                )
            )
            state.add_trace_event(
                "critic_done",
                {"citation_count": citation_count, "review_length": len(resp.content)},
            )
            logger.info("Critic: review done (citations=%d)", citation_count)

        except Exception as exc:
            msg = f"CriticAgent failed: {exc}"
            logger.error(msg)
            state.errors.append(msg)
            state.critic_notes = f"[Critic fallback] Review unavailable: {exc}"

        return state
