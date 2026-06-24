"""Writer agent — produces final answer from research + analysis."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a writer agent. Your job is to:
1. Synthesise the research notes and analysis into a clear, well-structured answer.
2. Target the specified audience — calibrate vocabulary and depth accordingly.
3. Include in-text citations ([S1], [S2], ...) wherever you make factual claims.
4. Use markdown headings and bullet lists for readability.
5. Add a "## Conclusion" section at the end.
6. Keep the answer between 400-600 words unless the query requires more.

Do NOT invent facts beyond what the notes contain.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        research = state.research_notes or "[no research notes]"
        analysis = state.analysis_notes or "[no analysis notes]"

        # Build source list for context
        source_list = "\n".join(
            f"[S{i}] {s.title} — {s.url or 'N/A'}"
            for i, s in enumerate(state.sources, 1)
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{research}\n\n"
            f"Analysis notes:\n{analysis}\n\n"
            f"Source index:\n{source_list or 'N/A'}"
        )

        try:
            with trace_span("writer.llm") as span:
                resp = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                span["attributes"].update(
                    {
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "model": resp.model,
                    }
                )

            state.final_answer = resp.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "word_count": len(resp.content.split()),
                    },
                )
            )
            state.add_trace_event("writer_done", {"answer_length": len(resp.content)})
            logger.info("Writer: final answer ready (%d chars)", len(resp.content))

        except Exception as exc:
            msg = f"WriterAgent failed: {exc}"
            logger.error(msg)
            state.errors.append(msg)
            state.final_answer = (
                f"[Writer fallback] Could not generate answer: {exc}\n\n"
                f"Research summary:\n{research[:500]}"
            )

        return state
