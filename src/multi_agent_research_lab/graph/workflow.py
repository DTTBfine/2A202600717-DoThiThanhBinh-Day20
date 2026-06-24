"""LangGraph multi-agent workflow.

Graph topology:
  START → supervisor → (conditional edge) → researcher / analyst / writer / critic / END
  researcher → supervisor  (loop back)
  analyst    → supervisor
  writer     → supervisor
  critic     → supervisor
  supervisor → END when route == "done"
"""

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import RunTrace
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

# LangGraph state is just the serialised ResearchState dict
_State = dict[str, Any]


def _to_state(rs: ResearchState) -> _State:
    return rs.model_dump()


def _from_state(d: _State) -> ResearchState:
    return ResearchState.model_validate(d)


def _route_from_history(state: _State) -> str:
    """Read the last supervisor decision from route_history."""
    history: list[str] = state.get("route_history", [])
    if not history:
        return "supervisor"
    last = history[-1]
    return last  # "researcher" | "analyst" | "writer" | "critic" | "done"


class MultiAgentWorkflow:
    """Builds and runs the LangGraph multi-agent graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

        self._supervisor = SupervisorAgent(settings=self._settings)
        self._researcher = ResearcherAgent(llm=self._llm, search=self._search)
        self._analyst = AnalystAgent(llm=self._llm)
        self._writer = WriterAgent(llm=self._llm)
        self._critic = CriticAgent(llm=self._llm)

    # ------------------------------------------------------------------
    # Node functions (dict → dict for LangGraph)
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: _State) -> _State:
        rs = _from_state(state)
        rs = self._supervisor.run(rs)
        return _to_state(rs)

    def _researcher_node(self, state: _State) -> _State:
        rs = _from_state(state)
        rs = self._researcher.run(rs)
        return _to_state(rs)

    def _analyst_node(self, state: _State) -> _State:
        rs = _from_state(state)
        rs = self._analyst.run(rs)
        return _to_state(rs)

    def _writer_node(self, state: _State) -> _State:
        rs = _from_state(state)
        rs = self._writer.run(rs)
        return _to_state(rs)

    def _critic_node(self, state: _State) -> _State:
        rs = _from_state(state)
        rs = self._critic.run(rs)
        return _to_state(rs)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Any:
        """Create and compile a LangGraph StateGraph."""
        from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]

        builder: StateGraph = StateGraph(dict)  # type: ignore[type-arg]

        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)
        builder.add_node("critic", self._critic_node)

        # Entry point
        builder.set_entry_point("supervisor")

        # Conditional edges from supervisor
        builder.add_conditional_edges(
            "supervisor",
            _route_from_history,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )

        # All workers loop back to supervisor
        for worker in ("researcher", "analyst", "writer", "critic"):
            builder.add_edge(worker, "supervisor")

        return builder.compile()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final ResearchState."""
        trace = RunTrace(run_name="multi-agent")

        with trace.span("workflow.total", query=state.request.query):
            graph = self.build()
            initial: _State = _to_state(state)

            logger.info("Starting multi-agent workflow: %s", state.request.query)
            result: _State = graph.invoke(initial)
            logger.info("Workflow complete after %d iterations", result.get("iteration", 0))

        final = _from_state(result)
        # Append workflow-level trace summary
        final.add_trace_event(
            "workflow_summary",
            {
                "total_duration_seconds": trace.total_duration,
                "total_input_tokens": trace.total_input_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "total_cost_usd": trace.total_cost_usd,
                "iterations": final.iteration,
                "route_history": final.route_history,
            },
        )
        return final
