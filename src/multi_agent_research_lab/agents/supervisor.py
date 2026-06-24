"""Supervisor / router agent.

Uses a deterministic state machine for routing correctness.
Max-iterations and error fallback are enforced here — workers never need to handle them.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Deterministic routing policy with max-iterations guard.

        Routing order:
          researcher → analyst → writer → critic → done
        Any unrecoverable errors skip straight to done.
        """
        # Safety: stop immediately if too many errors
        if len(state.errors) >= 3:
            next_route = "done"
            logger.warning("Supervisor: stopping due to %d errors", len(state.errors))

        # Max-iterations guard (check BEFORE incrementing so the final step still runs)
        elif state.iteration >= self._settings.max_iterations:
            next_route = "done"
            logger.warning("Supervisor: max_iterations=%d reached", self._settings.max_iterations)

        elif state.research_notes is None:
            next_route = "researcher"

        elif state.analysis_notes is None:
            next_route = "analyst"

        elif state.final_answer is None:
            next_route = "writer"

        elif state.critic_notes is None:
            next_route = "critic"

        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {
                "next": next_route,
                "iteration": state.iteration,
                "has_research": state.research_notes is not None,
                "has_analysis": state.analysis_notes is not None,
                "has_answer": state.final_answer is not None,
                "has_critic": state.critic_notes is not None,
                "error_count": len(state.errors),
            },
        )
        logger.info("Supervisor → %s (iter=%d)", next_route, state.iteration)
        return state
