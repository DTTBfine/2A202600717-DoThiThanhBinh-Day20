"""Tests for the implemented agent pipeline."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

def test_supervisor_routes_researcher_first() -> None:
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_analyst_after_research() -> None:
    state = _make_state()
    state.research_notes = "some notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_writer_after_analysis() -> None:
    state = _make_state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_critic_after_answer() -> None:
    state = _make_state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "critic"


def test_supervisor_routes_done_when_complete() -> None:
    state = _make_state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.critic_notes = "critic"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_max_iterations_guard() -> None:
    from multi_agent_research_lab.core.config import Settings
    settings = Settings(MAX_ITERATIONS=2)  # type: ignore[call-arg]
    state = _make_state()
    state.iteration = 2  # already at limit
    result = SupervisorAgent(settings=settings).run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_stops_on_too_many_errors() -> None:
    state = _make_state()
    state.errors = ["e1", "e2", "e3"]
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

def test_researcher_populates_sources_and_notes() -> None:
    state = _make_state("GraphRAG state-of-the-art")
    result = ResearcherAgent().run(state)
    assert len(result.sources) > 0
    assert result.research_notes is not None
    assert len(result.research_notes) > 50


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------

def test_analyst_populates_analysis() -> None:
    state = _make_state("Compare single vs multi-agent")
    state.research_notes = (
        "## Research Notes\nKey finding: multi-agent is better for complex tasks."
    )
    result = AnalystAgent().run(state)
    assert result.analysis_notes is not None
    assert len(result.analysis_notes) > 30


def test_analyst_handles_missing_research() -> None:
    state = _make_state()
    result = AnalystAgent().run(state)
    # Should not raise; should set fallback
    assert result.analysis_notes is not None
    assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_populates_final_answer() -> None:
    state = _make_state("GraphRAG overview")
    state.research_notes = "GraphRAG uses knowledge graphs."
    state.analysis_notes = "Key claim: improved multi-hop reasoning."
    result = WriterAgent().run(state)
    assert result.final_answer is not None
    assert len(result.final_answer) > 100


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_critic_reviews_answer() -> None:
    state = _make_state("Production guardrails")
    state.research_notes = "Guardrails: max_iterations, timeout."
    state.analysis_notes = "Evidence: strong from [S2]."
    state.final_answer = "Use max_iterations [S2] and timeout [S1] for production safety."
    result = CriticAgent().run(state)
    assert result.critic_notes is not None


def test_critic_skips_when_no_answer() -> None:
    state = _make_state()
    result = CriticAgent().run(state)
    assert result.critic_notes is not None  # fallback set
    assert len(result.errors) > 0
