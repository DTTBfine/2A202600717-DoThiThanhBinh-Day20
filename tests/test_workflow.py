"""Integration tests for MultiAgentWorkflow and API server."""

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _simple_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


class TestWorkflow:
    def test_full_pipeline_runs(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("GraphRAG state-of-the-art")
        result = wf.run(state)

        assert result.research_notes is not None
        assert result.analysis_notes is not None
        assert result.final_answer is not None
        assert result.critic_notes is not None

    def test_route_history_correct_order(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("Compare single vs multi-agent")
        result = wf.run(state)

        history = result.route_history
        # Must contain the expected routes in order
        assert "researcher" in history
        assert "analyst" in history
        assert "writer" in history
        assert "critic" in history
        assert history[-1] == "done"

    def test_no_infinite_loop(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("Test max iterations guard")
        result = wf.run(state)
        # Should always terminate
        assert result.iteration <= 15

    def test_sources_populated(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("Production guardrails for LLM agents")
        result = wf.run(state)
        assert len(result.sources) > 0

    def test_agent_results_recorded(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("LangGraph overview")
        result = wf.run(state)
        agent_names = [r.agent for r in result.agent_results]
        assert "researcher" in agent_names
        assert "analyst" in agent_names
        assert "writer" in agent_names

    def test_trace_events_recorded(self) -> None:
        wf = MultiAgentWorkflow()
        state = _simple_state("Multi-agent survey")
        result = wf.run(state)
        event_names = [e["name"] for e in result.trace]
        assert "researcher_done" in event_names
        assert "analyst_done" in event_names
        assert "writer_done" in event_names
        assert "workflow_summary" in event_names


class TestAPIServer:
    @pytest.fixture
    def client(self) -> object:
        from fastapi.testclient import TestClient

        from multi_agent_research_lab.api.server import api
        return TestClient(api)

    def test_health(self, client: object) -> None:
        resp = client.get("/health")  # type: ignore[union-attr]
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "provider" in data

    def test_config(self, client: object) -> None:
        resp = client.get("/research/config")  # type: ignore[union-attr]
        assert resp.status_code == 200
        data = resp.json()
        assert "max_iterations" in data
        assert data["api_port"] == 8080

    def test_baseline_endpoint(self, client: object) -> None:
        resp = client.post(  # type: ignore[union-attr]
            "/research/baseline",
            json={"query": "Explain multi-agent systems", "max_sources": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_type"] == "single-agent-baseline"
        assert data["final_answer"] is not None
        assert data["latency_seconds"] >= 0

    def test_multi_agent_endpoint(self, client: object) -> None:
        resp = client.post(  # type: ignore[union-attr]
            "/research/multi-agent",
            json={"query": "GraphRAG state-of-the-art", "max_sources": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_type"] == "multi-agent"
        assert data["final_answer"] is not None
        assert len(data["route_history"]) > 0

    def test_invalid_query_rejected(self, client: object) -> None:
        resp = client.post("/research/baseline", json={"query": "hi"})  # type: ignore[union-attr]
        assert resp.status_code == 422  # pydantic min_length validation
