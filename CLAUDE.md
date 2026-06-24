# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (dev + llm extras)
pip install -e "[dev,llm]"

# Run tests
make test
pytest tests/test_state.py          # run a single test file
pytest -k "test_state_records"      # run a single test by name

# Lint and format
make lint       # ruff check
make format     # ruff format

# Type checking
make typecheck  # mypy src (strict mode)

# Run the CLI
malab baseline --query "..."
malab multi-agent --query "..."
python -m multi_agent_research_lab.cli baseline --query "..."

# Find all student TODO markers
grep -R "TODO(student)" -n src tests docs
```

Pre-commit hooks run `ruff --fix` and `ruff-format` automatically on commit. Install with `pre-commit install`.

## Architecture

This is a **lab scaffold** for a multi-agent research system. Most of the core logic is intentionally left as `TODO(student)` — `StudentTodoError` is raised wherever learners must implement logic. The skeleton is already wired up; students fill in the implementations.

### Data flow

```
CLI (cli.py)
  └── MultiAgentWorkflow.run()          # graph/workflow.py — TODO
        └── SupervisorAgent.run()       # agents/supervisor.py — TODO (routes to next agent)
              ├── ResearcherAgent       # agents/researcher.py — TODO
              ├── AnalystAgent          # agents/analyst.py — TODO
              ├── WriterAgent           # agents/writer.py — TODO
              └── CriticAgent (opt.)    # agents/critic.py — TODO
```

All agents share a single `ResearchState` (Pydantic model) that flows through the LangGraph workflow. State fields accumulate: `sources → research_notes → analysis_notes → final_answer`.

### Key modules

| Module | Role |
|---|---|
| `core/schemas.py` | Public Pydantic types: `ResearchQuery`, `AgentResult`, `SourceDocument`, `BenchmarkMetrics`, `AgentName` enum |
| `core/state.py` | `ResearchState` — single source of truth; has `record_route()` and `add_trace_event()` helpers |
| `core/config.py` | `Settings` (pydantic-settings); use `get_settings()` singleton — never read env vars directly in agents |
| `core/errors.py` | `StudentTodoError`, `AgentExecutionError`, `ValidationError` |
| `agents/base.py` | `BaseAgent` ABC — all agents implement `run(state) -> state` |
| `services/llm_client.py` | `LLMClient.complete(system, user) -> LLMResponse` — TODO; keep retry/timeout/token logging here |
| `services/search_client.py` | Web/search client skeleton — TODO |
| `services/storage.py` | Storage client skeleton — TODO |
| `graph/workflow.py` | `MultiAgentWorkflow.build()` / `.run()` — wire LangGraph nodes here |
| `observability/tracing.py` | Tracing hooks — TODO (LangSmith, Langfuse, or OpenTelemetry) |
| `evaluation/benchmark.py` | Benchmark runner skeleton — TODO |
| `evaluation/report.py` | Report generator — saves to `reports/` |

### Configuration

Runtime config comes from `.env` (copy `.env.example`). Key variables:

- `OPENAI_API_KEY` / `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `LANGSMITH_API_KEY`, `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional tracing
- `TAVILY_API_KEY` — optional search
- `MAX_ITERATIONS` (default: 6), `TIMEOUT_SECONDS` (default: 60)

Per-agent model/temperature overrides live in `configs/lab_default.yaml`.

### Conventions

- All agent inputs/outputs must use the Pydantic schemas in `core/schemas.py`.
- Agents must not import LLM SDKs directly — go through `LLMClient`.
- Keep orchestration logic in `graph/workflow.py`, not inside agents.
- `ResearchState.iteration` is auto-incremented by `record_route()`. Enforce `max_iterations` in the supervisor to prevent infinite loops.
- Python ≥ 3.11, strict mypy, ruff line-length 100.
