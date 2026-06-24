"""LLM client abstraction.

Agents depend on this interface instead of importing an SDK directly.
Provider selection: Anthropic (if key set) → OpenAI (if key set) → Mock.
"""

import logging
from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Cost per 1K tokens (USD) — approximate
_COST_INPUT: dict[str, float] = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.005,
    "claude-haiku-4-5-20251001": 0.00025,
    "claude-sonnet-4-6": 0.003,
}
_COST_OUTPUT: dict[str, float] = {
    "gpt-4o-mini": 0.0006,
    "gpt-4o": 0.015,
    "claude-haiku-4-5-20251001": 0.00125,
    "claude-sonnet-4-6": 0.015,
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str = ""


# ---------------------------------------------------------------------------
# Mock provider — deterministic responses for testing / no-key environments
# ---------------------------------------------------------------------------

_MOCK_RESEARCH = """## Research Notes

**Topic**: {query}

### Key Findings
1. Recent advances in this area show significant progress in scalability and accuracy.
2. Benchmark results (2024–2025) indicate 15–30% quality improvements over prior methods.
3. Production deployments report latency p95 under 2 seconds for most use cases.
4. Leading open-source implementations: LangGraph, AutoGen, CrewAI, and OpenAI Agents SDK.
5. Cost analysis: multi-step pipelines add ~20-40% overhead vs single-agent but improve quality.

### Sources
- [S1] "Survey of Multi-Agent LLM Architectures" (arXiv 2024)
- [S2] "Production Guardrails for LLM Systems" (ML Engineering Blog 2025)
- [S3] "Benchmarking Agentic Pipelines" (NeurIPS Workshop 2024)
- [S4] "GraphRAG: Graph-Enhanced Retrieval" (Microsoft Research 2024)
"""

_MOCK_ANALYSIS = """## Analysis Notes

**Based on research about**: {query}

### Key Claims
1. **Strongest evidence** [S1, S4]: Graph-based retrieval improves multi-hop reasoning by 25-40%.
2. **Moderate evidence** [S2]: Guardrails (max_iterations, timeout, retry) reduce failure rate
   from 18% to 3%.
3. **Weak evidence** [S3]: Quality score improvement depends heavily on task complexity.

### Comparison: Single-Agent vs Multi-Agent
| Dimension | Single-Agent | Multi-Agent |
|---|---|---|
| Latency | ~2s | ~6-12s |
| Quality (0-10) | 6.2 | 8.1 |
| Cost per query | $0.002 | $0.006 |
| Failure rate | 12% | 4% |

### Key Risks
- Multi-agent adds orchestration complexity; failure modes multiply.
- Token cost scales linearly with agents and iterations.
- State handoff requires careful schema design to avoid information loss.

### Recommendation
Use multi-agent when: complex reasoning, multiple knowledge domains, or quality > latency.
Use single-agent when: simple queries, cost-sensitive, latency-critical.
"""

_MOCK_ANSWER = """# {query}

## Executive Summary
Based on comprehensive research and analysis, this report synthesizes current state-of-the-art
findings on the requested topic.

## Findings

### 1. Current State
Modern approaches leverage graph-enhanced retrieval and multi-step reasoning pipelines.
Benchmark results show 25-40% quality improvement over traditional single-step methods [S1, S4].

### 2. Key Techniques
- **Supervisor-Worker pattern**: A routing agent delegates to specialized sub-agents
- **Shared state**: All agents read/write a single Pydantic state object (ResearchState)
- **Guardrails**: max_iterations=10, timeout=120s, retry with exponential backoff

### 3. Production Considerations
Deployment requires: structured logging, distributed tracing (LangSmith/Langfuse),
benchmark baselines, and human-in-the-loop for high-stakes outputs [S2].

### 4. Benchmark Results (this run)
- Single-agent baseline: latency ~2s, quality 6.2/10
- Multi-agent system: latency ~8s, quality 8.1/10
- Quality gain: +30% at 3x cost

## Conclusion
Multi-agent architectures deliver measurably better quality for complex research tasks.
The tradeoff (latency, cost) is justified when output quality is critical.

*Sources: [S1] arXiv 2024 · [S2] ML Eng Blog 2025 · [S3] NeurIPS 2024 · [S4] MSFT Research 2024*
"""

_MOCK_CRITIC = """## Critic Review

### Fact-Check Results
- [VERIFIED] Graph-based retrieval 25-40% improvement — consistent with [S1, S4]
- [VERIFIED] Guardrails reduce failure rate 18% → 3% — supported by [S2]
- [UNCERTAIN] Benchmark latency figures are illustrative; exact values are environment-dependent
- [OK] Source citations present and internally consistent

### Citation Coverage: 4/4 major claims cited (100%)

### Hallucination Risk: LOW
The answer stays within the scope of the research notes and avoids speculative claims.

### Recommended edits: None critical. Consider adding confidence intervals for benchmark numbers.
"""


class _MockLLMClient:
    """Deterministic mock — no API key required."""

    model = "mock-v1"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        sp = system_prompt.lower()
        up = user_prompt

        # Extract query hint from user prompt
        query = "the requested topic"
        for line in up.splitlines():
            if "query:" in line.lower() or "question:" in line.lower() or "topic:" in line.lower():
                query = line.split(":", 1)[-1].strip()
                break
        if not query or query == "the requested topic":
            # take first meaningful line
            lines = [line.strip() for line in up.splitlines() if len(line.strip()) > 10]
            if lines:
                query = lines[0][:80]

        # Use agent-role markers at the START of the system prompt (most specific first)
        if "researcher agent" in sp:
            content = _MOCK_RESEARCH.format(query=query)
        elif "analyst agent" in sp:
            content = _MOCK_ANALYSIS.format(query=query)
        elif "writer agent" in sp:
            content = _MOCK_ANSWER.format(query=query)
        elif "critic agent" in sp:
            content = _MOCK_CRITIC
        elif "supervisor" in sp or "route" in sp or "next agent" in sp:
            content = '{"next": "done"}'
        elif "research assistant" in sp or "answer" in sp:
            content = _MOCK_ANSWER.format(query=query)
        else:
            content = f"Mock response for: {query}"

        tokens_in = len(system_prompt.split()) + len(user_prompt.split())
        tokens_out = len(content.split())
        return LLMResponse(
            content=content,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=0.0,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class _AnthropicLLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic  # type: ignore[import-untyped]

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import anthropic  # type: ignore[import-untyped]

        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = msg.content[0].text if msg.content else ""
            in_tok = msg.usage.input_tokens
            out_tok = msg.usage.output_tokens
            cost = (in_tok / 1000) * _COST_INPUT.get(self.model, 0.0003) + (
                out_tok / 1000
            ) * _COST_OUTPUT.get(self.model, 0.0015)
            logger.debug(
                "Anthropic %s — in=%d out=%d cost=$%.5f", self.model, in_tok, out_tok, cost
            )
            return LLMResponse(
                content=content,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                model=self.model,
            )
        except anthropic.RateLimitError:
            logger.warning("Anthropic rate limit, retrying...")
            raise
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class _OpenAILLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        import openai  # type: ignore[import-untyped]

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import openai  # type: ignore[import-untyped]

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
            )
            content = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            cost = (in_tok / 1000) * _COST_INPUT.get(self.model, 0.00015) + (
                out_tok / 1000
            ) * _COST_OUTPUT.get(self.model, 0.0006)
            logger.debug("OpenAI %s — in=%d out=%d cost=$%.5f", self.model, in_tok, out_tok, cost)
            return LLMResponse(
                content=content,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                model=self.model,
            )
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit, retrying...")
            raise
        except openai.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """Provider-agnostic LLM client. Auto-selects backend from Settings."""

    _settings: Settings = field(default_factory=get_settings)
    _backend: object = field(init=False)

    def __post_init__(self) -> None:
        provider = self._settings.llm_provider
        if provider == "anthropic":
            self._backend = _AnthropicLLMClient(
                api_key=self._settings.anthropic_api_key,  # type: ignore[arg-type]
                model=self._settings.anthropic_model,
            )
            logger.info("LLM backend: Anthropic (%s)", self._settings.anthropic_model)
        elif provider == "openai":
            self._backend = _OpenAILLMClient(
                api_key=self._settings.openai_api_key,  # type: ignore[arg-type]
                model=self._settings.openai_model,
            )
            logger.info("LLM backend: OpenAI (%s)", self._settings.openai_model)
        else:
            self._backend = _MockLLMClient()
            logger.info("LLM backend: Mock (no API key configured)")

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Retry and cost tracking handled internally."""
        return self._backend.complete(system_prompt, user_prompt)  # type: ignore[union-attr]

    @property
    def model_name(self) -> str:
        return getattr(self._backend, "model", "unknown")
