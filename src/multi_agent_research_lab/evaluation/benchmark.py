"""Benchmark runner: single-agent vs multi-agent — detailed per-dimension scoring.

Scoring dimensions (each 0–10):
  1. factual_accuracy    — VERIFIED / (VERIFIED + UNCERTAIN + UNSUPPORTED) from critic
  2. citation_coverage   — unique [S#] in answer / total sources × 10
  3. structure_score     — headings, bullets, conclusion, word-count range
  4. completeness_score  — query keyword coverage + length adequacy
  5. critic_quality_score — numeric value parsed from critic's "Quality Score: X/10"
  6. hallucination_risk  — LOW→10 / MEDIUM→6 / HIGH→2 (mapped from critic review)

quality_score = weighted average of the above (weights below).
"""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

# Dimension weights (must sum to 1.0)
_WEIGHTS = {
    "factual_accuracy": 0.25,
    "citation_coverage": 0.15,
    "structure_score": 0.15,
    "completeness_score": 0.20,
    "critic_quality_score": 0.15,
    "hallucination_risk_score": 0.10,
}

# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------


def score_factual_accuracy(state: ResearchState) -> float:
    """Parse [VERIFIED] / [UNCERTAIN] / [UNSUPPORTED] from critic notes.

    Score = (verified + 0.5 × uncertain) / total_checked × 10
    Returns 5.0 if no critic notes (neutral/unknown).
    """
    if not state.critic_notes:
        return 5.0
    text = state.critic_notes
    verified = len(re.findall(r"\[VERIFIED\]", text, re.IGNORECASE))
    uncertain = len(re.findall(r"\[UNCERTAIN\]", text, re.IGNORECASE))
    unsupported = len(re.findall(r"\[UNSUPPORTED\]", text, re.IGNORECASE))
    total = verified + uncertain + unsupported
    if total == 0:
        return 5.0
    raw = (verified + 0.5 * uncertain) / total
    return round(min(raw * 10, 10.0), 1)


def score_citation_coverage(state: ResearchState) -> float:
    """Fraction of sources cited inline in the final answer × 10."""
    if not state.final_answer or not state.sources:
        return 0.0
    cited_indices = set(re.findall(r"\[S(\d+)\]", state.final_answer))
    total = len(state.sources)
    coverage = len(cited_indices) / total if total else 0.0
    return round(min(coverage * 10, 10.0), 1)


def score_structure(state: ResearchState) -> float:
    """Award points for well-structured markdown answer.

    Rubric (10 pts total):
      2 pts — word count in target range 300–800
      2 pts — ≥3 H2/H3 headings
      1 pt  — ≥4 bullet-list items
      2 pts — has explicit Conclusion section
      1 pt  — has an intro / summary paragraph
      2 pts — no error/fallback markers
    """
    if not state.final_answer:
        return 0.0
    answer = state.final_answer
    score = 0.0

    words = len(answer.split())
    if 300 <= words <= 800:
        score += 2.0
    elif 150 <= words < 300 or 800 < words <= 1200:
        score += 1.0

    headings = len(re.findall(r"^#{1,3} ", answer, re.MULTILINE))
    if headings >= 3:
        score += 2.0
    elif headings >= 1:
        score += 1.0

    bullets = len(re.findall(r"^[-*] ", answer, re.MULTILINE))
    if bullets >= 4:
        score += 1.0
    elif bullets >= 2:
        score += 0.5

    if re.search(r"#{1,3}\s*conclusion", answer, re.IGNORECASE):
        score += 2.0
    elif "conclusion" in answer.lower():
        score += 1.0

    # Intro: first 100 chars after title has substantive text
    body = re.sub(r"^#[^\n]*\n", "", answer, count=1).strip()
    if len(body) > 80:
        score += 1.0

    if "[fallback]" not in answer.lower() and "[error]" not in answer.lower():
        score += 2.0

    return round(min(score, 10.0), 1)


def score_completeness(state: ResearchState) -> float:
    """How well the answer covers the query intent.

    Rubric (10 pts):
      4 pts — query keywords present in answer (≥60% coverage)
      3 pts — answer mentions ≥3 sources or concepts from research notes
      2 pts — answer length is substantial (≥200 words)
      1 pt  — multi-agent: has analysis_notes feeding the answer
    """
    if not state.final_answer:
        return 0.0
    answer = state.final_answer.lower()
    score = 0.0

    # Keyword coverage
    query_words = set(
        w for w in re.findall(r"\b\w{4,}\b", state.request.query.lower()) if w not in _STOPWORDS
    )
    if query_words:
        matched = sum(1 for w in query_words if w in answer)
        coverage = matched / len(query_words)
        if coverage >= 0.6:
            score += 4.0
        elif coverage >= 0.3:
            score += 2.0

    # Concepts from research
    if state.research_notes:
        research_words = set(re.findall(r"\b\w{6,}\b", state.research_notes.lower()))
        overlap = sum(1 for w in research_words if w in answer)
        if overlap >= 20:
            score += 3.0
        elif overlap >= 10:
            score += 1.5

    words = len(state.final_answer.split())
    if words >= 200:
        score += 2.0
    elif words >= 100:
        score += 1.0

    if state.analysis_notes:
        score += 1.0

    return round(min(score, 10.0), 1)


def parse_critic_quality_score(state: ResearchState) -> float | None:
    """Extract the numeric quality score from critic notes.

    Looks for patterns like: "Quality Score: 9/10" or "### Quality Score: 8/10"
    Returns None if no critic notes or no match.
    """
    if not state.critic_notes:
        return None
    pattern = r"quality\s+score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10"
    match = re.search(pattern, state.critic_notes, re.IGNORECASE)
    if match:
        return round(float(match.group(1)), 1)
    return None


def parse_hallucination_risk(state: ResearchState) -> tuple[str | None, float]:
    """Extract hallucination risk level and convert to numeric score.

    Returns (risk_label, score_0_to_10).
    LOW→10, MEDIUM→6, HIGH→2, None→5 (unknown).
    """
    if not state.critic_notes:
        return None, 5.0
    text = state.critic_notes.upper()
    match = re.search(r"hallucination\s+risk[:\s]+(\w+)", text, re.IGNORECASE)
    if not match:
        return None, 5.0
    risk = match.group(1).upper()
    mapping = {"LOW": (10.0, "LOW"), "MEDIUM": (6.0, "MEDIUM"), "HIGH": (2.0, "HIGH")}
    score, label = mapping.get(risk, (5.0, risk))
    return label, score


def _count_total_tokens(state: ResearchState) -> int:
    total = 0
    for r in state.agent_results:
        total += int(r.metadata.get("input_tokens") or 0)
        total += int(r.metadata.get("output_tokens") or 0)
    return total


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "been", "were", "they",
    "their", "which", "what", "when", "also", "more", "into", "over", "each",
}


def compute_detailed_metrics(
    run_name: str,
    query: str,
    state: ResearchState,
    latency: float,
    failed: bool,
) -> BenchmarkMetrics:
    """Compute all dimension scores and aggregate into BenchmarkMetrics."""

    factual = score_factual_accuracy(state)
    citation = score_citation_coverage(state)
    structure = score_structure(state)
    completeness = score_completeness(state)
    critic_score = parse_critic_quality_score(state)
    hallucination_label, hallucination_score = parse_hallucination_risk(state)

    # Weighted overall score
    critic_for_weight = critic_score if critic_score is not None else 5.0
    quality = (
        factual * _WEIGHTS["factual_accuracy"]
        + citation * _WEIGHTS["citation_coverage"]
        + structure * _WEIGHTS["structure_score"]
        + completeness * _WEIGHTS["completeness_score"]
        + critic_for_weight * _WEIGHTS["critic_quality_score"]
        + hallucination_score * _WEIGHTS["hallucination_risk_score"]
    )
    quality = round(min(quality, 10.0), 1)

    word_count = len(state.final_answer.split()) if state.final_answer else 0
    total_tokens = _count_total_tokens(state)

    cost = round(
        sum(float(r.metadata.get("cost_usd") or 0) for r in state.agent_results), 6
    )

    notes_parts = [
        f"iter={state.iteration}",
        f"words={word_count}",
        f"tokens={total_tokens}",
    ]
    if failed:
        notes_parts.append("FAILED")

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost if cost > 0 else None,
        quality_score=quality,
        factual_accuracy=factual,
        citation_coverage=citation,
        structure_score=structure,
        completeness_score=completeness,
        critic_quality_score=critic_score,
        hallucination_risk=hallucination_label,
        answer_word_count=word_count,
        total_tokens=total_tokens if total_tokens > 0 else None,
        iterations_used=state.iteration,
        failure_count=len(state.errors),
        notes=" | ".join(notes_parts),
    )


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a single trial and return (state, detailed metrics)."""
    logger.info("Benchmark [%s]: %s", run_name, query[:60])
    started = perf_counter()

    failed = False
    try:
        state = runner(query)
        failed = bool(state.errors)
    except Exception as exc:
        logger.error("Benchmark runner failed: %s", exc)
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        failed = True

    latency = perf_counter() - started
    metrics = compute_detailed_metrics(run_name, query, state, latency, failed)

    logger.info(
        "Benchmark [%s] done: latency=%.2fs quality=%.1f "
        "factual=%.1f citation=%.1f structure=%.1f completeness=%.1f",
        run_name,
        latency,
        metrics.quality_score or 0,
        metrics.factual_accuracy or 0,
        metrics.citation_coverage or 0,
        metrics.structure_score or 0,
        metrics.completeness_score or 0,
    )
    return state, metrics


def run_comparison(
    query: str,
    baseline_runner: Runner,
    multi_agent_runner: Runner,
) -> list[tuple[ResearchState, BenchmarkMetrics]]:
    """Run baseline and multi-agent on the same query."""
    return [
        run_benchmark("single-agent-baseline", query, baseline_runner),
        run_benchmark("multi-agent", query, multi_agent_runner),
    ]
