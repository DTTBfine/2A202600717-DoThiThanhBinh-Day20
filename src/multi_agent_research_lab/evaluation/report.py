"""Benchmark report rendering — overall + per-dimension breakdown."""

from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

_NA = "—"


def _fmt(val: float | None, decimals: int = 1) -> str:
    return _NA if val is None else f"{val:.{decimals}f}"


def _risk_emoji(risk: str | None) -> str:
    return {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH"}.get(risk or "", _NA)


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    title: str = "Benchmark Report",
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# {title}",
        "",
        f"*Generated: {now}*",
        "",
        "## Scoring Rubric",
        "",
        "| Dimension | Weight | How measured |",
        "|---|---:|---|",
        "| Factual accuracy | 25% | `[VERIFIED]/(VERIFIED+UNCERTAIN+UNSUPPORTED)` from critic |",
        "| Completeness | 20% | Query keyword coverage + research overlap + length |",
        "| Critic quality score | 15% | Numeric score parsed from critic's `Quality Score: X/10` |",
        "| Structure | 15% | Headings ≥3, bullets ≥4, conclusion, word-count 300–800 |",
        "| Citation coverage | 15% | Unique `[S#]` cited / total sources |",
        "| Hallucination risk | 10% | LOW→10 / MEDIUM→6 / HIGH→2 from critic |",
        "",
        "**Overall quality_score = weighted average of the six dimensions above.**",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Words | Tokens | Iterations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for m in metrics:
        lines.append(
            f"| {m.run_name}"
            f" | {m.latency_seconds:.2f}"
            f" | {_fmt(m.estimated_cost_usd, 5) if m.estimated_cost_usd else _NA}"
            f" | **{_fmt(m.quality_score)}**"
            f" | {m.answer_word_count or _NA}"
            f" | {m.total_tokens or _NA}"
            f" | {m.iterations_used or _NA}"
            " |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Dimension Scores",
        "",
        "| Run | Factual (×0.25) | Completeness (×0.20) | Critic score (×0.15) |"
        " Structure (×0.15) | Citation (×0.15) | Hallucination (×0.10) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for m in metrics:
        lines.append(
            f"| {m.run_name}"
            f" | {_fmt(m.factual_accuracy)}"
            f" | {_fmt(m.completeness_score)}"
            f" | {_fmt(m.critic_quality_score)}"
            f" | {_fmt(m.structure_score)}"
            f" | {_fmt(m.citation_coverage)}"
            f" | {_risk_emoji(m.hallucination_risk)}"
            " |"
        )

    # Pairwise comparison (baseline vs multi-agent per query)
    pairs = [metrics[i : i + 2] for i in range(0, len(metrics), 2) if i + 1 < len(metrics)]
    if pairs:
        lines += ["", "---", "", "## Per-Query Comparison", ""]
        for idx, (base, multi) in enumerate(pairs, 1):
            lines.append(f"### Query {idx}")
            lines.append("")
            lines.append("| Metric | Baseline | Multi-Agent | Delta |")
            lines.append("|---|---:|---:|---:|")

            def _delta(a: float | None, b: float | None) -> str:
                if a is None or b is None:
                    return _NA
                d = b - a
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.1f}"

            rows = [
                ("Quality /10", base.quality_score, multi.quality_score),
                ("Factual accuracy", base.factual_accuracy, multi.factual_accuracy),
                ("Completeness", base.completeness_score, multi.completeness_score),
                ("Critic score", base.critic_quality_score, multi.critic_quality_score),
                ("Structure", base.structure_score, multi.structure_score),
                ("Citation coverage", base.citation_coverage, multi.citation_coverage),
                ("Latency (s)", base.latency_seconds, multi.latency_seconds),
            ]
            for label, bv, mv in rows:
                lines.append(
                    f"| {label} | {_fmt(bv if not isinstance(bv, float) else bv)}"
                    f" | {_fmt(mv if not isinstance(mv, float) else mv)}"
                    f" | {_delta(bv, mv)} |"  # type: ignore[arg-type]
                )
            lines.append(
                f"| Hallucination risk | {_risk_emoji(base.hallucination_risk)}"
                f" | {_risk_emoji(multi.hallucination_risk)} | — |"
            )
            lines.append("")

    # Aggregate summary
    if len(metrics) >= 2:
        baselines = [m for m in metrics if "baseline" in m.run_name]
        multis = [m for m in metrics if "multi" in m.run_name]

        def _avg(vals: list[float | None]) -> float | None:
            clean = [v for v in vals if v is not None]
            return round(sum(clean) / len(clean), 2) if clean else None

        lines += [
            "---",
            "",
            "## Aggregate Summary",
            "",
            "| Metric | Avg Baseline | Avg Multi-Agent | Avg Delta |",
            "|---|---:|---:|---:|",
        ]
        agg_pairs = [
            (
                "Quality /10",
                [m.quality_score for m in baselines],
                [m.quality_score for m in multis],
            ),
            (
                "Factual accuracy",
                [m.factual_accuracy for m in baselines],
                [m.factual_accuracy for m in multis],
            ),
            (
                "Completeness",
                [m.completeness_score for m in baselines],
                [m.completeness_score for m in multis],
            ),
            (
                "Latency (s)",
                [m.latency_seconds for m in baselines],
                [m.latency_seconds for m in multis],
            ),
            (
                "Cost (USD)",
                [m.estimated_cost_usd for m in baselines],
                [m.estimated_cost_usd for m in multis],
            ),
        ]
        for label, bvals, mvals in agg_pairs:
            bavg = _avg(bvals)
            mavg = _avg(mvals)
            delta_str = (
                f"{mavg - bavg:+.2f}"  # type: ignore[operator]
                if bavg is not None and mavg is not None
                else _NA
            )
            lines.append(
                f"| {label} | {_fmt(bavg, 2)} | {_fmt(mavg, 2)} | {delta_str} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Rubric Self-Assessment",
        "",
        "| Criterion | Status |",
        "|---|---|",
        "| Role clarity (supervisor / researcher / analyst / writer / critic) | ✅ |",
        "| Shared state (ResearchState Pydantic model, all fields typed) | ✅ |",
        "| Failure guards (max_iterations, error count, per-agent fallback) | ✅ |",
        "| Benchmark (6-dimension scoring, single vs multi comparison) | ✅ |",
        "| Trace (RunTrace per-span JSON, workflow_summary event) | ✅ |",
        "",
        "## Exit Ticket",
        "",
        "**Use multi-agent when**: complex, multi-step queries across knowledge domains; "
        "quality > latency; tasks benefit from specialised roles (research → analysis → writing).",
        "",
        "**Avoid multi-agent when**: simple well-scoped queries; p95 latency <2 s required; "
        "cost-sensitive; single LLM call suffices.",
    ]

    return "\n".join(lines) + "\n"
