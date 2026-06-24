"""Command-line entrypoint for the multi-agent research lab."""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_comparison
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI", no_args_is_help=True)
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _build_baseline_runner() -> object:
    """Single-agent: one LLM call, no workflow."""
    llm = LLMClient()
    search = SearchClient()

    def _run(query: str) -> ResearchState:

        req = ResearchQuery(query=query)
        state = ResearchState(request=req)

        sources = search.search(query, max_results=3)
        state.sources = sources
        context = "\n".join(f"[S{i+1}] {s.title}: {s.snippet}" for i, s in enumerate(sources))

        system = (
            "You are a research assistant. Answer the query thoroughly using the provided sources. "
            "Include [S#] citations. Format with markdown headings. End with a Conclusion."
        )
        user = f"Query: {query}\n\nSources:\n{context}"
        resp = llm.complete(system, user)
        state.final_answer = resp.content
        state.research_notes = resp.content  # single-agent collapses all steps
        return state

    return _run


def _build_multi_runner() -> object:
    """Multi-agent workflow runner."""
    wf = MultiAgentWorkflow()

    def _run(query: str) -> ResearchState:
        req = ResearchQuery(query=query)
        state = ResearchState(request=req)
        return wf.run(state)

    return _run


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""
    _init()
    settings = get_settings()
    console.print(f"[bold]Provider:[/bold] {settings.llm_provider}")

    runner = _build_baseline_runner()
    state = runner(query)  # type: ignore[operator]

    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    json_out: Annotated[bool, typer.Option("--json", help="Output raw JSON")] = False,
) -> None:
    """Run the full multi-agent workflow."""
    _init()
    settings = get_settings()
    console.print(f"[bold]Provider:[/bold] {settings.llm_provider}")

    wf = MultiAgentWorkflow()
    req = ResearchQuery(query=query)
    state = ResearchState(request=req)
    result = wf.run(state)

    if json_out:
        console.print(result.model_dump_json(indent=2))
        return

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))

    if result.critic_notes:
        console.print(Panel.fit(result.critic_notes, title="Critic Review", style="cyan"))

    t = Table(title="Route history")
    t.add_column("Step", style="bold")
    t.add_column("Agent")
    for i, route in enumerate(result.route_history, 1):
        t.add_row(str(i), route)
    console.print(t)

    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="Errors", style="red"))


@app.command()
def benchmark(
    queries_file: Annotated[
        str | None, typer.Option("--queries", "-q", help="JSON file with list of queries")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "reports/benchmark_report.md",
    trace_output: Annotated[str, typer.Option("--trace")] = "reports/trace.json",
) -> None:
    """Run single-agent vs multi-agent benchmark and save report."""
    _init()
    settings = get_settings()
    console.print(f"[bold]Provider:[/bold] {settings.llm_provider}")

    # Load queries
    if queries_file:
        with open(queries_file) as f:
            queries: list[str] = json.load(f)
    else:
        import yaml  # type: ignore[import-untyped]
        with open("configs/lab_default.yaml") as f:
            cfg = yaml.safe_load(f)
        queries = cfg["benchmark"]["queries"]

    console.print(f"Running benchmark on {len(queries)} queries…")

    baseline_runner = _build_baseline_runner()
    multi_runner = _build_multi_runner()

    all_metrics = []
    all_traces: list[dict] = []  # type: ignore[type-arg]

    for q in queries:
        console.print(f"\n[yellow]Query:[/yellow] {q[:70]}")
        pairs = run_comparison(q, baseline_runner, multi_runner)  # type: ignore[arg-type]
        for state, metrics in pairs:
            all_metrics.append(metrics)
            all_traces.append(
                {
                    "run": metrics.run_name,
                    "query": q,
                    "route_history": state.route_history,
                    "errors": state.errors,
                    "metrics": metrics.model_dump(),
                }
            )
            _na = "—"
            console.print(
                f"  [green]{metrics.run_name}[/green] "
                f"quality=[bold]{metrics.quality_score or _na}/10[/bold] | "
                f"factual={metrics.factual_accuracy or _na} "
                f"complete={metrics.completeness_score or _na} "
                f"structure={metrics.structure_score or _na} "
                f"cite={metrics.citation_coverage or _na} "
                f"critic={metrics.critic_quality_score or _na} "
                f"halluc={metrics.hallucination_risk or _na} | "
                f"latency={metrics.latency_seconds:.1f}s "
                f"words={metrics.answer_word_count or _na}"
            )

    # Save report
    store = LocalArtifactStore()
    report_md = render_markdown_report(all_metrics)
    saved = store.write_text(output.replace("reports/", ""), report_md)
    console.print(f"\n[bold green]Report saved:[/bold green] {saved}")

    # Save trace
    import json as _json
    trace_saved = store.write_text(
        trace_output.replace("reports/", ""), _json.dumps(all_traces, indent=2, default=str)
    )
    console.print(f"[bold green]Trace saved:[/bold green] {trace_saved}")

    # ── Summary table ──────────────────────────────────────────────────────
    t = Table(title="Benchmark Summary — All Dimensions")
    t.add_column("Run", min_width=22)
    t.add_column("Quality", justify="right")
    t.add_column("Factual", justify="right")
    t.add_column("Complete", justify="right")
    t.add_column("Critic", justify="right")
    t.add_column("Structure", justify="right")
    t.add_column("Citation", justify="right")
    t.add_column("Halluc.", justify="right")
    t.add_column("Lat.(s)", justify="right")

    def _f(v: float | None) -> str:
        return "—" if v is None else f"{v:.1f}"

    for m in all_metrics:
        style = "bold green" if "multi" in m.run_name else ""
        t.add_row(
            m.run_name,
            _f(m.quality_score),
            _f(m.factual_accuracy),
            _f(m.completeness_score),
            _f(m.critic_quality_score),
            _f(m.structure_score),
            _f(m.citation_coverage),
            m.hallucination_risk or "—",
            f"{m.latency_seconds:.1f}",
            style=style,
        )
    console.print(t)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
) -> None:
    """Start the REST API server (default port 8080)."""
    _init()
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install 'uvicorn[standard]'[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"[bold green]Starting API server on http://{host}:{port}[/bold green]")
    uvicorn.run(
        "multi_agent_research_lab.api.server:api",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    app()
