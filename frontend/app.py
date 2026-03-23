"""
NeuroAgent Gradio frontend.

Talks to the FastAPI backend at NEUROAGENT_API_URL (default: http://localhost:8000).
Deploy on Hugging Face Spaces by setting NEUROAGENT_API_URL to your Railway URL.

Tabs:
  1. Research  — run the full pipeline, see synthesis + papers + contradictions + RAGAS
  2. Evaluations — browse stored RAGAS scores + aggregate stats
  3. Knowledge Graph — force-directed concept graph via Plotly
  4. System — live health-check of all backend services
"""

import json
import math
import os
from typing import Any

import httpx
import gradio as gr

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("NEUROAGENT_API_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = 120.0  # seconds — LLM pipeline can be slow


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{API_BASE}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()


def _post(path: str, body: dict) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{API_BASE}{path}", json=body)
        resp.raise_for_status()
        return resp.json()


# ── Tab 1: Research ────────────────────────────────────────────────────────────

def run_research(
    query: str,
    user_id: str,
    max_papers: int,
) -> tuple[str, str, str, str, str, str]:
    """
    Returns:
        synthesis, papers_md, contradictions_md, concepts_md, eval_md, errors_md
    """
    if not query or len(query.strip()) < 3:
        msg = "Please enter a research question (at least 3 characters)."
        return msg, "", "", "", "", ""

    payload: dict[str, Any] = {
        "query": query.strip(),
        "max_papers": int(max_papers),
    }
    if user_id.strip():
        payload["user_id"] = user_id.strip()

    try:
        data = _post("/api/research", payload)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        return f"❌ API error: {detail}", "", "", "", "", ""
    except Exception as exc:
        return f"❌ Connection error: {exc}", "", "", "", "", ""

    # ── Synthesis ─────────────────────────────────────────────────────────────
    synthesis = data.get("final_synthesis") or "_No synthesis generated._"

    # ── Papers ────────────────────────────────────────────────────────────────
    n_papers = data.get("papers_fetched", 0)
    n_summaries = data.get("summaries_generated", 0)
    papers_md = f"**Papers fetched:** {n_papers} | **Summaries generated:** {n_summaries}"

    # ── Contradictions ────────────────────────────────────────────────────────
    conflicts = data.get("contradictions", [])
    if conflicts:
        lines = [f"**{len(conflicts)} contradiction(s) detected:**\n"]
        for i, c in enumerate(conflicts, 1):
            lines.append(
                f"**{i}. {c['conflict_type'].upper()}** (confidence: {c['confidence']:.0%})\n"
                f"> {c['description']}\n"
                f"_Papers: `{c['paper_a_id'][:8]}…` vs `{c['paper_b_id'][:8]}…`_\n"
            )
        contradictions_md = "\n".join(lines)
    else:
        contradictions_md = "_No contradictions detected._"

    # ── Concepts ──────────────────────────────────────────────────────────────
    concepts = data.get("concepts", [])
    if concepts:
        tags = " · ".join(f"`{c}`" for c in concepts[:30])
        concepts_md = f"**{len(concepts)} concept(s):** {tags}"
    else:
        concepts_md = "_No concepts extracted._"

    # ── Evaluation ────────────────────────────────────────────────────────────
    ev = data.get("evaluation")
    if ev:
        faith = ev.get("faithfulness") or 0.0
        relevancy = ev.get("answer_relevancy") or 0.0
        precision = ev.get("context_precision") or 0.0
        avg = ev.get("average_score") or round((faith + relevancy + precision) / 3, 3)
        passed = ev.get("passed_quality_threshold") or avg >= 0.7

        err = ev.get("evaluation_error")
        if err and faith == 0.0:
            # Genuine failure — show error text
            eval_md = f"⚠️ Evaluation error: {err}"
        else:
            badge = "✅ PASS" if passed else "⚠️ NEEDS REVIEW"
            note = f"\n\n_{err}_" if err else ""
            eval_md = (
                f"**RAGAS Scores** — {badge}\n\n"
                f"| Metric | Score |\n"
                f"|--------|-------|\n"
                f"| Faithfulness | {faith:.3f} |\n"
                f"| Answer Relevancy | {relevancy:.3f} |\n"
                f"| Context Precision | {precision:.3f} |\n"
                f"| **Average** | **{avg:.3f}** |"
                f"{note}"
            )
    else:
        eval_md = "_Evaluation not available (MongoDB may be offline)._"

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = data.get("errors", [])
    errors_md = "\n".join(f"- {e}" for e in errors) if errors else "_No errors._"

    return synthesis, papers_md, contradictions_md, concepts_md, eval_md, errors_md


# ── Tab 2: Evaluations ────────────────────────────────────────────────────────

def load_evaluations(limit: int) -> tuple[list[list], str]:
    """Returns (table_rows, stats_markdown)."""
    try:
        data = _get("/api/evaluations", {"limit": int(limit), "skip": 0})
        stats_data = _get("/api/evaluations/stats")
    except Exception as exc:
        return [], f"❌ Error: {exc}"

    rows = []
    for ev in data.get("evaluations", []):
        faith = ev.get("faithfulness") or 0.0
        relevancy = ev.get("answer_relevancy") or 0.0
        precision = ev.get("context_precision") or 0.0
        # average_score may be missing from older MongoDB docs (saved before @computed_field fix)
        avg = ev.get("average_score") or round((faith + relevancy + precision) / 3, 3)
        passed = ev.get("passed_quality_threshold") or avg >= 0.7
        rows.append([
            ev.get("session_id", "")[:8] + "…",
            ev.get("query", "")[:60],
            f"{faith:.3f}",
            f"{relevancy:.3f}",
            f"{precision:.3f}",
            f"{avg:.3f}",
            "✅" if passed else "⚠️",
        ])

    total = stats_data.get("total_evaluations", 0)
    if total == 0:
        stats_md = "_No evaluations stored yet. Run a research query first._"
    else:
        pass_rate = stats_data.get("pass_rate", 0) or 0
        stats_md = (
            f"**Total evaluations:** {total} | "
            f"**Valid:** {stats_data.get('valid_evaluations', 0)} | "
            f"**Pass rate (≥0.7):** {pass_rate:.0%}\n\n"
            f"| Metric | Mean Score |\n"
            f"|--------|------------|\n"
            f"| Faithfulness | {stats_data.get('mean_faithfulness') or 0:.3f} |\n"
            f"| Answer Relevancy | {stats_data.get('mean_answer_relevancy') or 0:.3f} |\n"
            f"| Context Precision | {stats_data.get('mean_context_precision') or 0:.3f} |\n"
            f"| **Average** | **{stats_data.get('mean_average_score') or 0:.3f}** |"
        )

    return rows, stats_md


# ── Tab 3: Knowledge Graph ─────────────────────────────────────────────────────

def load_graph(limit: int):
    """Return a Plotly figure of the concept graph."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None, "❌ Install plotly: `pip install plotly`"

    try:
        data = _get("/api/graph/concepts", {"limit": int(limit)})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            return None, "⚠️ Neo4j is not available — knowledge graph disabled."
        return None, f"❌ Error: {exc}"
    except Exception as exc:
        return None, f"❌ Connection error: {exc}"

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        return None, "_Knowledge graph is empty. Run a research query to populate it._"

    # ── Circular layout ───────────────────────────────────────────────────────
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node["id"]] = (math.cos(angle), math.sin(angle))

    # ── Edge traces ───────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for edge in edges:
        x0, y0 = positions.get(edge["source"], (0, 0))
        x1, y1 = positions.get(edge["target"], (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#aaa"),
        hoverinfo="none",
    )

    # ── Node traces ───────────────────────────────────────────────────────────
    node_x = [positions[nd["id"]][0] for nd in nodes]
    node_y = [positions[nd["id"]][1] for nd in nodes]
    node_text = [nd["label"] for nd in nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=10,
            color="#4f8ef7",
            line=dict(width=1, color="#fff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Concept Knowledge Graph — {len(nodes)} nodes, {len(edges)} edges",
                font=dict(size=14),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=520,
        ),
    )

    info = (
        f"**{len(nodes)} concept nodes** | **{len(edges)} edges**  \n"
        f"Hover over nodes to see concept names."
    )
    return fig, info


# ── Tab 4: System health ──────────────────────────────────────────────────────

def check_health() -> str:
    try:
        data = _get("/health")
    except Exception as exc:
        return f"❌ Cannot reach backend at `{API_BASE}` — {exc}"

    overall = data.get("status", "unknown")
    badge = "✅ OK" if overall == "ok" else "⚠️ DEGRADED"
    services = data.get("services", {})

    rows = "\n".join(
        f"| {svc.capitalize()} | {'✅' if ok else '❌'} |"
        for svc, ok in services.items()
    )
    return (
        f"**Backend:** `{API_BASE}`  \n"
        f"**Overall status:** {badge}\n\n"
        f"| Service | Status |\n"
        f"|---------|--------|\n"
        f"{rows}"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    )

    with gr.Blocks(title="NeuroAgent — Multi-Agent Research Assistant", theme=theme) as demo:

        gr.Markdown(
            """
# 🧠 NeuroAgent
**Multi-agent AI research assistant** — fetches academic papers, summarizes them,
detects contradictions, builds a knowledge graph, and evaluates quality with RAGAS.

> All processing happens on the backend (FastAPI + LangGraph). This UI is a thin client.
"""
        )

        # ── Tab 1: Research ───────────────────────────────────────────────────
        with gr.Tab("🔬 Research"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_in = gr.Textbox(
                        label="Research Question",
                        placeholder="e.g. What are the limitations of RAG systems in production?",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    user_id_in = gr.Textbox(
                        label="User ID (optional)",
                        placeholder="Leave blank for anonymous",
                    )
                    max_papers_in = gr.Slider(
                        label="Max papers to fetch",
                        minimum=1,
                        maximum=30,
                        value=10,
                        step=1,
                    )
                    run_btn = gr.Button("Run Research Pipeline", variant="primary")

            gr.Markdown("---")

            synthesis_out = gr.Markdown(label="📝 Synthesis")

            with gr.Row():
                papers_out = gr.Markdown(label="📄 Papers")
                concepts_out = gr.Markdown(label="🏷️ Concepts")

            contradictions_out = gr.Markdown(label="⚡ Contradictions")
            eval_out = gr.Markdown(label="📊 RAGAS Evaluation")

            with gr.Accordion("Pipeline errors", open=False):
                errors_out = gr.Markdown()

            run_btn.click(
                fn=run_research,
                inputs=[query_in, user_id_in, max_papers_in],
                outputs=[synthesis_out, papers_out, contradictions_out,
                         concepts_out, eval_out, errors_out],
                show_progress=True,
            )

        # ── Tab 2: Evaluations ────────────────────────────────────────────────
        with gr.Tab("📊 Evaluations"):
            with gr.Row():
                eval_limit = gr.Slider(
                    label="Number of results",
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=5,
                )
                refresh_eval_btn = gr.Button("Refresh", variant="secondary")

            stats_out = gr.Markdown()

            eval_table = gr.Dataframe(
                headers=["Session", "Query", "Faithfulness", "Relevancy",
                         "Precision", "Avg Score", "Pass"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                label="Evaluation Results",
                wrap=True,
            )

            refresh_eval_btn.click(
                fn=load_evaluations,
                inputs=[eval_limit],
                outputs=[eval_table, stats_out],
            )

            # Load on tab render
            demo.load(
                fn=load_evaluations,
                inputs=[eval_limit],
                outputs=[eval_table, stats_out],
            )

        # ── Tab 3: Knowledge Graph ────────────────────────────────────────────
        with gr.Tab("🕸️ Knowledge Graph"):
            with gr.Row():
                graph_limit = gr.Slider(
                    label="Max concepts to display",
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                )
                refresh_graph_btn = gr.Button("Refresh Graph", variant="secondary")

            graph_info = gr.Markdown()
            graph_plot = gr.Plot(label="Concept Graph")

            refresh_graph_btn.click(
                fn=load_graph,
                inputs=[graph_limit],
                outputs=[graph_plot, graph_info],
            )

        # ── Tab 4: System health ──────────────────────────────────────────────
        with gr.Tab("⚙️ System"):
            gr.Markdown(f"**Backend URL:** `{API_BASE}`")
            health_out = gr.Markdown()
            health_btn = gr.Button("Check Health", variant="secondary")

            health_btn.click(fn=check_health, outputs=[health_out])
            demo.load(fn=check_health, outputs=[health_out])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
