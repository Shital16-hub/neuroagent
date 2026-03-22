"""
Synthesis Agent — citation-grounded research synthesis.

Combines all pipeline outputs into a comprehensive answer:
  - Qdrant semantic search results (RAG context for grounding)
  - Structured paper summaries (claims, methods, findings)
  - Contradiction report (conflicting evidence to acknowledge)
  - Extracted concepts (topical context)
  - Mem0 user context (personalization from prior sessions)

Uses the reasoning LLM (llama-3.3-70b-versatile) for higher-quality
multi-source synthesis compared to the fast model used for summarization.

Reads from state:
    query, summaries, contradiction_report, concepts, mem0_context

Writes to state:
    final_synthesis  — The full citation-grounded answer
    qdrant_context   — Context chunks used (for RAGAS evaluation)
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.models.state import AgentState
from app.models.summary import ContradictionReport, PaperSummary
from app.services.llm_factory import LLMFactory
from app.services.qdrant_service import QdrantService
from app.config import get_settings


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a research synthesis expert. Produce a comprehensive, well-structured "
    "answer to a research question based solely on the provided academic papers. "
    "\n\nGuidelines:"
    "\n- Cite every major claim inline: [paper_id] format"
    "\n- Acknowledge contradictions — present multiple viewpoints when evidence conflicts"
    "\n- Organize with clear sections when the answer is complex"
    "\n- Be specific about what papers actually found — no vague generalizations"
    "\n- End with a 'Key Takeaways' section (3-5 bullet points)"
    "\n- Write for a researcher audience: precise, technical, and readable"
    "\n- Do NOT fabricate findings — only use information from the provided papers"
)

_USER_TEMPLATE = """\
Research Question: {query}

{mem0_section}\
Available Papers ({num_papers} papers):
{papers_block}

{contradictions_section}\
Key Concepts Identified: {concepts}

Based on the papers above, provide a comprehensive synthesis that answers the \
research question. Use [paper_id] for all inline citations.\
"""


# ── Formatting Helpers ─────────────────────────────────────────────────────────

def _format_papers_block(summaries: list[PaperSummary]) -> str:
    """Format paper summaries into a compact block for the LLM prompt."""
    blocks = []
    for s in summaries:
        claims_text = " | ".join(s.key_claims[:3])
        blocks.append(
            f"[{s.paper_id}]\n"
            f"  Claims:   {claims_text}\n"
            f"  Method:   {s.methodology}\n"
            f"  Findings: {s.findings}\n"
            f"  Limits:   {s.limitations}"
        )
    return "\n\n".join(blocks)


def _format_contradictions_section(report: Optional[ContradictionReport]) -> str:
    """Format contradiction report for inclusion in the synthesis prompt."""
    if not report or not report.conflicts:
        return ""

    lines = [f"Known Contradictions ({len(report.conflicts)} detected — address these in your synthesis):"]
    for c in report.conflicts[:5]:  # top 5 to avoid token overflow
        lines.append(
            f"  [{c.conflict_type.upper()} | conf={c.confidence:.2f}] "
            f"{c.paper_a_id} vs {c.paper_b_id}: {c.description[:150]}"
        )
    return "\n".join(lines) + "\n\n"


# ── Agent ──────────────────────────────────────────────────────────────────────

class SynthesisAgent:
    """
    Agent that synthesizes a comprehensive research answer from all pipeline outputs.

    Uses Qdrant for RAG context retrieval, then calls the reasoning LLM to
    produce a citation-grounded synthesis. If Qdrant is unavailable, synthesis
    still proceeds using the paper summaries alone.

    Usage (within LangGraph or standalone):
        agent = SynthesisAgent(qdrant=qdrant_service)
        state = await agent.run(state)
    """

    def __init__(self, qdrant: Optional[QdrantService] = None) -> None:
        self._qdrant = qdrant
        self._settings = get_settings()

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["query"], state["summaries"], state["contradiction_report"],
                 state["concepts"], state["mem0_context"]
        Writes:  state["final_synthesis"], state["qdrant_context"], state["errors"]
        """
        query = state["query"]
        session_id = state["session_id"]
        summaries = state.get("summaries", [])

        if not summaries:
            logger.warning(
                "SynthesisAgent: no summaries available | session={}", session_id
            )
            return {
                **state,
                "final_synthesis": (
                    "Insufficient data: no paper summaries available for synthesis. "
                    "Please try a different query."
                ),
                "qdrant_context": [],
            }

        logger.info(
            "SynthesisAgent starting | session={} summaries={} has_qdrant={}",
            session_id,
            len(summaries),
            self._qdrant is not None,
        )

        # ── Qdrant semantic search for RAG context ───────────────────────────
        qdrant_context: list[str] = []
        if self._qdrant:
            try:
                similar_papers = await self._qdrant.search_similar(
                    query, limit=5, score_threshold=0.25
                )
                qdrant_context = [
                    f"{p.title}: {p.abstract[:400]}" for p in similar_papers
                ]
                logger.debug(
                    "SynthesisAgent: Qdrant context retrieved | chunks={}", len(qdrant_context)
                )
            except Exception as exc:
                logger.warning(
                    "SynthesisAgent: Qdrant search failed — continuing without RAG context "
                    "| session={} error={}",
                    session_id,
                    exc,
                )

        # ── Build prompt ─────────────────────────────────────────────────────
        papers_block = _format_papers_block(summaries)
        contradictions_section = _format_contradictions_section(
            state.get("contradiction_report")
        )

        concepts = state.get("concepts", [])
        concepts_text = ", ".join(concepts[:20]) if concepts else "None extracted"

        mem0_context = state.get("mem0_context", "")
        mem0_section = ""
        if mem0_context:
            mem0_section = (
                f"Your Past Research Context (personalized from memory):\n"
                f"{mem0_context}\n\n"
            )

        user_text = _USER_TEMPLATE.format(
            query=query,
            mem0_section=mem0_section,
            num_papers=len(summaries),
            papers_block=papers_block,
            contradictions_section=contradictions_section,
            concepts=concepts_text,
        )

        # ── LLM call ─────────────────────────────────────────────────────────
        llm = LLMFactory.get_reasoning_llm(temperature=0.2, max_tokens=2048)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]

        new_errors: list[str] = []
        synthesis = ""
        try:
            response = await llm.ainvoke(messages)
            synthesis = response.content.strip()

            # Strip <think> tags from DeepSeek-style models
            import re
            synthesis = re.sub(r"<think>.*?</think>", "", synthesis, flags=re.DOTALL).strip()

            logger.info(
                "SynthesisAgent complete | session={} synthesis_len={}",
                session_id,
                len(synthesis),
            )
        except Exception as exc:
            logger.error(
                "SynthesisAgent LLM call failed | session={} error={}", session_id, exc
            )
            synthesis = f"Synthesis generation failed: {exc}"
            new_errors.append(f"SynthesisAgent: LLM call failed — {exc}")

        return {
            **state,
            "final_synthesis": synthesis,
            "qdrant_context": qdrant_context,
            "errors": state.get("errors", []) + new_errors,
        }
