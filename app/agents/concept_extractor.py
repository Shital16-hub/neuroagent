"""
Concept Extractor Agent — builds the Neo4j knowledge graph.

Extracts key technical concepts from paper summaries using an LLM,
then writes Paper nodes, Concept nodes, and MENTIONS edges to Neo4j.

If Neo4j is not reachable (not configured or connection fails), the agent
logs a warning and continues — concept extraction from LLM still runs and
populates state.concepts even without the graph write.
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.models.paper import Paper
from app.models.state import AgentState
from app.models.summary import PaperSummary
from app.services.llm_factory import LLMFactory
from app.services.neo4j_service import Neo4jService

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a scientific knowledge extractor. "
    "Return only valid JSON — no markdown fences, no explanation, no extra text."
)

_USER_TEMPLATE = """\
Extract key technical concepts from these research paper summaries.

{summaries_block}

Return exactly this JSON structure:
{{
  "concepts": ["concept one", "concept two", "concept three"]
}}

Rules:
- Include: methods, architectures, datasets, metrics, algorithms, tasks
- Each concept: 1–4 words, lowercase
- Maximum 20 concepts total, no duplicates
- Prefer specific terms over generic ones (e.g. "attention mechanism" not "technique")
- Return ONLY valid JSON. No markdown, no explanation."""


def _format_summaries_block(summaries: list[PaperSummary]) -> str:
    """Compact summary block for concept extraction prompt."""
    lines = []
    for s in summaries:
        lines.append(f"Paper {s.paper_id}:")
        lines.append(f"  Claims: {' | '.join(s.key_claims)}")
        lines.append(f"  Method: {s.methodology}")
        lines.append(f"  Findings: {s.findings}")
        lines.append("")
    return "\n".join(lines)


def _extract_json(raw: str) -> Optional[dict]:
    """Extract JSON from LLM response, stripping DeepSeek think tags and fences."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


class ConceptExtractorAgent:
    """
    Specialist agent that extracts concepts and writes the Neo4j knowledge graph.

    Neo4j graph written:
      (:Paper {paper_id, title, year}) nodes
      (:Concept {name}) nodes
      (:Paper)-[:MENTIONS]->(:Concept) edges

    If Neo4j is unavailable, concept extraction from LLM still runs and
    state.concepts is populated — only the graph write is skipped.

    Usage (within LangGraph):
        agent = ConceptExtractorAgent(neo4j=neo4j_service)
        state = await agent.run(state)
    """

    def __init__(self, neo4j: Optional[Neo4jService] = None) -> None:
        """
        Args:
            neo4j: Connected Neo4jService instance. If None, graph write is skipped.
        """
        self._neo4j = neo4j

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["papers"], state["summaries"]
        Writes:  state["concepts"], state["errors"]
        """
        summaries = state["summaries"]
        papers = state["papers"]
        session_id = state["session_id"]

        if not summaries:
            logger.warning(
                "ConceptExtractorAgent: no summaries to extract from | session={}",
                session_id,
            )
            return {**state, "concepts": []}

        logger.info(
            "ConceptExtractorAgent starting | session={} summaries={}",
            session_id,
            len(summaries),
        )

        # ── Step 1: Extract concepts via LLM ─────────────────────────────────
        concepts = await self._extract_concepts_llm(summaries, session_id)

        # ── Step 2: Write to Neo4j (graceful skip if unavailable) ─────────────
        errors = state["errors"]
        if self._neo4j is not None:
            try:
                await self._write_graph(papers, summaries, concepts)
            except Exception as exc:
                error_msg = f"ConceptExtractorAgent Neo4j write failed (non-fatal): {exc}"
                logger.warning("ConceptExtractorAgent: {}", error_msg)
                errors = errors + [error_msg]
        else:
            logger.warning(
                "ConceptExtractorAgent: Neo4j not configured — skipping graph write | session={}",
                session_id,
            )

        logger.info(
            "ConceptExtractorAgent complete | session={} concepts={}",
            session_id,
            len(concepts),
        )

        return {**state, "concepts": concepts, "errors": errors}

    async def _extract_concepts_llm(
        self, summaries: list[PaperSummary], session_id: str
    ) -> list[str]:
        """
        Call the LLM to extract technical concepts from all summaries at once.
        Returns a cleaned, deduplicated list of concept strings.
        """
        summaries_block = _format_summaries_block(summaries)
        user_text = _USER_TEMPLATE.format(summaries_block=summaries_block)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]

        llm = LLMFactory.get_llm(temperature=0.0, max_tokens=512)

        try:
            response = await llm.ainvoke(messages)
            raw = response.content.strip()
            logger.debug(
                "ConceptExtractorAgent raw LLM response | session={} raw={}",
                session_id,
                raw[:300],
            )
        except Exception as exc:
            logger.error(
                "ConceptExtractorAgent LLM call failed | session={} error={}",
                session_id,
                exc,
            )
            return []

        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning(
                "ConceptExtractorAgent: JSON parse failed — returning empty concepts | session={}",
                session_id,
            )
            return []

        raw_concepts: list = parsed.get("concepts", [])
        concepts = list(
            dict.fromkeys(  # preserve order, deduplicate
                c.lower().strip()
                for c in raw_concepts
                if isinstance(c, str) and c.strip()
            )
        )
        return concepts[:20]  # hard cap

    async def _write_graph(
        self,
        papers: list[Paper],
        summaries: list[PaperSummary],
        concepts: list[str],
    ) -> None:
        """
        Write Paper nodes, Concept nodes, and MENTIONS edges to Neo4j.

        Paper nodes are upserted for every fetched paper.
        MENTIONS edges are created between each paper and all extracted concepts
        (since we extracted concepts from the whole batch, all papers are
        associated with all session concepts — a reasonable approximation).
        For production, per-paper concept association would require N LLM calls.
        """
        neo4j = self._neo4j  # type: ignore[assignment]

        # Build paper_id → Paper lookup for summary association
        papers_by_id = {p.paper_id: p for p in papers}

        # Upsert all Paper nodes
        for paper in papers:
            await neo4j.save_paper_node(paper)

        # Upsert all Concept nodes
        for concept in concepts:
            await neo4j.save_concept_node(concept)

        # Wire MENTIONS edges (each paper → each concept)
        summary_ids = {s.paper_id for s in summaries}
        for paper_id in summary_ids:
            for concept in concepts:
                await neo4j.link_paper_to_concept_mentions(paper_id, concept)

        logger.info(
            "ConceptExtractorAgent: graph written | papers={} concepts={} edges={}",
            len(papers),
            len(concepts),
            len(summary_ids) * len(concepts),
        )
