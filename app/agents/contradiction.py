"""
Contradiction Detector Agent — the star feature of NeuroAgent.

Uses a reasoning LLM (deepseek-r1 or GPT-4o) to identify conflicts
across a set of paper summaries in a single structured LLM call.

Conflict types:
  "direct"         — papers make contradictory factual claims
  "methodological" — different methods yield incompatible conclusions
  "scope"          — claims only appear to conflict because they apply
                     to different conditions / datasets / scales

ReAct-inspired prompt structure forces the model to:
  1. THINK: explicitly state what each paper claims
  2. COMPARE: look for claims that are logically incompatible
  3. CONCLUDE: assign conflict type and confidence (0.0–1.0)

All summaries are sent in a single LLM call for efficiency.
Pairwise LLM calls (N*(N-1)/2) would hit rate limits for N > 5.
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.models.state import AgentState
from app.models.summary import Conflict, ContradictionReport, PaperSummary
from app.services.llm_factory import LLMFactory

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scientific fact-checker specializing in detecting contradictions \
between research papers. You reason step-by-step before concluding.

Return only valid JSON — no markdown fences, no explanation, no extra text."""

_USER_TEMPLATE = """\
I have {n} research paper summaries. Identify ALL meaningful conflicts \
between their claims.

Follow this reasoning process for every potential conflict:

STEP 1 — THINK: State what Paper A claims and what Paper B claims.
STEP 2 — COMPARE: Are these claims logically incompatible? Consider:
  - Are they talking about the same task, dataset, or condition?
  - Could both claims be true simultaneously?
  - Is the disagreement about facts, methods, or scope?
STEP 3 — CONCLUDE: Assign conflict_type and confidence.

Conflict types:
  "direct"         — directly contradictory factual claims (e.g., "X improves Y" vs "X does not improve Y")
  "methodological" — different methods lead to incompatible conclusions about the same phenomenon
  "scope"          — claims only appear to conflict because they study different settings (lower confidence)

Confidence scale:
  0.9–1.0 = definitive contradiction, same claim domain
  0.7–0.8 = strong conflict, minor scope difference
  0.5–0.6 = moderate conflict, some ambiguity
  0.3–0.4 = weak or possible conflict
  < 0.3   = do NOT include (too speculative)

--- PAPERS ---
{papers_block}
--- END PAPERS ---

Return this exact JSON structure:
{{
  "conflicts": [
    {{
      "paper_a_id": "exact paper_id from above",
      "paper_b_id": "exact paper_id from above",
      "description": "One clear sentence explaining the conflict.",
      "confidence": 0.85,
      "conflict_type": "direct",
      "claim_a": "The exact claim from paper A (quote or close paraphrase).",
      "claim_b": "The exact claim from paper B (quote or close paraphrase)."
    }}
  ]
}}

Rules:
- Only include conflicts with confidence >= 0.3
- If NO conflicts exist, return {{"conflicts": []}}
- paper_a_id and paper_b_id must be exact IDs from the papers above
- Return ONLY valid JSON. No markdown, no explanation."""


def _format_papers_block(summaries: list[PaperSummary]) -> str:
    """Format summaries into a compact, clearly delimited block for the LLM."""
    lines = []
    for i, s in enumerate(summaries, 1):
        lines.append(f"[Paper {i}] ID: {s.paper_id}")
        lines.append(f"  Key claims: {' | '.join(s.key_claims)}")
        lines.append(f"  Methodology: {s.methodology}")
        lines.append(f"  Findings: {s.findings}")
        lines.append(f"  Limitations: {s.limitations}")
        lines.append("")
    return "\n".join(lines)


def _fix_trailing_commas(s: str) -> str:
    """Remove trailing commas before ] or } — a common LLM JSON defect."""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _extract_json(raw: str) -> Optional[dict]:
    """
    Extract JSON from LLM response, handling:
      1. DeepSeek-style <think>...</think> wrappers
      2. Markdown code fences (```json ... ```)
      3. Prose surrounding the JSON object
      4. Trailing commas before ] or } (common LLM defect)
    """
    raw = raw.strip()

    # DeepSeek-R1 wraps reasoning in <think>...</think> — strip it
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fence  ```json { ... } ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_trailing_commas(candidate))
            except json.JSONDecodeError:
                pass

    # 3. Bare JSON object — greedy match from first { to last }
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 4. Fix trailing commas and retry
            try:
                return json.loads(_fix_trailing_commas(candidate))
            except json.JSONDecodeError:
                pass

    return None


def _parse_conflicts(
    data: dict,
    valid_ids: set[str],
) -> list[Conflict]:
    """
    Parse and validate the conflicts list from LLM output.

    Filters out:
    - Conflicts referencing unknown paper IDs
    - Conflicts below confidence 0.3
    - Malformed conflict objects
    """
    raw_conflicts = data.get("conflicts", [])
    conflicts: list[Conflict] = []

    for item in raw_conflicts:
        try:
            a_id = item.get("paper_a_id", "")
            b_id = item.get("paper_b_id", "")

            if a_id not in valid_ids:
                logger.debug(
                    "ContradictionAgent: unknown paper_a_id='{}' — skipping conflict", a_id
                )
                continue
            if b_id not in valid_ids:
                logger.debug(
                    "ContradictionAgent: unknown paper_b_id='{}' — skipping conflict", b_id
                )
                continue
            if a_id == b_id:
                logger.debug("ContradictionAgent: paper_a_id == paper_b_id — skipping")
                continue

            confidence = float(item.get("confidence", 0.0))
            if confidence < 0.3:
                logger.debug(
                    "ContradictionAgent: confidence={:.2f} below threshold — skipping",
                    confidence,
                )
                continue

            conflict = Conflict(
                paper_a_id=a_id,
                paper_b_id=b_id,
                description=str(item.get("description", "Conflict detected.")),
                confidence=min(max(confidence, 0.0), 1.0),
                conflict_type=item.get("conflict_type", "scope"),
                claim_a=item.get("claim_a"),
                claim_b=item.get("claim_b"),
            )
            conflicts.append(conflict)

        except Exception as exc:
            logger.warning(
                "ContradictionAgent: failed to parse conflict item | error={} item={}",
                exc,
                item,
            )

    return conflicts


class ContradictionDetectorAgent:
    """
    Specialist agent that identifies conflicts across a set of paper summaries.

    Uses a single LLM call with all summaries to avoid hitting rate limits.
    The reasoning model (deepseek-r1-distill) is used for analytical precision.

    Usage (within LangGraph):
        agent = ContradictionDetectorAgent()
        state = await agent.run(state)
    """

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["summaries"]
        Writes:  state["contradiction_report"], state["errors"]
        """
        summaries = state["summaries"]
        session_id = state["session_id"]

        # Need at least 2 papers to find contradictions
        if len(summaries) < 2:
            logger.info(
                "ContradictionAgent: fewer than 2 summaries — skipping | session={}",
                session_id,
            )
            report = ContradictionReport(conflicts=[], total_papers_compared=len(summaries))
            return {**state, "contradiction_report": report}

        logger.info(
            "ContradictionAgent starting | session={} summaries={}",
            session_id,
            len(summaries),
        )

        valid_ids = {s.paper_id for s in summaries}
        papers_block = _format_papers_block(summaries)

        user_text = _USER_TEMPLATE.format(
            n=len(summaries),
            papers_block=papers_block,
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]

        llm = LLMFactory.get_reasoning_llm(temperature=0.0, max_tokens=4096, json_mode=True)

        try:
            response = await llm.ainvoke(messages)
            raw = response.content.strip()
        except Exception as exc:
            error_msg = f"ContradictionAgent LLM call failed: {exc}"
            logger.error("ContradictionAgent: {}", error_msg)
            report = ContradictionReport(conflicts=[], total_papers_compared=len(summaries))
            return {
                **state,
                "contradiction_report": report,
                "errors": state["errors"] + [error_msg],
            }

        logger.debug(
            "ContradictionAgent raw LLM response | session={} chars={} preview={}",
            session_id,
            len(raw),
            raw[:400],
        )

        parsed = _extract_json(raw)

        if parsed is None:
            logger.warning(
                "ContradictionAgent: JSON parse failed — returning empty report | session={}",
                session_id,
            )
            report = ContradictionReport(conflicts=[], total_papers_compared=len(summaries))
            return {
                **state,
                "contradiction_report": report,
                "errors": state["errors"] + ["ContradictionAgent: LLM returned unparseable JSON"],
            }

        conflicts = _parse_conflicts(parsed, valid_ids)

        report = ContradictionReport(
            conflicts=conflicts,
            total_papers_compared=len(summaries),
        )

        logger.info(
            "ContradictionAgent complete | session={} conflicts={} high_confidence={}",
            session_id,
            len(conflicts),
            len(report.high_confidence_conflicts),
        )

        return {**state, "contradiction_report": report}
