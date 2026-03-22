"""
Summarizer Agent — parallel LLM-based paper summarization.

Sends each paper's title + abstract to Groq and extracts structured JSON:
  key_claims (max 3), methodology, findings, limitations

Runs all papers in parallel with a semaphore cap of SUMMARIZER_CONCURRENCY
(default 5) to stay within Groq's free-tier rate limit (30 req/min).

Failure policy: per-paper failures are logged and skipped — the agent
returns summaries for however many papers succeeded.
"""

import asyncio
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models.paper import Paper
from app.models.state import AgentState
from app.models.summary import PaperSummary
from app.services.llm_factory import LLMFactory
from app.config import get_settings

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a research paper analyst. Extract structured information concisely. "
    "Return only valid JSON — no markdown fences, no explanation, no extra text."
)

_USER_TEMPLATE = """\
Analyze this research paper and return a JSON object.

Title: {title}
Abstract: {abstract}

Return exactly this JSON structure:
{{
  "key_claims": ["claim 1 (max 15 words)", "claim 2", "claim 3"],
  "methodology": "One sentence describing the research method or approach.",
  "findings": "One sentence summarizing the key results or contributions.",
  "limitations": "One sentence on stated or implied limitations."
}}

Rules:
- key_claims: list of 1–3 strings, each under 20 words
- methodology, findings, limitations: exactly one sentence each
- Return ONLY valid JSON. No markdown, no explanation."""


def _extract_json(raw: str) -> Optional[dict]:
    """
    Extract a JSON object from an LLM response.

    Handles:
    - Clean JSON responses
    - Responses wrapped in ```json ... ``` markdown fences
    - Responses with leading/trailing prose
    """
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _build_fallback_summary(paper: Paper, model_id: str) -> PaperSummary:
    """
    Create a minimal PaperSummary when LLM parsing fails.
    Keeps abstract as the single claim so no information is lost.
    """
    return PaperSummary(
        paper_id=paper.paper_id,
        key_claims=[paper.abstract[:200] + "..."] if len(paper.abstract) > 200 else [paper.abstract],
        methodology="Methodology not extracted (LLM parse failure).",
        findings="Findings not extracted (LLM parse failure).",
        limitations="Limitations not extracted (LLM parse failure).",
        summary_model=f"{model_id}:fallback",
    )


class SummarizerAgent:
    """
    Specialist agent that summarizes a list of Paper objects using an LLM.

    Summaries run in parallel, gated by a semaphore to respect rate limits.
    Each failed paper summary is skipped gracefully — it never aborts others.

    Usage (within LangGraph):
        agent = SummarizerAgent()
        state = await agent.run(state)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["papers"]
        Writes:  state["summaries"], state["errors"]
        """
        papers = state["papers"]
        session_id = state["session_id"]

        if not papers:
            logger.warning("SummarizerAgent: no papers to summarize | session={}", session_id)
            return {**state, "summaries": []}

        logger.info(
            "SummarizerAgent starting | session={} papers={} concurrency={}",
            session_id,
            len(papers),
            self._settings.summarizer_concurrency,
        )

        semaphore = asyncio.Semaphore(self._settings.summarizer_concurrency)
        tasks = [self._summarize_one(paper, semaphore) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        summaries = [r for r in results if r is not None]

        logger.info(
            "SummarizerAgent complete | session={} summarized={}/{}",
            session_id,
            len(summaries),
            len(papers),
        )

        failed = len(papers) - len(summaries)
        errors = state["errors"]
        if failed:
            errors = errors + [f"SummarizerAgent: {failed} paper(s) failed to summarize"]

        return {**state, "summaries": summaries, "errors": errors}

    async def _summarize_one(
        self, paper: Paper, semaphore: asyncio.Semaphore
    ) -> Optional[PaperSummary]:
        """
        Summarize a single paper under the semaphore.

        Returns None only if the LLM call itself fails (network/API error).
        JSON parse failures yield a fallback PaperSummary instead.
        """
        async with semaphore:
            llm = LLMFactory.get_llm(temperature=0.1, max_tokens=512)
            model_id = self._settings.default_llm_model

            user_text = _USER_TEMPLATE.format(
                title=paper.title,
                abstract=paper.abstract[:3000],  # cap to avoid token overflow
            )
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_text),
            ]

            raw = ""
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=1, min=5, max=30),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        response = await llm.ainvoke(messages)
                        raw = response.content.strip()
                logger.debug(
                    "SummarizerAgent raw LLM response | paper_id={} raw={}",
                    paper.paper_id,
                    raw[:300],
                )
            except Exception as exc:
                logger.error(
                    "SummarizerAgent LLM call failed | paper_id={} error={}",
                    paper.paper_id,
                    exc,
                )
                return None

            parsed = _extract_json(raw)

            if parsed is None:
                logger.warning(
                    "SummarizerAgent JSON parse failed — using fallback | paper_id={}",
                    paper.paper_id,
                )
                return _build_fallback_summary(paper, model_id)

            try:
                summary = PaperSummary(
                    paper_id=paper.paper_id,
                    key_claims=parsed.get("key_claims", [paper.abstract[:100]]),
                    methodology=parsed.get("methodology", "Not extracted."),
                    findings=parsed.get("findings", "Not extracted."),
                    limitations=parsed.get("limitations", "Not extracted."),
                    summary_model=model_id,
                )
                logger.debug(
                    "SummarizerAgent success | paper_id={} claims={}",
                    paper.paper_id,
                    len(summary.key_claims),
                )
                return summary
            except Exception as exc:
                logger.warning(
                    "SummarizerAgent model validation failed — using fallback | paper_id={} error={}",
                    paper.paper_id,
                    exc,
                )
                return _build_fallback_summary(paper, model_id)
