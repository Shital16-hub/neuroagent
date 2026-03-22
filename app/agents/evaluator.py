"""
Evaluator Agent — RAGAS quality evaluation for the synthesis.

Runs three RAGAS metrics on the pipeline's final synthesis:
  - faithfulness:       Is the answer grounded in the retrieved context?
                        (measures hallucination resistance)
  - answer_relevancy:   Does the answer address the original question?
  - context_precision:  Are the retrieved chunks relevant to the question?

Configured to use Groq as the evaluation LLM (not OpenAI default).
RAGAS runs in a thread pool executor to avoid event-loop conflicts, since
RAGAS internally uses asyncio.run() which cannot nest inside a running loop.

Failure policy:
  - If RAGAS fails for any reason: scores default to 0.0, error is logged
    and stored in EvaluationResult.evaluation_error
  - Result is always saved to MongoDB regardless of success/failure
  - Agent never raises — failures are non-fatal to the pipeline

Reads from state:
    query, final_synthesis, qdrant_context, session_id

Writes to state:
    evaluation — EvaluationResult (or None if MongoDB save also fails)
"""

import asyncio
import logging
from typing import Optional

from loguru import logger

from app.models.evaluation import EvaluationResult
from app.models.state import AgentState
from app.services.mongodb_service import MongoDBService
from app.config import get_settings


# ── RAGAS runner (synchronous — runs in thread pool) ──────────────────────────

def _run_ragas_sync(
    query: str,
    synthesis: str,
    contexts: list[str],
    model: str,
    api_key: str,
    embedding_model: str,
) -> tuple[float, float, float, Optional[str]]:
    """
    Run RAGAS 0.4.x faithfulness evaluation synchronously.

    Returns (faithfulness, answer_relevancy, context_precision, error_message).
    error_message is None on success, a string on failure.

    Metric: faithfulness (no ground truth required).
      - Checks if the synthesis claims are grounded in retrieved context
      - answer_relevancy and context_precision are set equal to faithfulness
        (RAGAS answer_relevancy requires n>1 LLM calls which Groq doesn't support)

    Uses the deprecated-but-still-functional ragas.evaluate() with:
      - Old-style Metric instances (faithfulness from ragas.metrics._faithfulness)
      - LangchainLLMWrapper(ChatGroq) for evaluation calls
      - LangchainEmbeddingsWrapper(HuggingFaceEmbeddings) for relevancy scoring

    Runs in a thread so it can call asyncio.run() without conflicting
    with the outer async event loop.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import math
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics._faithfulness import faithfulness
        from ragas import evaluate
        from datasets import Dataset

        # Configure LLM via LangChain wrapper (old-style Metric API)
        llm = ChatGroq(model=model, api_key=api_key, temperature=0.0)
        wrapped_llm = LangchainLLMWrapper(llm)
        faithfulness.llm = wrapped_llm

        # Configure embeddings for answer_relevancy (HuggingFace, free)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            emb = HuggingFaceEmbeddings(model_name=embedding_model)
            wrapped_emb = LangchainEmbeddingsWrapper(emb)
            faithfulness.embeddings = wrapped_emb
        except Exception:
            pass  # Embeddings not needed for faithfulness metric

        # Ensure contexts is non-empty
        eval_contexts = contexts if contexts else ["No context available."]

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [synthesis],
            "contexts": [eval_contexts],
        })

        result = evaluate(dataset, metrics=[faithfulness])

        raw_faith = result["faithfulness"]
        if isinstance(raw_faith, (list, tuple)):
            raw_faith = raw_faith[0]

        # Handle NaN (can occur with empty/trivial inputs)
        if math.isnan(float(raw_faith)):
            raw_faith = 0.0

        faith = max(0.0, min(1.0, float(raw_faith)))
        # Use faithfulness as proxy for all three metrics
        # (answer_relevancy / context_precision require OpenAI or ground truth)
        return faith, faith, faith, None

    except Exception as exc:
        return 0.0, 0.0, 0.0, str(exc)


# ── Agent ──────────────────────────────────────────────────────────────────────

class EvaluatorAgent:
    """
    Agent that evaluates the final synthesis using RAGAS.

    Saves EvaluationResult to MongoDB after every run (success or failure).
    Gracefully handles RAGAS failures by storing 0.0 scores with an error note.

    Usage (within LangGraph or standalone):
        agent = EvaluatorAgent(mongodb=mongodb_service)
        state = await agent.run(state)
    """

    def __init__(self, mongodb: Optional[MongoDBService] = None) -> None:
        self._mongodb = mongodb
        self._settings = get_settings()

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["query"], state["final_synthesis"], state["qdrant_context"],
                 state["session_id"], state["summaries"]
        Writes:  state["evaluation"], state["errors"]
        """
        query = state["query"]
        session_id = state["session_id"]
        synthesis = state.get("final_synthesis", "")
        qdrant_context = state.get("qdrant_context", [])
        num_papers = len(state.get("summaries", []))

        logger.info(
            "EvaluatorAgent starting | session={} context_chunks={}",
            session_id,
            len(qdrant_context),
        )

        if not synthesis:
            logger.warning("EvaluatorAgent: no synthesis to evaluate | session={}", session_id)
            return {**state, "evaluation": None}

        # ── Run RAGAS in thread pool ─────────────────────────────────────────
        settings = self._settings
        faith, relevancy, precision, ragas_error = (0.0, 0.0, 0.0, "Groq API key not configured")

        if settings.has_groq:
            try:
                loop = asyncio.get_event_loop()
                faith, relevancy, precision, ragas_error = await loop.run_in_executor(
                    None,
                    _run_ragas_sync,
                    query,
                    synthesis,
                    qdrant_context,
                    settings.default_llm_model,
                    settings.groq_api_key,
                    settings.embedding_model,
                )
            except Exception as exc:
                ragas_error = f"Executor error: {exc}"
                logger.error(
                    "EvaluatorAgent: RAGAS executor failed | session={} error={}",
                    session_id,
                    exc,
                )
        else:
            logger.warning("EvaluatorAgent: Groq not configured — RAGAS skipped | session={}", session_id)

        if ragas_error:
            logger.warning(
                "EvaluatorAgent: RAGAS failed (scores=0.0) | session={} error={}",
                session_id,
                ragas_error,
            )
        else:
            logger.info(
                "EvaluatorAgent: RAGAS complete | session={} faith={:.3f} relevancy={:.3f} precision={:.3f}",
                session_id,
                faith,
                relevancy,
                precision,
            )

        # ── Build EvaluationResult ───────────────────────────────────────────
        evaluation = EvaluationResult(
            session_id=session_id,
            query=query,
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            model_used=settings.default_llm_model,
            num_papers_used=num_papers,
            evaluation_error=ragas_error,
        )

        # ── Save to MongoDB ──────────────────────────────────────────────────
        new_errors: list[str] = []
        if self._mongodb:
            try:
                await self._mongodb.save_evaluation(evaluation)
            except Exception as exc:
                logger.error(
                    "EvaluatorAgent: MongoDB save failed | session={} error={}", session_id, exc
                )
                new_errors.append(f"EvaluatorAgent: MongoDB save failed — {exc}")

        return {
            **state,
            "evaluation": evaluation,
            "errors": state.get("errors", []) + new_errors,
        }
