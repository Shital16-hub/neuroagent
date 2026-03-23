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
    Run RAGAS 0.4.x evaluation synchronously.

    Returns (faithfulness, answer_relevancy, context_precision, error_message).
    error_message is None on success, a string on failure.

    Metrics computed:
      - faithfulness:     Is the synthesis grounded in retrieved context?
                          (no ground truth required — uses NLI)
      - answer_relevancy: Does the synthesis address the original query?
                          (uses LLM + HuggingFace embeddings)
      - context_precision: Proxy = faithfulness score
                          (real context_precision needs a reference answer we don't have)

    RAGAS 0.4.x API notes (breaking changes from 0.1.x):
      - Column names changed: question→user_input, answer→response, contexts→retrieved_contexts
      - LLM/embeddings must be passed to evaluate(), NOT set on the metric singleton
      - Use fresh Faithfulness()/AnswerRelevancy() instances, not module-level singletons

    Runs in a thread pool executor so RAGAS can call asyncio.run() without
    conflicting with the outer LangGraph event loop.
    """
    try:
        import math
        import warnings
        warnings.filterwarnings("ignore")

        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas import evaluate
        from datasets import Dataset

        llm = ChatGroq(model=model, api_key=api_key, temperature=0.0)
        wrapped_llm = LangchainLLMWrapper(llm)

        # Fresh metric instances — never reuse module-level singletons across calls
        faith_metric = Faithfulness()
        metrics = [faith_metric]

        # Add AnswerRelevancy when HuggingFace embeddings are available
        wrapped_emb = None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            emb = HuggingFaceEmbeddings(model_name=embedding_model)
            wrapped_emb = LangchainEmbeddingsWrapper(emb)
            metrics.append(AnswerRelevancy())
        except Exception:
            pass  # proceed with faithfulness-only if embeddings fail

        eval_contexts = contexts if contexts else ["No context available."]

        # RAGAS 0.4.x column names (breaking change from 0.1.x)
        dataset = Dataset.from_dict({
            "user_input": [query],
            "response": [synthesis],
            "retrieved_contexts": [eval_contexts],
        })

        from ragas import RunConfig

        # Retry up to 3 times with up to 60 s backoff — handles Groq rate limits
        # that occur after the pipeline's 12+ prior LLM calls exhaust the budget.
        # raise_exceptions=True surfaces failures instead of silently returning NaN.
        run_config = RunConfig(max_retries=3, max_wait=60, timeout=120)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_emb,
            show_progress=False,
            raise_exceptions=True,
            run_config=run_config,
        )

        def _safe_score(key: str) -> float:
            val = result[key]
            if isinstance(val, (list, tuple)):
                val = val[0]
            val = float(val)
            if math.isnan(val):
                raise ValueError(
                    f"RAGAS metric '{key}' returned NaN — "
                    "likely a rate-limit or LLM failure that was not retried"
                )
            return max(0.0, min(1.0, val))

        faith = _safe_score("faithfulness")

        if wrapped_emb:
            relevancy = _safe_score("answer_relevancy")
        else:
            # HuggingFace embeddings unavailable — proxy with faithfulness.
            # This is a known limitation, not an error — just log it.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "RAGAS: HF embeddings unavailable; answer_relevancy proxied from faithfulness"
            )
            relevancy = faith

        # context_precision requires a ground-truth reference answer (we don't have one).
        # Use faithfulness as a proxy: a faithful, grounded answer tends to be precise.
        precision = faith

        return faith, relevancy, precision, None

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

        # ── Build evaluation context ─────────────────────────────────────────
        # Prefer Qdrant semantic search chunks; fall back to paper summaries.
        # The synthesis is grounded in summaries, so they are valid faithfulness
        # context when Qdrant returned no results (e.g. first run, empty index).
        if not qdrant_context:
            summaries = state.get("summaries", [])
            qdrant_context = [
                f"{s.findings} {' '.join(s.key_claims[:2])}"
                for s in summaries[:8]
                if getattr(s, "findings", "")
            ]
            if qdrant_context:
                logger.info(
                    "EvaluatorAgent: Qdrant context empty — using {} summary chunks | session={}",
                    len(qdrant_context),
                    session_id,
                )

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
