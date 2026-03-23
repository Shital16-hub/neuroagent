"""
Tests for the Contradiction Detector Agent.

Tests three layers of the agent:
  1. _extract_json   — pure JSON extraction from messy LLM output
  2. _parse_conflicts — validation and filtering of conflict objects
  3. ContradictionDetectorAgent.run — end-to-end with mocked LLM

These tests avoid hitting any live API — the LLM is mocked via
unittest.mock.AsyncMock patching LLMFactory.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.contradiction import (
    ContradictionDetectorAgent,
    _extract_json,
    _parse_conflicts,
)
from app.models.state import initial_state
from app.models.summary import PaperSummary


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_summary(
    paper_id: str,
    claims: list[str],
    findings: str = "No significant findings.",
) -> PaperSummary:
    return PaperSummary(
        paper_id=paper_id,
        key_claims=claims,
        methodology="Empirical evaluation on standard benchmarks",
        findings=findings,
        limitations="Limited to specific datasets",
        summary_model="test-model",
    )


def _state_with_summaries(summaries: list[PaperSummary]) -> dict:
    state = initial_state(query="test query", session_id="sess-contradiction-test")
    return {**state, "summaries": summaries}


# ── _extract_json ──────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json_parsed_directly(self):
        raw = json.dumps({"conflicts": []})
        result = _extract_json(raw)
        assert result == {"conflicts": []}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"conflicts": [{"paper_a_id": "a"}]}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert "conflicts" in result

    def test_json_surrounded_by_prose(self):
        raw = 'After careful analysis:\n{"conflicts": []} \nEnd of report.'
        result = _extract_json(raw)
        assert result == {"conflicts": []}

    def test_deepseek_think_tags_stripped(self):
        raw = '<think>Let me reason step by step...</think>\n{"conflicts": []}'
        result = _extract_json(raw)
        assert result == {"conflicts": []}

    def test_trailing_commas_fixed(self):
        raw = '{"conflicts": [{"paper_a_id": "a", "paper_b_id": "b",}]}'
        result = _extract_json(raw)
        assert result is not None

    def test_completely_invalid_returns_none(self):
        result = _extract_json("The LLM refused to answer.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _extract_json("")
        assert result is None


# ── _parse_conflicts ───────────────────────────────────────────────────────────

class TestParseConflicts:
    _VALID_CONFLICT = {
        "paper_a_id": "paper_a",
        "paper_b_id": "paper_b",
        "description": "Contradictory claims about model performance.",
        "confidence": 0.85,
        "conflict_type": "direct",
        "claim_a": "Fine-tuning is superior.",
        "claim_b": "Prompting is superior.",
    }

    def test_valid_conflict_parsed(self):
        data = {"conflicts": [self._VALID_CONFLICT]}
        result = _parse_conflicts(data, valid_ids={"paper_a", "paper_b"})
        assert len(result) == 1
        assert result[0].paper_a_id == "paper_a"
        assert result[0].confidence == pytest.approx(0.85, abs=0.01)

    def test_unknown_paper_id_filtered(self):
        data = {"conflicts": [self._VALID_CONFLICT]}
        result = _parse_conflicts(data, valid_ids={"paper_a"})  # paper_b missing
        assert len(result) == 0

    def test_confidence_below_threshold_filtered(self):
        low_conf = {**self._VALID_CONFLICT, "confidence": 0.2}
        data = {"conflicts": [low_conf]}
        result = _parse_conflicts(data, valid_ids={"paper_a", "paper_b"})
        assert len(result) == 0

    def test_self_conflict_filtered(self):
        self_conflict = {**self._VALID_CONFLICT, "paper_b_id": "paper_a"}
        data = {"conflicts": [self_conflict]}
        result = _parse_conflicts(data, valid_ids={"paper_a"})
        assert len(result) == 0

    def test_confidence_clamped_to_1(self):
        over_conf = {**self._VALID_CONFLICT, "confidence": 1.5}
        data = {"conflicts": [over_conf]}
        result = _parse_conflicts(data, valid_ids={"paper_a", "paper_b"})
        assert result[0].confidence <= 1.0

    def test_malformed_item_skipped_gracefully(self):
        data = {"conflicts": [{"bad_key": "junk"}, self._VALID_CONFLICT]}
        result = _parse_conflicts(data, valid_ids={"paper_a", "paper_b"})
        # Malformed item skipped, valid item kept
        assert len(result) == 1

    def test_empty_conflicts_returns_empty_list(self):
        result = _parse_conflicts({"conflicts": []}, valid_ids={"paper_a"})
        assert result == []


# ── ContradictionDetectorAgent ─────────────────────────────────────────────────

class TestContradictionDetectorAgent:
    def _mock_llm_response(self, content: str) -> MagicMock:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=content))
        return mock_llm

    @pytest.mark.asyncio
    async def test_contradicting_papers_returns_conflicts(self):
        """Two papers with opposing factual claims should produce at least one conflict."""
        summaries = [
            _make_summary(
                "paper_a",
                ["Fine-tuning always outperforms few-shot prompting on NLP benchmarks"],
                findings="Fine-tuned BERT achieves 95% vs GPT-3 prompting at 72% on GLUE",
            ),
            _make_summary(
                "paper_b",
                ["Few-shot prompting outperforms fine-tuning in low-resource settings"],
                findings="GPT-4 prompting beats fine-tuned models by 20% with fewer than 100 examples",
            ),
        ]

        llm_output = json.dumps({
            "conflicts": [{
                "paper_a_id": "paper_a",
                "paper_b_id": "paper_b",
                "description": "paper_a claims fine-tuning is superior; paper_b claims prompting wins",
                "confidence": 0.80,
                "conflict_type": "direct",
                "claim_a": "Fine-tuning always outperforms prompting",
                "claim_b": "Prompting outperforms fine-tuning",
            }]
        })

        mock_llm = self._mock_llm_response(llm_output)
        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        report = result["contradiction_report"]
        assert report is not None
        assert len(report.conflicts) == 1
        assert report.conflicts[0].conflict_type == "direct"
        assert 0.0 <= report.conflicts[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_agreeing_papers_returns_empty_conflicts(self):
        """Two papers reaching the same conclusions should produce no conflicts."""
        summaries = [
            _make_summary(
                "paper_c",
                ["Transformer models achieve state-of-the-art on NLP tasks"],
            ),
            _make_summary(
                "paper_d",
                ["Attention mechanisms improve NLP performance significantly"],
            ),
        ]

        llm_output = json.dumps({"conflicts": []})

        mock_llm = self._mock_llm_response(llm_output)
        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        report = result["contradiction_report"]
        assert report is not None
        assert len(report.conflicts) == 0
        assert not report.has_conflicts

    @pytest.mark.asyncio
    async def test_single_summary_skips_llm_call(self):
        """With only one paper there is nothing to compare — LLM must not be called."""
        summaries = [_make_summary("solo_paper", ["Single claim"])]

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        mock_llm.ainvoke.assert_not_awaited()
        assert result["contradiction_report"].conflicts == []

    @pytest.mark.asyncio
    async def test_confidence_scores_always_between_0_and_1(self):
        """Confidence values must be clamped regardless of what the LLM returns."""
        summaries = [
            _make_summary("paper_e", ["Claim A"]),
            _make_summary("paper_f", ["Contradicts claim A"]),
        ]

        # LLM returns out-of-range confidence (1.5)
        llm_output = json.dumps({
            "conflicts": [{
                "paper_a_id": "paper_e",
                "paper_b_id": "paper_f",
                "description": "Conflicting claims.",
                "confidence": 1.5,
                "conflict_type": "direct",
            }]
        })

        mock_llm = self._mock_llm_response(llm_output)
        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        for conflict in result["contradiction_report"].conflicts:
            assert 0.0 <= conflict.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_report_not_exception(self):
        """If the LLM call fails the agent must return gracefully, never raise."""
        summaries = [
            _make_summary("paper_g", ["Claim"]),
            _make_summary("paper_h", ["Counter claim"]),
        ]

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("Groq rate limit hit"))

        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        assert result["contradiction_report"].conflicts == []
        assert any("LLM call failed" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_unparseable_json_returns_empty_report(self):
        """If the LLM returns garbage the agent must recover gracefully."""
        summaries = [
            _make_summary("paper_i", ["Claim"]),
            _make_summary("paper_j", ["Counter claim"]),
        ]

        mock_llm = self._mock_llm_response("Sorry, I cannot help with that.")

        with patch("app.agents.contradiction.LLMFactory.get_reasoning_llm", return_value=mock_llm):
            agent = ContradictionDetectorAgent()
            state = _state_with_summaries(summaries)
            result = await agent.run(state)

        assert result["contradiction_report"].conflicts == []
        assert len(result["errors"]) == 1
