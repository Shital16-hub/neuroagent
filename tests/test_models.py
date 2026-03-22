"""
Tests for Pydantic data models.

Validates that all models accept valid input, reject invalid input,
and that derived properties work correctly.
"""

import pytest
from datetime import datetime
from app.models.paper import Paper
from app.models.summary import Conflict, ContradictionReport, PaperSummary
from app.models.evaluation import EvaluationResult
from app.models.state import initial_state


class TestPaper:
    def test_valid_paper(self):
        paper = Paper(
            paper_id="2307.09288",
            title="Llama 2",
            abstract="We release Llama 2, a collection of pretrained and fine-tuned LLMs.",
            authors=["Hugo Touvron"],
            year=2023,
            source="arxiv",
        )
        assert paper.paper_id == "2307.09288"
        assert paper.source == "arxiv"
        assert isinstance(paper.fetched_at, datetime)

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source must be one of"):
            Paper(
                paper_id="x",
                title="Test",
                abstract="Test abstract.",
                source="invalid_source",
            )

    def test_empty_abstract_raises(self):
        with pytest.raises(ValueError, match="abstract cannot be empty"):
            Paper(paper_id="x", title="Test", abstract="   ", source="arxiv")

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="title cannot be empty"):
            Paper(paper_id="x", title="", abstract="Some abstract.", source="arxiv")

    def test_optional_fields_default_to_none(self):
        paper = Paper(
            paper_id="x",
            title="Test Paper",
            abstract="Abstract.",
            source="pubmed",
        )
        assert paper.doi is None
        assert paper.pdf_url is None
        assert paper.year is None
        assert paper.citation_count is None


class TestPaperSummary:
    def test_valid_summary(self):
        summary = PaperSummary(
            paper_id="2307.09288",
            key_claims=["Llama 2 outperforms previous open models"],
            methodology="Supervised fine-tuning + RLHF",
            findings="Competitive with GPT-3.5 on most benchmarks",
            limitations="Limited safety eval coverage",
            summary_model="llama-3.1-8b-instant",
        )
        assert summary.paper_id == "2307.09288"
        assert len(summary.key_claims) == 1

    def test_empty_key_claims_raises(self):
        with pytest.raises(ValueError, match="at least one claim"):
            PaperSummary(
                paper_id="x",
                key_claims=[],
                methodology="m",
                findings="f",
                limitations="l",
                summary_model="model",
            )


class TestContradictionReport:
    def test_empty_report(self):
        report = ContradictionReport(total_papers_compared=5)
        assert not report.has_conflicts
        assert report.conflicts == []

    def test_report_with_conflicts(self):
        conflict = Conflict(
            paper_a_id="paper_a",
            paper_b_id="paper_b",
            description="Paper A says X, Paper B says not X",
            confidence=0.85,
            conflict_type="direct",
        )
        report = ContradictionReport(
            conflicts=[conflict],
            total_papers_compared=2,
        )
        assert report.has_conflicts
        assert len(report.high_confidence_conflicts) == 1

    def test_low_confidence_not_in_high_confidence(self):
        conflict = Conflict(
            paper_a_id="a",
            paper_b_id="b",
            description="Mild disagreement",
            confidence=0.5,
            conflict_type="scope",
        )
        report = ContradictionReport(conflicts=[conflict], total_papers_compared=2)
        assert report.has_conflicts
        assert len(report.high_confidence_conflicts) == 0


class TestEvaluationResult:
    def test_valid_evaluation(self):
        result = EvaluationResult(
            session_id="sess_123",
            query="What is RAG?",
            faithfulness=0.87,
            answer_relevancy=0.91,
            context_precision=0.83,
            model_used="llama-3.1-8b-instant",
            num_papers_used=5,
        )
        assert result.average_score == pytest.approx(0.8700, abs=0.001)
        assert result.passed_quality_threshold

    def test_low_scores_fail_threshold(self):
        result = EvaluationResult(
            session_id="sess_456",
            query="Test query",
            faithfulness=0.5,
            answer_relevancy=0.5,
            context_precision=0.5,
            model_used="llama-3.1-8b-instant",
            num_papers_used=3,
        )
        assert not result.passed_quality_threshold

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(
                session_id="sess_789",
                query="test",
                faithfulness=1.5,  # > 1.0
                answer_relevancy=0.8,
                context_precision=0.8,
                model_used="model",
                num_papers_used=1,
            )


class TestAgentState:
    def test_initial_state_factory(self):
        state = initial_state(
            query="RAG limitations",
            session_id="sess_abc",
            user_id="user_123",
            max_papers=10,
        )
        assert state["query"] == "RAG limitations"
        assert state["session_id"] == "sess_abc"
        assert state["papers"] == []
        assert state["errors"] == []
        assert state["final_synthesis"] == ""
        assert state["mem0_context"] == ""
