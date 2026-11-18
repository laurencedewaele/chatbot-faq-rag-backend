import unittest

from src.application.dto.ask_question_dto import AskQuestionCommand, FaqLookupParams
from src.application.ports.outbound.observability_port import ScoreInput
from src.application.services.faq_lookup import FaqLookupService
from src.application.use_cases.ask_question import AskQuestionService
from src.domain.errors.exceptions import EmptyQuestionError, InvalidThresholdsError
from src.domain.model.evaluation_models import AnswerEvaluation
from src.domain.model.rag_models import Document
from tests.fakes import (
    FakeEvaluationPort,
    FakeFaqSearchPort,
    FakeJudgePort,
    FakeLlmPort,
    FakeObservabilityPort,
    FakeRerankerPort,
    FakeRetrievalPort,
    make_candidate,
)

DEFAULTS = FaqLookupParams(top_k=3, similarity_threshold=0.85, accept_threshold=0.9, judge_threshold=0.6)


class Harness:
    def __init__(self, chunks=(), faq_scores=None, judge=None):
        faq_scores = faq_scores or {}
        self.faq_search = FakeFaqSearchPort([make_candidate(theme) for theme in faq_scores])
        self.judge = judge or FakeJudgePort()
        self.retrieval = FakeRetrievalPort(list(chunks))
        self.llm = FakeLlmPort()
        self.observability = FakeObservabilityPort()
        self.evaluation = FakeEvaluationPort()
        faq_lookup = FaqLookupService(
            faq_search_port=self.faq_search,
            reranker_port=FakeRerankerPort(faq_scores),
            judge_port=self.judge,
            observability_port=self.observability,
        )
        self.service = AskQuestionService(
            retrieval_port=self.retrieval,
            llm_port=self.llm,
            observability_port=self.observability,
            default_answer="fallback",
            faq_lookup=faq_lookup,
            faq_defaults=DEFAULTS,
            evaluation_port=self.evaluation,
        )


class AskQuestionServiceTests(unittest.TestCase):
    # ---------------------------------------------------------------- FAQ path

    def test_raises_on_empty_question(self):
        with self.assertRaises(EmptyQuestionError):
            Harness().service.execute(AskQuestionCommand(question="   "))

    def test_raises_on_inconsistent_thresholds(self):
        with self.assertRaises(InvalidThresholdsError):
            Harness().service.execute(
                AskQuestionCommand(question="hello", faq_accept_threshold=0.5, faq_judge_threshold=0.8)
            )

    def test_faq_answer_accepted_skips_rag(self):
        h = Harness(chunks=[Document(id="1", content="context", score=0.9)], faq_scores={"a": 0.95})

        result = h.service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.answer, "Réponse a")
        self.assertEqual(result.source, "faq")
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.faq_best_score, 0.95)
        self.assertIsNone(result.judge)
        self.assertEqual(h.retrieval.calls, [])
        self.assertEqual(h.llm.calls, [])
        self.assertEqual(h.evaluation.items, [])
        self.assertEqual(h.observability.traces[0][0], "faq-rag-pipeline")

    def test_faq_answer_validated_by_judge(self):
        h = Harness(faq_scores={"a": 0.7})

        result = h.service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.source, "faq_judged")
        self.assertEqual(result.answer, "Réponse a")
        self.assertTrue(result.judge.is_answering)
        self.assertEqual(h.llm.calls, [])

    def test_judge_rejection_falls_back_to_rag_with_faq_candidates(self):
        chunk = Document(id="1", content="context", score=0.9)
        h = Harness(chunks=[chunk], faq_scores={"a": 0.7, "b": 0.2}, judge=FakeJudgePort(is_answering=False))

        result = h.service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.source, "rag")
        self.assertEqual(result.answer, "answer-from-llm")
        self.assertFalse(result.judge.is_answering)
        self.assertEqual(len(result.chunks), 1)
        question, chunks, faq_candidates = h.llm.calls[0]
        self.assertEqual([c.theme for c in faq_candidates], ["a", "b"])
        self.assertEqual(h.evaluation.items[0][0].contexts, ["context", "Réponse a", "Réponse b"])

    def test_request_thresholds_override_defaults(self):
        h = Harness(faq_scores={"a": 0.95})

        result = h.service.execute(
            AskQuestionCommand(question="hello", faq_accept_threshold=0.99, faq_judge_threshold=0.5, faq_top_k=2)
        )

        self.assertEqual(result.source, "faq_judged")
        self.assertEqual(h.faq_search.calls[0][1], 2)
        self.assertEqual(h.observability.traces[0][1]["faq_accept_threshold"], 0.99)

    def test_answer_source_score_on_trace(self):
        h = Harness(faq_scores={"a": 0.95})

        h.service.execute(AskQuestionCommand(question="hello"))

        scores = {s.name: s.value for s in h.observability.trace_handle.scores}
        self.assertEqual(scores["answer_source"], "faq")
        self.assertIn("total_latency", scores)
        self.assertTrue(all(handle.closed for handle in h.observability.all_handles))

    # ---------------------------------------------------------------- RAG path

    def test_returns_default_answer_when_no_chunks_and_no_faq_candidate(self):
        h = Harness()

        result = h.service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.answer, "fallback")
        self.assertEqual(result.source, "rag")
        self.assertEqual(result.trace_id, "trace-123")
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(h.llm.calls, [])
        self.assertEqual(h.observability.handle("answer-generation").updates[0]["status_message"], "DEFAULT_ANSWER")
        self.assertIsInstance(h.evaluation.items[0][0], AnswerEvaluation)
        self.assertEqual(h.evaluation.items[0][0].contexts, [])
        self.assertEqual(h.evaluation.items[0][1], "trace-123")

    def test_llm_called_when_no_chunks_but_faq_candidates(self):
        h = Harness(faq_scores={"a": 0.3})

        result = h.service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.source, "rag")
        self.assertEqual(result.answer, "answer-from-llm")
        self.assertEqual(len(h.llm.calls[0][2]), 1)

    def test_uses_llm_when_chunks_exist(self):
        h = Harness(chunks=[Document(id="1", content="context", score=0.91)])

        result = h.service.execute(AskQuestionCommand(question="hello", use_mmr=True, lambda_mult=0.5))

        self.assertEqual(result.answer, "answer-from-llm")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(h.retrieval.calls[0][2], 0.5)
        self.assertEqual(h.observability.traces[0][1]["lambda_mult"], 0.5)
        self.assertEqual(h.llm.calls[0][0], "hello")
        self.assertEqual(h.observability.handle("retrieval").updates[0]["status_message"], "SUCCESS")
        generation_metadata = h.observability.handle("answer-generation").updates[0]["metadata"]
        self.assertEqual(generation_metadata["thinking_budget"], 200)
        self.assertEqual(generation_metadata["max_output_tokens"], 512)
        self.assertEqual(h.evaluation.items[0][0].contexts, ["context"])

    def test_score_payloads_are_emitted_as_score_input_objects(self):
        h = Harness(chunks=[Document(id="1", content="context", score=0.91)])

        h.service.execute(AskQuestionCommand(question="hello"))

        retrieval_scores = h.observability.handle("retrieval").scores
        self.assertIsInstance(retrieval_scores[0], ScoreInput)
        self.assertEqual(retrieval_scores[0].name, "retrieve_latency")

    def test_trace_payload_uses_none_lambda_mult_when_mmr_disabled(self):
        h = Harness()

        h.service.execute(AskQuestionCommand(question="hello", use_mmr=False, lambda_mult=0.5))

        self.assertIsNone(h.observability.traces[0][1]["lambda_mult"])

    def test_passes_trace_name_and_user_id_and_resets_session_id_per_call(self):
        h = Harness()

        h.service.execute(AskQuestionCommand(question="hello", trace_name="chatbot-search", user_id="user-42"))
        h.service.execute(AskQuestionCommand(question="hello again", trace_name="chatbot-search", user_id="user-42"))

        first = h.observability.traces[0][2:5]
        second = h.observability.traces[1][2:5]
        self.assertEqual(first[:2], ("chatbot-search", "user-42"))
        self.assertEqual(second[:2], ("chatbot-search", "user-42"))
        self.assertIsNotNone(first[2])
        self.assertNotEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
