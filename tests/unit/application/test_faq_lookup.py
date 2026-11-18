import unittest

from src.application.dto.ask_question_dto import FaqLookupParams
from src.application.services.faq_lookup import FaqLookupService
from src.domain.model.faq_models import AnswerSource, FaqRoute
from tests.fakes import (
    FakeFaqSearchPort,
    FakeJudgePort,
    FakeObservabilityPort,
    FakeRerankerPort,
    make_candidate,
)

PARAMS = FaqLookupParams(top_k=3, similarity_threshold=0.85, accept_threshold=0.9, judge_threshold=0.6)


def build_service(scores_by_theme, judge=None):
    candidates = [make_candidate(theme) for theme in scores_by_theme]
    observability = FakeObservabilityPort()
    judge = judge or FakeJudgePort()
    service = FaqLookupService(
        faq_search_port=FakeFaqSearchPort(candidates),
        reranker_port=FakeRerankerPort(scores_by_theme),
        judge_port=judge,
        observability_port=observability,
    )
    return service, judge, observability


class FaqLookupServiceTests(unittest.TestCase):
    def test_high_score_is_accepted_without_judge(self):
        service, judge, observability = build_service({"a": 0.4, "b": 0.95})

        outcome = service.lookup("question", PARAMS)

        self.assertIs(outcome.route, FaqRoute.ACCEPT)
        self.assertIs(outcome.source, AnswerSource.FAQ)
        self.assertEqual(outcome.best.theme, "b")
        self.assertEqual([c.theme for c in outcome.candidates], ["b", "a"])
        self.assertEqual(judge.calls, [])
        self.assertNotIn("faq-judge", observability.span_names())

    def test_intermediate_score_validated_by_judge(self):
        service, judge, observability = build_service({"a": 0.7, "b": 0.3})

        outcome = service.lookup("question", PARAMS)

        self.assertIs(outcome.route, FaqRoute.JUDGE)
        self.assertIs(outcome.source, AnswerSource.FAQ_JUDGED)
        self.assertTrue(outcome.verdict.is_answering)
        self.assertEqual(judge.calls, [("question", "Question a", "Réponse a")])
        self.assertIn("faq-judge", observability.span_names())

    def test_intermediate_score_rejected_by_judge(self):
        service, _, _ = build_service({"a": 0.7}, judge=FakeJudgePort(is_answering=False))

        outcome = service.lookup("question", PARAMS)

        self.assertIs(outcome.route, FaqRoute.JUDGE)
        self.assertIsNone(outcome.source)
        self.assertFalse(outcome.is_answered)
        self.assertFalse(outcome.verdict.is_answering)
        self.assertEqual(len(outcome.candidates), 1)

    def test_judge_failure_falls_back_to_rag(self):
        service, _, observability = build_service({"a": 0.7}, judge=FakeJudgePort(error=RuntimeError("boom")))

        outcome = service.lookup("question", PARAMS)

        self.assertFalse(outcome.is_answered)
        self.assertIsNone(outcome.verdict)
        self.assertEqual(observability.handle("faq-judge").updates[-1]["status_message"], "ERROR")

    def test_low_score_falls_back_without_judge(self):
        service, judge, _ = build_service({"a": 0.5})

        outcome = service.lookup("question", PARAMS)

        self.assertIs(outcome.route, FaqRoute.FALLBACK)
        self.assertFalse(outcome.is_answered)
        self.assertEqual(judge.calls, [])
        self.assertEqual(len(outcome.candidates), 1)

    def test_no_candidate_falls_back_and_skips_rerank(self):
        service, judge, observability = build_service({})

        outcome = service.lookup("question", PARAMS)

        self.assertIs(outcome.route, FaqRoute.FALLBACK)
        self.assertIsNone(outcome.best)
        self.assertEqual(outcome.candidates, [])
        self.assertEqual(observability.span_names(), ["faq-retrieval"])

    def test_spans_are_closed(self):
        service, _, observability = build_service({"a": 0.7})

        service.lookup("question", PARAMS)

        self.assertTrue(all(h.closed for h in observability.all_handles))


if __name__ == "__main__":
    unittest.main()
