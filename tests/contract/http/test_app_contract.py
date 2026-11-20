import unittest

from fastapi.testclient import TestClient

from src.adapters.inbound.api.app import create_app
from src.application.dto.ask_question_dto import AskQuestionResult, FaqCandidateView, JudgeView
from src.application.dto.faq_catalog_dto import FaqIntentView, ListFaqResult
from src.domain.errors.exceptions import InvalidThresholdsError


class FakeAskQuestionService:
    def __init__(self):
        self.error = None

    def execute(self, command):
        self.last_command = command
        if self.error is not None:
            raise self.error
        return AskQuestionResult(
            answer="answer",
            chunks=[],
            trace_id="trace-123",
            source="faq_judged",
            faq_best_score=0.75,
            faq_candidates=[
                FaqCandidateView(formulation="Faq ?", theme="t", response="answer",
                                 similarity_score=0.9, cross_encoder_score=0.75)
            ],
            judge=JudgeView(is_answering=True, reason="ok"),
        )


class FakeSubmitFeedbackResult:
    def __init__(self):
        self.status = "ok"


class FakeSubmitFeedbackService:
    def execute(self, command):
        self.last_command = command
        return FakeSubmitFeedbackResult()


class FakeListFaqService:
    def __init__(self):
        self.error = None

    def execute(self):
        if self.error is not None:
            raise self.error
        return ListFaqResult(intents=[
            FaqIntentView(intent_key="k1", category="Cat", canonical_question="Canon ?",
                          utterances=["U1", "U2"], response="R1"),
            FaqIntentView(intent_key="k2", category=None, canonical_question=None,
                          utterances=["U3"], response="R2"),
        ])


class FakeFaqSearch:
    def count(self):
        return 42


class FakeContainer:
    def __init__(self):
        self.ask_question = FakeAskQuestionService()
        self.submit_feedback = FakeSubmitFeedbackService()
        self.faq_search = FakeFaqSearch()
        self.list_faq = FakeListFaqService()


class AppContractTests(unittest.TestCase):
    def setUp(self):
        self.container = FakeContainer()
        self.client = TestClient(create_app(container=self.container))

    def test_root_endpoint_returns_welcome_message(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Bienvenue sur l'API FAQ + RAG.", "docs": "/docs", "health": "/health"},
        )

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "faq_count": 42})

    def test_ask_endpoint_returns_answer_and_faq_details(self):
        response = self.client.post(
            "/api/v1/ask",
            json={
                "question": "Hello?",
                "top_k": 3,
                "use_mmr": False,
                "lambda_mult": None,
                "similarity_threshold": 0.4,
                "faq_accept_threshold": 0.95,
                "faq_judge_threshold": 0.5,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "answer")
        self.assertEqual(body["source"], "faq_judged")
        self.assertEqual(body["faq_best_score"], 0.75)
        self.assertEqual(body["faq_candidates"][0]["theme"], "t")
        self.assertEqual(body["judge"], {"is_answering": True, "reason": "ok"})
        self.assertIn("x-request-id", response.headers)
        command = self.container.ask_question.last_command
        self.assertEqual(command.question, "Hello?")
        self.assertEqual(command.faq_accept_threshold, 0.95)
        self.assertEqual(command.faq_judge_threshold, 0.5)
        self.assertIsNone(command.faq_top_k)

    def test_ask_endpoint_returns_400_on_inconsistent_thresholds(self):
        self.container.ask_question.error = InvalidThresholdsError("bad thresholds")

        response = self.client.post("/api/v1/ask", json={"question": "Hello?"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "bad thresholds"})

    def test_ask_endpoint_rejects_out_of_range_threshold(self):
        response = self.client.post("/api/v1/ask", json={"question": "Hello?", "faq_accept_threshold": 1.5})

        self.assertEqual(response.status_code, 422)

    def test_faq_endpoint_lists_intents(self):
        response = self.client.get("/api/v1/faq")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "count": 2,
            "intents": [
                {"intent_key": "k1", "category": "Cat", "canonical_question": "Canon ?",
                 "utterances": ["U1", "U2"], "response": "R1"},
                {"intent_key": "k2", "category": None, "canonical_question": None,
                 "utterances": ["U3"], "response": "R2"},
            ],
        })

    def test_faq_endpoint_returns_500_on_failure(self):
        self.container.list_faq.error = RuntimeError("boom")

        response = self.client.get("/api/v1/faq")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "boom"})

    def test_feedback_endpoint_returns_ok(self):
        response = self.client.post(
            "/api/v1/feedback",
            json={"trace_id": "trace-123", "score_value": "OK", "feedback_type": "helpful", "comment": "nice"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.container.submit_feedback.last_command.trace_id, "trace-123")


if __name__ == "__main__":
    unittest.main()
