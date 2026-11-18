import unittest
from types import SimpleNamespace

from src.adapters.outbound.llm.gemini_judge_adapter import GeminiJudgeAdapter, JudgeOutput, JudgeOutputError


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return self.response


def build(response, **kwargs):
    models = FakeModels(response)
    adapter = GeminiJudgeAdapter(client=SimpleNamespace(models=models), **kwargs)
    return adapter, models


class GeminiJudgeAdapterTests(unittest.TestCase):
    def test_uses_parsed_structured_output(self):
        adapter, models = build(SimpleNamespace(parsed=JudgeOutput(is_answering=True, reason="ok"), text=None),
                                model_name="judge-model", temperature=0.0)

        verdict = adapter.judge("Q utilisateur", "Q faq", "R faq")

        self.assertTrue(verdict.is_answering)
        self.assertEqual(verdict.reason, "ok")
        model, contents, config = models.calls[0]
        self.assertEqual(model, "judge-model")
        self.assertIn("Q utilisateur", contents)
        self.assertIn("R faq", contents)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.temperature, 0.0)

    def test_falls_back_to_json_text(self):
        adapter, _ = build(SimpleNamespace(parsed=None, text='{"is_answering": false, "reason": "hors sujet"}'))

        verdict = adapter.judge("q", "fq", "fa")

        self.assertFalse(verdict.is_answering)
        self.assertEqual(verdict.reason, "hors sujet")

    def test_invalid_json_raises(self):
        adapter, _ = build(SimpleNamespace(parsed=None, text="peut-être"))
        with self.assertRaises(JudgeOutputError):
            adapter.judge("q", "fq", "fa")

    def test_empty_response_raises(self):
        adapter, _ = build(SimpleNamespace(parsed=None, text=None))
        with self.assertRaises(JudgeOutputError):
            adapter.judge("q", "fq", "fa")


if __name__ == "__main__":
    unittest.main()
