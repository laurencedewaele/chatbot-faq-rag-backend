import unittest

from src.adapters.outbound.llm.rag_prompt_builder_adapter import RagPromptBuilderAdapter
from src.domain.model.rag_models import Document
from tests.fakes import make_candidate


class RagPromptBuilderTests(unittest.TestCase):
    def test_prompt_without_faq_is_unchanged(self):
        prompt = RagPromptBuilderAdapter().build_prompt(
            question="Q ?", chunks=[Document(id="1", content="c1", score=0.5), Document(id="2", content="c2", score=0.4)]
        )
        self.assertEqual(prompt, "CONTEXTE:\nc1\n\nc2\n\nQUESTION:\nQ ?")

    def test_prompt_with_faq_candidates_puts_them_before_context(self):
        prompt = RagPromptBuilderAdapter().build_prompt(
            question="Q ?",
            chunks=[Document(id="1", content="c1", score=0.5)],
            faq_candidates=[make_candidate("a", formulation="Faq a ?", response="Rep a")],
        )
        self.assertEqual(
            prompt,
            "RÉPONSES FAQ POTENTIELLEMENT LIÉES:\nQuestion FAQ : Faq a ?\nRéponse FAQ : Rep a"
            "\n\nCONTEXTE:\nc1\n\nQUESTION:\nQ ?",
        )


if __name__ == "__main__":
    unittest.main()
