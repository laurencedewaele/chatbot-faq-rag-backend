import unittest

from src.application.use_cases.list_faq import ListFaqService
from src.domain.model.faq_models import FaqIntent
from tests.fakes import FakeFaqCatalogPort


class ListFaqServiceTests(unittest.TestCase):
    def test_returns_intents_in_catalog_order(self):
        catalog = FakeFaqCatalogPort([
            FaqIntent(intent_key="k1", response="R1", utterances=("U1", "U2"),
                      category="Cat", canonical_question="Canon ?"),
            FaqIntent(intent_key="k2", response="R2", utterances=("U3",)),
        ])

        result = ListFaqService(faq_catalog_port=catalog).execute()

        self.assertEqual([i.intent_key for i in result.intents], ["k1", "k2"])
        first = result.intents[0]
        self.assertEqual((first.category, first.canonical_question, first.utterances, first.response),
                         ("Cat", "Canon ?", ["U1", "U2"], "R1"))
        self.assertIsNone(result.intents[1].category)


if __name__ == "__main__":
    unittest.main()
