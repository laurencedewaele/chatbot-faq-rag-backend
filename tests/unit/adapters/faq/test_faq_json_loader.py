import json
import tempfile
import unittest
from pathlib import Path

from src.adapters.outbound.faq.faq_json_loader import load_faq_entries, load_faq_intents
from src.adapters.outbound.faq.json_faq_catalog_adapter import JsonFaqCatalogAdapter
from src.adapters.outbound.faq.text_normalizer import normalize_text
from src.config.settings import PROJECT_ROOT


class FaqJsonLoaderTests(unittest.TestCase):
    def _write(self, payload) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink())
        return tmp.name

    def test_utterances_plus_missing_canonical_question_disabled_skipped(self):
        path = self._write({"intents": [
            {"intent_key": "k1", "enabled": True, "response": "R1",
             "canonical_question": "Canon ?", "utterances": ["U1", "U2"]},
            {"intent_key": "k2", "enabled": True, "response": "R2",
             "canonical_question": "U3", "utterances": ["U3"]},
            {"intent_key": "k3", "enabled": False, "response": "R3", "utterances": ["U4"]},
        ]})

        entries = load_faq_entries(path)

        self.assertEqual([(e.formulation, e.theme) for e in entries],
                         [("U1", "k1"), ("U2", "k1"), ("Canon ?", "k1"), ("U3", "k2")])

    def test_intents_keep_grouping_and_metadata(self):
        path = self._write({"intents": [
            {"intent_key": " k1 ", "category": "Cat A", "enabled": True, "response": " R1 ",
             "canonical_question": "Canon ?", "utterances": ["U1", "  ", "U2"]},
            {"intent_key": "k2", "enabled": True, "response": "R2", "utterances": ["U3"]},
            {"intent_key": "k3", "enabled": False, "response": "R3", "utterances": ["U4"]},
            {"intent_key": "k4", "enabled": True, "response": "", "utterances": ["U5"]},
        ]})

        intents = load_faq_intents(path)

        self.assertEqual([i.intent_key for i in intents], ["k1", "k2"])
        first = intents[0]
        self.assertEqual(first.category, "Cat A")
        self.assertEqual(first.canonical_question, "Canon ?")
        self.assertEqual(first.utterances, ("U1", "U2"))
        self.assertEqual(first.response, "R1")
        self.assertIsNone(intents[1].category)
        self.assertIsNone(intents[1].canonical_question)

    def test_catalog_adapter_lists_intents(self):
        path = self._write({"intents": [
            {"intent_key": "k1", "enabled": True, "response": "R1", "utterances": ["U1"]},
        ]})

        intents = JsonFaqCatalogAdapter(path).list_intents()

        self.assertEqual([i.intent_key for i in intents], ["k1"])

    def test_invalid_payload_raises(self):
        with self.assertRaises(ValueError):
            load_faq_entries(self._write({"no_intents": []}))

    def test_project_faq_file_loads(self):
        entries = load_faq_entries(PROJECT_ROOT / "data" / "faq.json")
        self.assertGreater(len(entries), 100)
        intents = load_faq_intents(PROJECT_ROOT / "data" / "faq.json")
        self.assertEqual(len(entries), sum(len(i.formulations()) for i in intents))


class TextNormalizerTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  C’est  quoi  «l'Asso» ?  "), 'C\'est quoi "l\'asso"')


if __name__ == "__main__":
    unittest.main()
