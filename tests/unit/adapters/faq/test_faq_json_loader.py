import json
import tempfile
import unittest
from pathlib import Path

from src.adapters.outbound.faq.faq_json_loader import load_faq_entries
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

    def test_invalid_payload_raises(self):
        with self.assertRaises(ValueError):
            load_faq_entries(self._write({"no_intents": []}))

    def test_project_faq_file_loads(self):
        entries = load_faq_entries(PROJECT_ROOT / "data" / "faq.json")
        self.assertGreater(len(entries), 100)


class TextNormalizerTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  C’est  quoi  «l'Asso» ?  "), 'C\'est quoi "l\'asso"')


if __name__ == "__main__":
    unittest.main()
