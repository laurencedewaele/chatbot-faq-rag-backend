"""Text normalization shared by the FAQ index, the FAQ queries and the reranker.

Taken as is from faq-backend-api/embeddings.py so that scores (and thus the
calibrated thresholds) stay comparable with the original FAQ project.
"""
import unicodedata

_TRANSLATION_TABLE = str.maketrans({
    "’": "'", "‘": "'",
    "«": '"', "»": '"',
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "?": "",
})


def normalize_text(text: str) -> str:
    text = text.translate(_TRANSLATION_TABLE)
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.capitalize()
