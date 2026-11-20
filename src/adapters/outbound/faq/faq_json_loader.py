"""Load FAQ entries from the intents JSON file (format of faq-backend-api/data/faq.json)."""
import json
import logging
from pathlib import Path
from typing import Any

from src.domain.model.faq_models import FaqEntry, FaqIntent

logger = logging.getLogger("chatbot.backend.faq_loader")


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_faq_intents(json_path: str | Path) -> list[FaqIntent]:
    """Return the enabled and valid intents of the file, in file order.

    An intent is kept when it has an `intent_key`, a `response` and at least one
    non-empty formulation (utterance or canonical_question). Strings are stripped.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the JSON is malformed or contains no valid intent.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {json_path}")

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload: dict[str, Any] = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid FAQ JSON payload: {exc}") from exc

    raw_intents = payload.get("intents")
    if not isinstance(raw_intents, list):
        raise ValueError("The FAQ JSON payload does not contain a valid 'intents' list")

    intents: list[FaqIntent] = []
    for raw in raw_intents:
        if not isinstance(raw, dict) or not raw.get("enabled", False):
            continue

        raw_utterances = raw.get("utterances")
        intent_key = _clean_str(raw.get("intent_key"))
        response = _clean_str(raw.get("response"))
        if not isinstance(raw_utterances, list) or not raw_utterances:
            continue
        if intent_key is None or response is None:
            continue

        utterances = tuple(u for u in (_clean_str(item) for item in raw_utterances) if u is not None)
        intent = FaqIntent(
            intent_key=intent_key,
            response=response,
            utterances=utterances,
            category=_clean_str(raw.get("category")),
            canonical_question=_clean_str(raw.get("canonical_question")),
        )
        if intent.formulations():
            intents.append(intent)

    if not intents:
        raise ValueError("No valid FAQ entries were found in the JSON payload")

    return intents


def load_faq_entries(json_path: str | Path) -> list[FaqEntry]:
    """Build one FaqEntry per utterance of each enabled intent, plus the
    canonical_question when it is not already among the utterances.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the JSON is malformed or contains no valid entry.
    """
    entries = [
        FaqEntry(formulation=formulation, theme=intent.intent_key, response=intent.response)
        for intent in load_faq_intents(json_path)
        for formulation in intent.formulations()
    ]
    logger.info("faq.loaded entries=%d path=%s", len(entries), json_path)
    return entries
