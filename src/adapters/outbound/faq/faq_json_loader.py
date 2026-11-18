"""Load FAQ entries from the intents JSON file (format of faq-backend-api/data/faq.json)."""
import json
import logging
from pathlib import Path
from typing import Any

from src.domain.model.faq_models import FaqEntry

logger = logging.getLogger("chatbot.backend.faq_loader")


def load_faq_entries(json_path: str | Path) -> list[FaqEntry]:
    """Build one FaqEntry per utterance of each enabled intent, plus the
    canonical_question when it is not already among the utterances.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the JSON is malformed or contains no valid entry.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {json_path}")

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload: dict[str, Any] = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid FAQ JSON payload: {exc}") from exc

    intents = payload.get("intents")
    if not isinstance(intents, list):
        raise ValueError("The FAQ JSON payload does not contain a valid 'intents' list")

    entries: list[FaqEntry] = []
    for intent in intents:
        if not isinstance(intent, dict) or not intent.get("enabled", False):
            continue

        utterances = intent.get("utterances")
        intent_key = intent.get("intent_key")
        response = intent.get("response")
        if not isinstance(utterances, list) or not utterances:
            continue
        if not isinstance(intent_key, str) or not intent_key.strip():
            continue
        if not isinstance(response, str) or not response.strip():
            continue

        formulations = list(utterances)
        canonical_question = intent.get("canonical_question")
        if isinstance(canonical_question, str) and canonical_question.strip():
            if canonical_question.strip() not in [u.strip() for u in utterances if isinstance(u, str)]:
                formulations.append(canonical_question)

        for formulation in formulations:
            if not isinstance(formulation, str) or not formulation.strip():
                continue
            entries.append(
                FaqEntry(formulation=formulation.strip(), theme=intent_key.strip(), response=response.strip())
            )

    if not entries:
        raise ValueError("No valid FAQ entries were found in the JSON payload")

    logger.info("faq.loaded entries=%d path=%s", len(entries), path)
    return entries
