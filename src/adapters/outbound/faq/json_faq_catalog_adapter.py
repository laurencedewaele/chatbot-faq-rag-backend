from pathlib import Path
from typing import List

from src.adapters.outbound.faq.faq_json_loader import load_faq_intents
from src.application.ports.outbound.faq_catalog_port import FaqCatalogPort
from src.domain.model.faq_models import FaqIntent


class JsonFaqCatalogAdapter(FaqCatalogPort):
    """FAQ catalog read once from the intents JSON file (the same file as the FAQ index)."""

    def __init__(self, faq_json_path: str | Path) -> None:
        self._intents = load_faq_intents(faq_json_path)

    def list_intents(self) -> List[FaqIntent]:
        return list(self._intents)
