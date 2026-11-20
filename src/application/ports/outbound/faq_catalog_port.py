from abc import ABC, abstractmethod
from typing import List

from src.domain.model.faq_models import FaqIntent


class FaqCatalogPort(ABC):
    @abstractmethod
    def list_intents(self) -> List[FaqIntent]:
        """Return every FAQ intent (theme) in source order."""
        raise NotImplementedError
