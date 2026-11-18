from abc import ABC, abstractmethod
from typing import List

from src.domain.model.faq_models import FaqCandidate


class FaqSearchPort(ABC):
    @abstractmethod
    def search(self, question: str, top_k: int, similarity_threshold: float) -> List[FaqCandidate]:
        """Return at most `top_k` candidates (one per FAQ theme) whose similarity
        with the question is >= `similarity_threshold`, ordered by similarity."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of indexed FAQ formulations."""
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None
