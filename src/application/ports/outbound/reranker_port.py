from abc import ABC, abstractmethod
from typing import List

from src.domain.model.faq_models import FaqCandidate


class RerankerPort(ABC):
    @abstractmethod
    def rerank(self, question: str, candidates: List[FaqCandidate]) -> List[FaqCandidate]:
        """Score each candidate against the question and return them sorted by
        `cross_encoder_score` (in [0, 1]) in descending order."""
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None
