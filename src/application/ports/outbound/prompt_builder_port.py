from abc import ABC, abstractmethod
from typing import List, Sequence

from src.domain.model.faq_models import FaqCandidate
from src.domain.model.rag_models import Document


class PromptBuilderPort(ABC):
    @abstractmethod
    def build_prompt(
        self,
        question: str,
        chunks: List[Document],
        faq_candidates: Sequence[FaqCandidate] = (),
    ) -> str:
        raise NotImplementedError
