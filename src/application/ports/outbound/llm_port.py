from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from src.domain.model.faq_models import FaqCandidate
from src.domain.model.rag_models import Document, GenerationResult


class LlmPort(ABC):
    @abstractmethod
    def generate_answer(
        self,
        question: str,
        chunks: List[Document],
        faq_candidates: Sequence[FaqCandidate] = (),
    ) -> GenerationResult:
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None

    def get_generation_config(self) -> Dict[str, Any]:
        """Return the actual generation config in use (e.g. thinking_budget, max_output_tokens).

        This is infrastructure/env-var driven configuration, exposed here so the
        application layer can report real values to observability without the
        domain model having to carry infrastructure concerns.
        """
        return {}
