import logging
from dataclasses import replace

import numpy as np
from sentence_transformers import CrossEncoder

from src.adapters.outbound.faq.text_normalizer import normalize_text
from src.application.ports.outbound.reranker_port import RerankerPort
from src.domain.model.faq_models import FaqCandidate

logger = logging.getLogger("chatbot.backend.reranker")

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderRerankerAdapter(RerankerPort):
    """Rerank (question, FAQ formulation) pairs with a cross-encoder.

    The raw logit is mapped to [0, 1] with a sigmoid; thresholds apply to this score.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name
        logger.info("reranker.model.loading %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: list[FaqCandidate]) -> list[FaqCandidate]:
        if not candidates:
            return []
        normalized_question = normalize_text(question)
        pairs = [(normalized_question, candidate.formulation) for candidate in candidates]
        logits = np.asarray(self._model.predict(pairs), dtype=float).reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-logits))
        scored = [
            replace(candidate, cross_encoder_logit=float(logit), cross_encoder_score=float(score))
            for candidate, logit, score in zip(candidates, logits, scores)
        ]
        return sorted(scored, key=lambda c: c.cross_encoder_score, reverse=True)

    def get_model_name(self) -> str | None:
        return self._model_name
