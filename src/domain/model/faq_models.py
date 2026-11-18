from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.domain.errors.exceptions import InvalidThresholdsError


@dataclass(frozen=True)
class FaqEntry:
    """One indexed FAQ formulation (a question variant) with its answer."""

    formulation: str
    theme: str
    response: str


@dataclass(frozen=True)
class FaqCandidate:
    """A FAQ entry retrieved for a question, with its retrieval and rerank scores."""

    formulation: str
    theme: str
    response: str
    similarity_score: float
    cross_encoder_logit: Optional[float] = None
    cross_encoder_score: Optional[float] = None


@dataclass(frozen=True)
class JudgeVerdict:
    """Decision of the LLM judge on whether a FAQ answer answers the user question."""

    is_answering: bool
    reason: Optional[str] = None


class AnswerSource(str, Enum):
    FAQ = "faq"
    FAQ_JUDGED = "faq_judged"
    RAG = "rag"


class FaqRoute(str, Enum):
    ACCEPT = "accept"
    JUDGE = "judge"
    FALLBACK = "fallback"


def validate_thresholds(accept_threshold: float, judge_threshold: float) -> None:
    if judge_threshold > accept_threshold:
        raise InvalidThresholdsError(
            f"faq_judge_threshold ({judge_threshold}) must be lower than or equal to "
            f"faq_accept_threshold ({accept_threshold})"
        )


def decide_faq_route(
    best_score: Optional[float],
    accept_threshold: float,
    judge_threshold: float,
) -> FaqRoute:
    """Routing policy on the best cross-encoder score (sigmoid, in [0, 1]).

    - no candidate                        -> FALLBACK (RAG)
    - score >= accept_threshold           -> ACCEPT   (FAQ answer returned as is)
    - judge_threshold <= score < accept   -> JUDGE    (LLM judge must confirm)
    - score < judge_threshold             -> FALLBACK (RAG)
    """
    validate_thresholds(accept_threshold, judge_threshold)
    if best_score is None:
        return FaqRoute.FALLBACK
    if best_score >= accept_threshold:
        return FaqRoute.ACCEPT
    if best_score >= judge_threshold:
        return FaqRoute.JUDGE
    return FaqRoute.FALLBACK
