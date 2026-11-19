from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FaqLookupParams:
    """Effective FAQ lookup parameters (request values, or defaults from settings)."""

    top_k: int = 3
    similarity_threshold: float = 0.85
    accept_threshold: float = 0.90
    judge_threshold: float = 0.60


@dataclass(frozen=True)
class AskQuestionCommand:
    question: str
    top_k: int = 3
    use_mmr: bool = False
    lambda_mult: Optional[float] = None
    similarity_threshold: float = 0.4
    trace_name: str = "chatbot-search"
    user_id: str = "anonymous"
    # FAQ step: None means "use the default from settings"
    faq_top_k: Optional[int] = None
    faq_similarity_threshold: Optional[float] = None
    faq_accept_threshold: Optional[float] = None
    faq_judge_threshold: Optional[float] = None
    # Public URL of the API endpoint that received the request (for tracing)
    api_url: Optional[str] = None


@dataclass(frozen=True)
class ChunkView:
    id: str
    content: str
    score: Optional[float]
    title: Optional[str]
    author: Optional[str]


@dataclass(frozen=True)
class FaqCandidateView:
    formulation: str
    theme: str
    response: str
    similarity_score: float
    cross_encoder_score: Optional[float]


@dataclass(frozen=True)
class JudgeView:
    is_answering: bool
    reason: Optional[str]


@dataclass(frozen=True)
class AskQuestionResult:
    answer: str
    chunks: List[ChunkView]
    trace_id: Optional[str]
    source: str = "rag"
    faq_best_score: Optional[float] = None
    faq_candidates: List[FaqCandidateView] = field(default_factory=list)
    judge: Optional[JudgeView] = None
