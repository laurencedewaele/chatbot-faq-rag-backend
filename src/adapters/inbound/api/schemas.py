from typing import Literal
from pydantic import BaseModel, Field

class AskQuestionRequest(BaseModel):
    question: str = Field(default="C'est quoi l'association ?", min_length=1)
    # RAG step
    top_k: int = Field(default=3, ge=1, le=20)
    use_mmr: bool = True
    lambda_mult: float | None = Field(default=0.55, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    trace_name: str = Field(default="chatbot-search", min_length=1)
    user_id: str = Field(default="anonymous", min_length=1)
    # FAQ step (None = server default from the environment)
    faq_top_k: int | None = Field(default=None, ge=1, le=20)
    faq_similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Seuil de similarité cosinus de la recherche FAQ."
    )
    faq_accept_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Score cross-encoder à partir duquel la réponse FAQ est renvoyée directement.",
    )
    faq_judge_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Score cross-encoder à partir duquel le LLM juge est consulté (doit être <= faq_accept_threshold).",
    )

class ChunkResponse(BaseModel):
    id: str
    content: str
    score: float | None
    title: str | None
    author: str | None

class FaqCandidateResponse(BaseModel):
    formulation: str
    theme: str
    response: str
    similarity_score: float
    cross_encoder_score: float | None

class JudgeResponse(BaseModel):
    is_answering: bool
    reason: str | None

class AskQuestionResponse(BaseModel):
    answer: str
    source: Literal["faq", "faq_judged", "rag"]
    chunks: list[ChunkResponse]
    faq_best_score: float | None = None
    faq_candidates: list[FaqCandidateResponse] = Field(default_factory=list)
    judge: JudgeResponse | None = None
    trace_id: str | None
