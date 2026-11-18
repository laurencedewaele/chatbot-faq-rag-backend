import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.application.dto.ask_question_dto import FaqLookupParams
from src.application.ports.outbound.answer_judge_port import AnswerJudgePort
from src.application.ports.outbound.faq_search_port import FaqSearchPort
from src.application.ports.outbound.observability_port import ObservabilityPort, ScoreInput
from src.application.ports.outbound.reranker_port import RerankerPort
from src.domain.model.faq_models import (
    AnswerSource,
    FaqCandidate,
    FaqRoute,
    JudgeVerdict,
    decide_faq_route,
)

logger = logging.getLogger("chatbot.backend.faq_lookup")


@dataclass(frozen=True)
class FaqLookupOutcome:
    route: FaqRoute
    source: Optional[AnswerSource]          # set when the FAQ answer is returned to the user
    best: Optional[FaqCandidate]
    candidates: List[FaqCandidate] = field(default_factory=list)  # reranked, best first
    verdict: Optional[JudgeVerdict] = None

    @property
    def is_answered(self) -> bool:
        return self.source is not None


class FaqLookupService:
    """FAQ step of the pipeline: dense search -> cross-encoder rerank -> threshold
    routing -> (optional) LLM judge on the best candidate."""

    def __init__(
        self,
        faq_search_port: FaqSearchPort,
        reranker_port: RerankerPort,
        judge_port: AnswerJudgePort,
        observability_port: ObservabilityPort,
    ) -> None:
        self._search = faq_search_port
        self._reranker = reranker_port
        self._judge = judge_port
        self._obs = observability_port

    def lookup(self, question: str, params: FaqLookupParams) -> FaqLookupOutcome:
        candidates = self._retrieve(question, params)
        candidates = self._rerank(question, candidates)

        best = candidates[0] if candidates else None
        best_score = best.cross_encoder_score if best is not None else None
        route = decide_faq_route(best_score, params.accept_threshold, params.judge_threshold)
        logger.info("faq.route=%s best_score=%s", route.value, best_score)

        if route is FaqRoute.ACCEPT:
            return FaqLookupOutcome(route=route, source=AnswerSource.FAQ, best=best, candidates=candidates)

        if route is FaqRoute.JUDGE:
            verdict = self._run_judge(question, best)
            source = AnswerSource.FAQ_JUDGED if verdict is not None and verdict.is_answering else None
            return FaqLookupOutcome(route=route, source=source, best=best, candidates=candidates, verdict=verdict)

        return FaqLookupOutcome(route=route, source=None, best=best, candidates=candidates)

    # ------------------------------------------------------------------ steps

    def _retrieve(self, question: str, params: FaqLookupParams) -> List[FaqCandidate]:
        handle = self._obs.start_observation(name="faq-retrieval", as_type="retriever")
        try:
            start = time.perf_counter()
            candidates = self._search.search(
                question=question,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
            )
            latency = time.perf_counter() - start
            if handle is not None:
                handle.update(
                    input={"query": question},
                    output={"candidates": [c.formulation for c in candidates]},
                    status_message="SUCCESS",
                    metadata={
                        "top_k": str(params.top_k),
                        "similarity_threshold": str(params.similarity_threshold),
                        "embedding_model": self._search.get_model_name(),
                    },
                )
                handle.score({"faq_retrieve_latency": ScoreInput(name="faq_retrieve_latency", value=round(latency, 3))})
            return candidates
        except Exception as exc:
            if handle is not None:
                handle.update(input={"query": question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if handle is not None:
                handle.close()

    def _rerank(self, question: str, candidates: List[FaqCandidate]) -> List[FaqCandidate]:
        if not candidates:
            return []
        handle = self._obs.start_observation(name="faq-rerank", as_type="span")
        try:
            start = time.perf_counter()
            reranked = self._reranker.rerank(question, candidates)
            latency = time.perf_counter() - start
            if handle is not None:
                handle.update(
                    input={"query": question},
                    output={
                        "ranking": [
                            {"formulation": c.formulation, "theme": c.theme, "score": c.cross_encoder_score}
                            for c in reranked
                        ]
                    },
                    status_message="SUCCESS",
                    metadata={"reranker_model": self._reranker.get_model_name()},
                )
                handle.score({
                    "faq_rerank_latency": ScoreInput(name="faq_rerank_latency", value=round(latency, 3)),
                    "faq_best_ce_score": ScoreInput(
                        name="faq_best_ce_score",
                        value=float(reranked[0].cross_encoder_score or 0.0),
                        data_type="NUMERIC",
                    ),
                })
            return reranked
        except Exception as exc:
            if handle is not None:
                handle.update(input={"query": question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if handle is not None:
                handle.close()

    def _run_judge(self, question: str, best: FaqCandidate) -> Optional[JudgeVerdict]:
        """Ask the LLM judge. Any failure means 'not validated' -> RAG fallback."""
        handle = self._obs.start_observation(name="faq-judge", as_type="generation", model=self._judge.get_model_name())
        try:
            start = time.perf_counter()
            verdict = self._judge.judge(question=question, faq_question=best.formulation, faq_answer=best.response)
            latency = time.perf_counter() - start
            if handle is not None:
                handle.update(
                    input={"question": question, "faq_question": best.formulation, "faq_answer": best.response},
                    output={"is_answering": verdict.is_answering, "reason": verdict.reason},
                    status_message="SUCCESS",
                )
                handle.score({
                    "judge_latency": ScoreInput(name="judge_latency", value=round(latency, 3)),
                    "judge_verdict": ScoreInput(name="judge_verdict", value=1 if verdict.is_answering else 0, data_type="NUMERIC"),
                })
            return verdict
        except Exception as exc:
            logger.warning("faq.judge.failed, falling back to RAG: %s", exc, exc_info=True)
            if handle is not None:
                handle.update(input={"question": question}, output=str(exc), status_message="ERROR")
            return None
        finally:
            if handle is not None:
                handle.close()
