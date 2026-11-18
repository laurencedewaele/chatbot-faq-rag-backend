import asyncio
import logging
import sys
from queue import Queue
from threading import Thread
from typing import Any, Optional
from unittest.mock import MagicMock

# Avoid import failures in some environments where Ragas probes Vertex AI modules.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())
sys.modules.setdefault("langchain_community.llms.vertexai", MagicMock())

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src.application.ports.outbound.evaluation_port import EvaluationPort
from src.application.ports.outbound.observability_port import ObservabilityPort
from src.domain.model.evaluation_models import AnswerEvaluation, EvaluationResult

OPENAI_FALLBACK_EMBEDDING_MODEL = "text-embedding-3-small"


class PendingEvaluation:
    def __init__(
        self,
        evaluation: AnswerEvaluation,
        trace_id: Optional[str],
        observation_id: Optional[str] = None,
    ) -> None:
        self.evaluation = evaluation
        self.trace_id = trace_id
        self.observation_id = observation_id


class RagasEvaluationAdapter(EvaluationPort):
    def __init__(
        self,
        observability_port: ObservabilityPort,
        llm_name: str,
        embedding_model_name: str,
    ) -> None:
        self._logger = logging.getLogger("chatbot.backend.ragas")
        self._obs = observability_port
        self._queue: Queue[PendingEvaluation] = Queue()

        cache = DiskCacheBackend()
        # OpenAI models require passing an AsyncOpenAI client to ragas llm_factory.
        if llm_name.startswith(("gpt", "o1", "o3", "o4")):
            self._evaluator_llm = llm_factory(llm_name, cache=cache, client=AsyncOpenAI())
        else:
            self._evaluator_llm = llm_factory(llm_name, cache=cache)

        self._faithfulness = Faithfulness(llm=self._evaluator_llm)
        # AnswerRelevancy (collections API) requires embeddings: skip it if neither
        # HuggingFace nor the OpenAI fallback can be built.
        ragas_embeddings = self._build_ragas_embeddings(model_name=embedding_model_name)
        self._answer_relevancy = (
            AnswerRelevancy(llm=self._evaluator_llm, embeddings=ragas_embeddings)
            if ragas_embeddings is not None
            else None
        )

        self._worker = Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _build_ragas_embeddings(self, model_name: str) -> Any | None:
        try:
            from ragas.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model=model_name)
        except Exception as exc:
            self._logger.warning(
                "ragas.evaluation.embeddings_init_failed",
                extra={"error": str(exc), "embedding_model": model_name},
            )

        # Fallback: OpenAI embeddings (requires OPENAI_API_KEY). Scores computed with this
        # fallback are not directly comparable to those computed with the HuggingFace model.
        try:
            from ragas.embeddings.base import embedding_factory

            embeddings = embedding_factory(
                "openai", model=OPENAI_FALLBACK_EMBEDDING_MODEL, client=AsyncOpenAI()
            )
            self._logger.warning(
                "ragas.evaluation.embeddings_fallback_openai",
                extra={"embedding_model": OPENAI_FALLBACK_EMBEDDING_MODEL},
            )
            return embeddings
        except Exception as exc:
            self._logger.warning(
                "ragas.evaluation.embeddings_fallback_failed",
                extra={"error": str(exc), "embedding_model": OPENAI_FALLBACK_EMBEDDING_MODEL},
            )
            return None

    def enqueue(
        self,
        evaluation: AnswerEvaluation,
        trace_id: Optional[str],
        observation_id: Optional[str] = None,
    ) -> None:
        if not evaluation.contexts:
            return

        self._queue.put(
            PendingEvaluation(evaluation=evaluation, trace_id=trace_id, observation_id=observation_id)
        )

    def _run_worker(self) -> None:
        # Dedicated event loop for this thread: the AsyncOpenAI client must stay bound to
        # a single loop across evaluations (asyncio.run() per item would close it each time).
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            item = self._queue.get()
            try:
                loop.run_until_complete(self._process_item(item))
            except Exception as exc:
                self._logger.error("ragas.evaluation.failed", extra={"error": str(exc)})
            finally:
                self._queue.task_done()

    async def _compute_raw_scores(self, evaluation: AnswerEvaluation) -> dict[str, Any]:
        coroutines = {
            self._faithfulness.name: self._faithfulness.ascore(
                user_input=evaluation.question,
                response=evaluation.answer,
                retrieved_contexts=evaluation.contexts,
            )
        }
        if self._answer_relevancy is not None:
            coroutines[self._answer_relevancy.name] = self._answer_relevancy.ascore(
                user_input=evaluation.question,
                response=evaluation.answer,
            )

        results = await asyncio.gather(*coroutines.values())
        return {name: result.value for name, result in zip(coroutines.keys(), results)}

    async def _process_item(self, pending: PendingEvaluation) -> None:
        evaluation = pending.evaluation
        raw_scores = await self._compute_raw_scores(evaluation)
        scores: dict[str, float] = {}
        for score_name, raw_value in raw_scores.items():
            if raw_value is None:
                value = 0.0
            else:
                parsed = float(raw_value)
                value = 0.0 if parsed != parsed else parsed
            scores[score_name] = value

        evaluation_result = EvaluationResult(scores=scores)

        trace_id = pending.trace_id
        if not trace_id:
            self._logger.info("ragas.evaluation.skip_scoring", extra={"reason": "missing_trace_id"})
            return

        for score_name, value in evaluation_result.scores.items():
            self._obs.create_score(
                trace_id=trace_id,
                name=score_name,
                value=value,
                observation_id=pending.observation_id,
            )

        self._logger.info(
            "ragas.evaluation.success",
            extra={"trace_id": trace_id, "scores": evaluation_result.scores},
        )
