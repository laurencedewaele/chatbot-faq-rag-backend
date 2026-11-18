import os
import time
import uuid
from typing import List, Sequence

from src.application.dto.ask_question_dto import (
    AskQuestionCommand,
    AskQuestionResult,
    ChunkView,
    FaqCandidateView,
    FaqLookupParams,
    JudgeView,
)
from src.application.ports.inbound.ask_question_use_case import AskQuestionUseCase
from src.application.ports.outbound.evaluation_port import EvaluationPort
from src.application.ports.outbound.llm_port import LlmPort
from src.application.ports.outbound.observability_port import ObservabilityPort, ScoreInput
from src.application.ports.outbound.retrieval_port import RetrievalPort
from src.application.services.faq_lookup import FaqLookupOutcome, FaqLookupService
from src.domain.errors.exceptions import EmptyQuestionError
from src.domain.model.evaluation_models import AnswerEvaluation
from src.domain.model.faq_models import AnswerSource, FaqCandidate, validate_thresholds
from src.domain.model.rag_models import Document, GenerationResult


MODEL_PRICING_PER_1K = {
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "Qwen/Qwen2.5-3B-Instruct": {"input": 0.0, "output": 0.0},
}


class AskQuestionService(AskQuestionUseCase):
    """FAQ first, then RAG.

    1. FAQ search + cross-encoder rerank.
    2. best score >= accept threshold          -> FAQ answer.
    3. judge threshold <= best score < accept  -> LLM judge; if it validates -> FAQ answer.
    4. otherwise                               -> RAG, with the FAQ candidates added to the prompt.
    """

    def __init__(
        self,
        retrieval_port: RetrievalPort,
        llm_port: LlmPort,
        observability_port: ObservabilityPort,
        default_answer: str,
        faq_lookup: FaqLookupService,
        faq_defaults: FaqLookupParams = FaqLookupParams(),
        evaluation_port: EvaluationPort | None = None,
    ) -> None:
        self._retrieval = retrieval_port
        self._llm = llm_port
        self._obs = observability_port
        self._default_answer = default_answer
        self._faq_lookup = faq_lookup
        self._faq_defaults = faq_defaults
        self._evaluation = evaluation_port

    def execute(self, command: AskQuestionCommand) -> AskQuestionResult:
        if not command.question or not command.question.strip():
            raise EmptyQuestionError("Question is empty")

        faq_params = self._resolve_faq_params(command)
        validate_thresholds(faq_params.accept_threshold, faq_params.judge_threshold)

        lambda_mult = command.lambda_mult if command.use_mmr else None
        start_total = time.perf_counter()

        # A new session_id is generated for every API call: the backend is stateless.
        session_id = str(uuid.uuid4())

        trace_handle = self._obs.start_trace(
            name="faq-rag-pipeline",
            payload={
                "query": command.question,
                "top_k": command.top_k,
                "use_mmr": command.use_mmr,
                "lambda_mult": lambda_mult,
                "similarity_threshold": command.similarity_threshold,
                "faq_top_k": faq_params.top_k,
                "faq_similarity_threshold": faq_params.similarity_threshold,
                "faq_accept_threshold": faq_params.accept_threshold,
                "faq_judge_threshold": faq_params.judge_threshold,
            },
            trace_name=command.trace_name,
            user_id=command.user_id,
            session_id=session_id,
        )

        try:
            outcome = self._faq_lookup.lookup(command.question, faq_params)

            if outcome.is_answered:
                answer = outcome.best.response
                chunks: List[Document] = []
                source = outcome.source
            else:
                answer, chunks = self._run_rag(command, lambda_mult, outcome.candidates, trace_handle)
                source = AnswerSource.RAG

            if trace_handle is not None:
                trace_handle.update(
                    input={"query": command.question},
                    output={"answer": answer, "source": source.value},
                    status_message="SUCCESS",
                )
                trace_handle.score({
                    "total_latency": ScoreInput(name="total_latency", value=float(round(time.perf_counter() - start_total, 2))),
                    "answer_source": ScoreInput(name="answer_source", value=source.value, data_type="CATEGORICAL"),
                })

            return self._build_result(answer, chunks, source, outcome, trace_handle)
        except Exception as exc:
            if trace_handle is not None:
                trace_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if trace_handle is not None:
                trace_handle.close()
            self._obs.finish_trace()

    # ------------------------------------------------------------------ helpers

    def _resolve_faq_params(self, command: AskQuestionCommand) -> FaqLookupParams:
        defaults = self._faq_defaults

        def pick(value, default):
            return default if value is None else value

        return FaqLookupParams(
            top_k=pick(command.faq_top_k, defaults.top_k),
            similarity_threshold=pick(command.faq_similarity_threshold, defaults.similarity_threshold),
            accept_threshold=pick(command.faq_accept_threshold, defaults.accept_threshold),
            judge_threshold=pick(command.faq_judge_threshold, defaults.judge_threshold),
        )

    def _run_rag(
        self,
        command: AskQuestionCommand,
        lambda_mult,
        faq_candidates: Sequence[FaqCandidate],
        trace_handle,
    ) -> tuple[str, List[Document]]:
        list_chunks = self._retrieve_chunks(command, lambda_mult)

        model_name = self._llm.get_model_name()
        pricing = MODEL_PRICING_PER_1K.get(model_name or "", {"input": 0.0, "output": 0.0})
        generation_handle = self._obs.start_observation(name="answer-generation", as_type="generation", model=model_name)
        try:
            generation_latency = 0.0
            if not list_chunks and not faq_candidates:
                answer = self._default_answer
                generation_result = GenerationResult(
                    text=answer, prompt="", input_tokens=0, output_tokens=0, thoughts_tokens=0, finish_reason=None,
                )
                if generation_handle is not None:
                    generation_handle.update(
                        input={"query": command.question},
                        output={"answer": answer},
                        status_message="DEFAULT_ANSWER",
                    )
            else:
                generation_start = time.perf_counter()
                generation_result = self._llm.generate_answer(
                    question=command.question,
                    chunks=list_chunks,
                    faq_candidates=faq_candidates,
                )
                generation_latency = time.perf_counter() - generation_start
                answer = generation_result.text
                if generation_handle is not None:
                    generation_config = self._llm.get_generation_config()
                    generation_handle.update(
                        input={"prompt": generation_result.prompt},
                        output={"answer": answer},
                        status_message="SUCCESS",
                        metadata={
                            "finish_reason": generation_result.finish_reason,
                            "prompt_version_file": "prompt_rag",
                            "thinking_budget": generation_config.get("thinking_budget"),
                            "max_output_tokens": generation_config.get("max_output_tokens"),
                            "faq_candidates_in_prompt": len(faq_candidates),
                        },
                        usage_details={
                            "input_tokens": int(generation_result.input_tokens),
                            "output_tokens": int(generation_result.output_tokens),
                            "thoughts_tokens": int(generation_result.thoughts_tokens),
                        },
                    )

            output_tokens_total = generation_result.output_tokens + generation_result.thoughts_tokens
            tokens_per_sec = int(output_tokens_total / generation_latency) if generation_latency > 0 else 0
            input_cost = (generation_result.input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens_total / 1000.0) * pricing["output"]

            if generation_handle is not None:
                generation_handle.score({
                    "model_latency": ScoreInput(name="model_latency", value=round(generation_latency, 2), data_type="NUMERIC"),
                    "tokens_per_sec": ScoreInput(name="tokens_per_sec", value=int(tokens_per_sec), data_type="NUMERIC"),
                    "input_tokens": ScoreInput(name="input_tokens", value=int(generation_result.input_tokens), data_type="NUMERIC"),
                    "output_tokens": ScoreInput(name="output_tokens", value=int(output_tokens_total), data_type="NUMERIC"),
                    "total_cost": ScoreInput(name="total_cost", value=float(input_cost + output_cost), data_type="NUMERIC"),
                })

            if self._evaluation is not None:
                contexts = [chunk.content for chunk in list_chunks] + [c.response for c in faq_candidates]
                self._evaluation.enqueue(
                    AnswerEvaluation(question=command.question, answer=answer, contexts=contexts),
                    trace_id=trace_handle.trace_id if trace_handle else None,
                    observation_id=getattr(generation_handle, "observation_id", None),
                )
            return answer, list_chunks
        except Exception as exc:
            if generation_handle is not None:
                generation_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if generation_handle is not None:
                generation_handle.close()

    def _retrieve_chunks(self, command: AskQuestionCommand, lambda_mult) -> List[Document]:
        retrieval_handle = self._obs.start_observation(name="retrieval", as_type="retriever")
        try:
            retrieval_start = time.perf_counter()
            list_chunks = self._retrieval.retrieve(
                question=command.question,
                top_k=command.top_k,
                lambda_mult=lambda_mult,
                similarity_threshold=command.similarity_threshold,
            )
            retrieval_latency = time.perf_counter() - retrieval_start
            if retrieval_handle is not None:
                retrieval_handle.update(
                    input={"query": command.question},
                    output={"retrieved_chunks": [chunk.content for chunk in list_chunks]},
                    status_message="SUCCESS",
                    metadata={
                        "reranker_used": False,
                        "mmr_used": command.use_mmr,
                        "top_k": str(command.top_k),
                        "mmr_lambda": str(lambda_mult) if command.use_mmr else None,
                        "reranker_name": None,
                        "similarity_threshold": str(command.similarity_threshold) if command.similarity_threshold is not None else None,
                        "retrieval_api_url": os.getenv("API_URL"),
                    },
                )
                retrieval_handle.score({
                    "retrieve_latency": ScoreInput(name="retrieve_latency", value=round(retrieval_latency, 2)),
                    "max_len_chunk": ScoreInput(name="max_len_chunk", value=int(max((len(c.content) for c in list_chunks), default=0))),
                    "total_doc_length": ScoreInput(name="total_doc_length", value=int(sum(len(c.content) for c in list_chunks))),
                })
            return list_chunks
        except Exception as exc:
            if retrieval_handle is not None:
                retrieval_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if retrieval_handle is not None:
                retrieval_handle.close()

    @staticmethod
    def _build_result(
        answer: str,
        chunks: List[Document],
        source: AnswerSource,
        outcome: FaqLookupOutcome,
        trace_handle,
    ) -> AskQuestionResult:
        return AskQuestionResult(
            answer=answer,
            chunks=[
                ChunkView(id=c.id, content=c.content, score=c.score, title=c.title, author=c.author)
                for c in chunks
            ],
            trace_id=trace_handle.trace_id if trace_handle else None,
            source=source.value,
            faq_best_score=outcome.best.cross_encoder_score if outcome.best else None,
            faq_candidates=[
                FaqCandidateView(
                    formulation=c.formulation,
                    theme=c.theme,
                    response=c.response,
                    similarity_score=c.similarity_score,
                    cross_encoder_score=c.cross_encoder_score,
                )
                for c in outcome.candidates
            ],
            judge=JudgeView(is_answering=outcome.verdict.is_answering, reason=outcome.verdict.reason)
            if outcome.verdict
            else None,
        )
