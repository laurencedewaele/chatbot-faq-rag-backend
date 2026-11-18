import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.application.dto.ask_question_dto import AskQuestionCommand
from src.application.dto.feedback_dto import SubmitFeedbackCommand
from src.domain.errors.exceptions import EmptyQuestionError, InvalidThresholdsError
from src.adapters.inbound.api.feedback_schemas import SubmitFeedbackRequest, SubmitFeedbackResponse
from src.adapters.inbound.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ChunkResponse,
    FaqCandidateResponse,
    JudgeResponse,
)


def create_app(container=None) -> FastAPI:
    """Build the FastAPI app.

    When no container is given, the real one (models, FAQ index, vector store...)
    is built at startup rather than at import time, so tests can import this module cheaply.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "container", None) is None:
            from src.bootstrap.container import Container

            app.state.container = Container()
        yield

    app = FastAPI(title="FAQ + RAG Backend API", version="0.2.0", lifespan=lifespan)
    app.state.container = container
    logger = logging.getLogger("chatbot.backend.inbound")

    def get_container():
        return app.state.container

    @app.get("/")
    def welcome() -> dict[str, str]:
        return {
            "message": "Bienvenue sur l'API FAQ + RAG.",
            "docs": "/docs",
            "health": "/health",
        }

    @app.middleware("http")
    async def inbound_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "http.request.failure",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        def log_request(status_code: int) -> None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "http.request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

        response.headers["x-request-id"] = request_id

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            log_request(response.status_code)
            return response

        async def wrapped_body_iterator():
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                log_request(response.status_code)

        response.body_iterator = wrapped_body_iterator()
        return response

    @app.get("/health")
    def health(container=Depends(get_container)) -> dict:
        faq_search = getattr(container, "faq_search", None)
        payload: dict = {"status": "ok"}
        if faq_search is not None:
            payload["faq_count"] = faq_search.count()
        return payload

    @app.post("/api/v1/ask", response_model=AskQuestionResponse)
    async def ask_question(
        payload: AskQuestionRequest,
        container=Depends(get_container),
    ) -> AskQuestionResponse:
        try:
            command = AskQuestionCommand(
                question=payload.question,
                top_k=payload.top_k,
                use_mmr=payload.use_mmr,
                lambda_mult=payload.lambda_mult,
                similarity_threshold=payload.similarity_threshold,
                trace_name=payload.trace_name,
                user_id=payload.user_id,
                faq_top_k=payload.faq_top_k,
                faq_similarity_threshold=payload.faq_similarity_threshold,
                faq_accept_threshold=payload.faq_accept_threshold,
                faq_judge_threshold=payload.faq_judge_threshold,
            )
            result = await run_in_threadpool(
                container.ask_question.execute,
                command,
            )
            return AskQuestionResponse(
                answer=result.answer,
                source=result.source,
                trace_id=result.trace_id,
                chunks=[
                    ChunkResponse(
                        id=chunk.id,
                        content=chunk.content,
                        score=chunk.score,
                        title=chunk.title,
                        author=chunk.author,
                    )
                    for chunk in result.chunks
                ],
                faq_best_score=result.faq_best_score,
                faq_candidates=[
                    FaqCandidateResponse(
                        formulation=c.formulation,
                        theme=c.theme,
                        response=c.response,
                        similarity_score=c.similarity_score,
                        cross_encoder_score=c.cross_encoder_score,
                    )
                    for c in result.faq_candidates
                ],
                judge=JudgeResponse(is_answering=result.judge.is_answering, reason=result.judge.reason)
                if result.judge
                else None,
            )
        except (EmptyQuestionError, InvalidThresholdsError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logger.exception("ask_question_failure")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/api/v1/feedback", response_model=SubmitFeedbackResponse)
    async def submit_feedback(
        payload: SubmitFeedbackRequest,
        container=Depends(get_container),
    ) -> SubmitFeedbackResponse:
        try:
            command = SubmitFeedbackCommand(
                trace_id=payload.trace_id,
                score_value=payload.score_value,
                feedback_type=payload.feedback_type,
                comment=payload.comment,
            )
            result = await run_in_threadpool(
                container.submit_feedback.execute,
                command,
            )
            return SubmitFeedbackResponse(status=result.status)
        except Exception as exc:
            logger.exception("submit_feedback_failure")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


app = create_app()
