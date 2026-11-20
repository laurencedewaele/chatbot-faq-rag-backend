import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path

from dotenv import load_dotenv

from src.adapters.outbound.embeddings.huggingface_embedding_adapter import HuggingFaceEmbeddingAdapter
from src.adapters.outbound.evaluation.noop_evaluation_adapter import NoOpEvaluationAdapter
from src.adapters.outbound.evaluation.ragas_evaluation_adapter import RagasEvaluationAdapter
from src.adapters.outbound.faq.chroma_faq_search_adapter import ChromaFaqSearchAdapter
from src.adapters.outbound.faq.json_faq_catalog_adapter import JsonFaqCatalogAdapter
from src.adapters.outbound.llm.gemini_judge_adapter import GeminiJudgeAdapter
from src.adapters.outbound.llm.gemini_llm_adapter import GeminiLlmAdapter
from src.adapters.outbound.llm.rag_prompt_builder_adapter import RagPromptBuilderAdapter
from src.adapters.outbound.observability.langfuse_observability_adapter import LangfuseObservabilityAdapter
from src.adapters.outbound.observability.logging_observability_adapter import LoggingObservabilityAdapter
from src.adapters.outbound.reranker.cross_encoder_reranker_adapter import CrossEncoderRerankerAdapter
from src.adapters.outbound.vectorstore.chroma_retrieval_adapter import NativeRetrievalAdapter
from src.application.services.faq_lookup import FaqLookupService
from src.application.use_cases.ask_question import AskQuestionService
from src.application.use_cases.list_faq import ListFaqService
from src.application.use_cases.submit_feedback import SubmitFeedbackService
from src.config.default_answer import DEFAULT_ANSWER
from src.config.settings import FaqSettings


class Container:
    def __init__(self) -> None:
        load_dotenv()
        log_file = os.getenv("LOG_FILE", "./logs/app.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 1. Création du handler quotidien
        handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=15,
            encoding="utf-8"
        )

        # 2. Configuration globale via basicConfig
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            handlers=[handler]
        )

        faq_settings = FaqSettings.from_env()

        # RAG adapters
        embeddings = HuggingFaceEmbeddingAdapter()
        retrieval = NativeRetrievalAdapter(embedding_port=embeddings)
        prompt_builder = RagPromptBuilderAdapter()
        llm = GeminiLlmAdapter(prompt_builder=prompt_builder)
        observability = self._build_observability_adapter()
        evaluation = self._build_evaluation_adapter(observability, embeddings)

        # FAQ adapters
        self.faq_search = ChromaFaqSearchAdapter(
            faq_json_path=faq_settings.faq_json_path,
            model_name=faq_settings.embed_model_name,
        )
        reranker = CrossEncoderRerankerAdapter(model_name=faq_settings.cross_encoder_model_name)
        judge = GeminiJudgeAdapter(
            model_name=faq_settings.judge_model_name,
            temperature=faq_settings.judge_temperature,
            thinking_budget=faq_settings.judge_thinking_budget,
            max_output_tokens=faq_settings.judge_max_output_tokens,
        )
        faq_lookup = FaqLookupService(
            faq_search_port=self.faq_search,
            reranker_port=reranker,
            judge_port=judge,
            observability_port=observability,
        )

        self.ask_question = AskQuestionService(
            retrieval_port=retrieval,
            llm_port=llm,
            observability_port=observability,
            default_answer=os.getenv("DEFAULT_ANSWER", DEFAULT_ANSWER),
            faq_lookup=faq_lookup,
            faq_defaults=faq_settings.lookup_defaults(),
            evaluation_port=evaluation,
        )
        self.submit_feedback = SubmitFeedbackService(observability_port=observability)
        self.list_faq = ListFaqService(
            faq_catalog_port=JsonFaqCatalogAdapter(faq_json_path=faq_settings.faq_json_path)
        )

    def _build_observability_adapter(self):
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseObservabilityAdapter()
        return LoggingObservabilityAdapter()

    def _build_evaluation_adapter(self, observability, embeddings):
        if os.getenv("ENABLE_RAGAS_EVAL", "true").lower() != "true":
            return NoOpEvaluationAdapter()

        try:
            ragas_llm_name = os.getenv("RAGAS_LLM_MODEL_NAME", "gpt-4o-mini")
            return RagasEvaluationAdapter(
                observability_port=observability,
                llm_name=ragas_llm_name,
                embedding_model_name=embeddings.get_model_name()
            )
        except Exception as exc:
            logging.getLogger("chatbot.backend.container").warning(
                "ragas.adapter.init_failed: %s",
                exc,
                exc_info=True,
            )
            return NoOpEvaluationAdapter()
