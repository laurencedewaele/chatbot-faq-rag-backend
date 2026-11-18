from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The container imports every adapter (sentence-transformers, chromadb, ...). On machines
# where their native DLLs are blocked (Windows application control), skip instead of erroring.
pytest.importorskip("sentence_transformers", reason="sentence_transformers cannot be imported here")

import src.bootstrap.container as container_module  # noqa: E402


class DummyEmbedding:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or "embed-model"

    def get_model_name(self) -> str | None:
        return self._model_name

    def get_embedding_function(self):
        return MagicMock()


def _patched(**overrides):
    defaults = {
        "HuggingFaceEmbeddingAdapter": dict(side_effect=lambda model_name=None: DummyEmbedding(model_name)),
        "GeminiLlmAdapter": dict(return_value=SimpleNamespace()),
        "NoOpEvaluationAdapter": dict(return_value=SimpleNamespace()),
        "NativeRetrievalAdapter": dict(return_value=SimpleNamespace()),
        "RagPromptBuilderAdapter": dict(return_value=SimpleNamespace()),
        "ChromaFaqSearchAdapter": dict(return_value=SimpleNamespace()),
        "CrossEncoderRerankerAdapter": dict(return_value=SimpleNamespace()),
        "GeminiJudgeAdapter": dict(return_value=SimpleNamespace()),
        "FaqLookupService": dict(return_value=SimpleNamespace()),
        "AskQuestionService": dict(return_value=SimpleNamespace()),
        "SubmitFeedbackService": dict(return_value=SimpleNamespace()),
        "LoggingObservabilityAdapter": dict(return_value=SimpleNamespace()),
        "LangfuseObservabilityAdapter": dict(return_value=SimpleNamespace()),
        "RagasEvaluationAdapter": dict(return_value=SimpleNamespace()),
    }
    defaults.update(overrides)
    return [patch.object(container_module, name, **kwargs) for name, kwargs in defaults.items()]


def _build_container(patches):
    for p in patches:
        p.start()
    try:
        return container_module.Container()
    finally:
        for p in patches:
            p.stop()


def test_container_uses_specific_ragas_llm_and_embedding(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "logs" / "app.log"))
    monkeypatch.setenv("RAGAS_LLM_MODEL_NAME", "ragas-llm")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("ENABLE_RAGAS_EVAL", "true")

    created_ragas_kwargs = {}

    def fake_ragas_adapter(**kwargs):
        created_ragas_kwargs.update(kwargs)
        return SimpleNamespace()

    _build_container(_patched(RagasEvaluationAdapter=dict(side_effect=fake_ragas_adapter)))

    assert created_ragas_kwargs["llm_name"] == "ragas-llm"
    assert created_ragas_kwargs["embedding_model_name"] == "embed-model"


def test_container_wires_faq_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "logs" / "app.log"))
    monkeypatch.setenv("ENABLE_RAGAS_EVAL", "false")
    monkeypatch.setenv("FAQ_EMBED_MODEL_NAME", "faq-embed")
    monkeypatch.setenv("CROSS_ENCODER_MODEL_NAME", "my-cross-encoder")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "judge-model")
    monkeypatch.setenv("FAQ_ACCEPT_THRESHOLD", "0.8")
    monkeypatch.setenv("FAQ_JUDGE_THRESHOLD", "0.5")

    faq_search = MagicMock()
    reranker = MagicMock()
    judge = MagicMock()
    ask = MagicMock()
    patches = _patched(
        ChromaFaqSearchAdapter=dict(new=faq_search),
        CrossEncoderRerankerAdapter=dict(new=reranker),
        GeminiJudgeAdapter=dict(new=judge),
        AskQuestionService=dict(new=ask),
    )
    _build_container(patches)

    assert faq_search.call_args.kwargs["model_name"] == "faq-embed"
    assert faq_search.call_args.kwargs["faq_json_path"].endswith("faq.json")
    assert reranker.call_args.kwargs["model_name"] == "my-cross-encoder"
    assert judge.call_args.kwargs["model_name"] == "judge-model"
    defaults = ask.call_args.kwargs["faq_defaults"]
    assert (defaults.accept_threshold, defaults.judge_threshold) == (0.8, 0.5)
