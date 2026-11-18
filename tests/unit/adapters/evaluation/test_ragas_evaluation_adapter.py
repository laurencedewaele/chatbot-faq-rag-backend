import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# Import through the adapter module: it stubs the Vertex AI modules ragas probes at import time.
module = pytest.importorskip(
    "src.adapters.outbound.evaluation.ragas_evaluation_adapter",
    reason="ragas cannot be imported here",
)

from src.adapters.outbound.evaluation.ragas_evaluation_adapter import (  # noqa: E402
    OPENAI_FALLBACK_EMBEDDING_MODEL,
    PendingEvaluation,
    RagasEvaluationAdapter,
)
from src.domain.model.evaluation_models import AnswerEvaluation  # noqa: E402

LOGGER_NAME = "chatbot.backend.ragas"


class FakeMetric:
    """Mimics a ragas.metrics.collections metric: a name and an async ascore()."""

    def __init__(self, name, value=0.5, error=None, **init_kwargs):
        self.name = name
        self.value = value
        self.error = error
        self.init_kwargs = init_kwargs
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(value=self.value)


@pytest.fixture
def ragas_mocks():
    """Patches every external dependency of the adapter (LLM, embeddings, metrics)."""
    created = {}

    def make_faithfulness(**kwargs):
        created["faithfulness"] = FakeMetric("faithfulness", value=0.9, **kwargs)
        return created["faithfulness"]

    def make_answer_relevancy(**kwargs):
        created["answer_relevancy"] = FakeMetric("answer_relevancy", value=0.7, **kwargs)
        return created["answer_relevancy"]

    hf_embeddings = MagicMock(name="hf_embeddings")
    openai_embeddings = MagicMock(name="openai_embeddings")

    with (
        patch.object(module, "DiskCacheBackend") as cache_cls,
        patch.object(module, "AsyncOpenAI") as openai_cls,
        patch.object(module, "llm_factory") as llm_factory,
        patch.object(module, "Faithfulness", side_effect=make_faithfulness),
        patch.object(module, "AnswerRelevancy", side_effect=make_answer_relevancy),
        patch("ragas.embeddings.HuggingFaceEmbeddings", return_value=hf_embeddings) as hf_cls,
        patch("ragas.embeddings.base.embedding_factory", return_value=openai_embeddings) as emb_factory,
    ):
        yield SimpleNamespace(
            cache_cls=cache_cls,
            openai_cls=openai_cls,
            llm_factory=llm_factory,
            hf_cls=hf_cls,
            hf_embeddings=hf_embeddings,
            embedding_factory=emb_factory,
            openai_embeddings=openai_embeddings,
            created=created,
        )


def _make_adapter(obs=None, llm_name="gpt-4o-mini", embedding_model_name="hf-model"):
    return RagasEvaluationAdapter(
        observability_port=obs or MagicMock(),
        llm_name=llm_name,
        embedding_model_name=embedding_model_name,
    )


def _evaluation(contexts=("ctx 1", "ctx 2")):
    return AnswerEvaluation(question="Q ?", answer="A.", contexts=list(contexts))


def _messages(caplog, level=logging.WARNING):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME and r.levelno == level]


# --- LLM construction ---------------------------------------------------------------------


@pytest.mark.parametrize("llm_name", ["gpt-4o-mini", "o1-mini", "o3", "o4-mini"])
def test_openai_llm_receives_async_openai_client(ragas_mocks, llm_name):
    _make_adapter(llm_name=llm_name)

    ragas_mocks.llm_factory.assert_called_once_with(
        llm_name,
        cache=ragas_mocks.cache_cls.return_value,
        client=ragas_mocks.openai_cls.return_value,
    )


def test_non_openai_llm_is_built_without_client(ragas_mocks):
    _make_adapter(llm_name="gemini-2.0-flash")

    ragas_mocks.llm_factory.assert_called_once_with(
        "gemini-2.0-flash", cache=ragas_mocks.cache_cls.return_value
    )


def test_metrics_share_the_evaluator_llm(ragas_mocks):
    _make_adapter()

    llm = ragas_mocks.llm_factory.return_value
    assert ragas_mocks.created["faithfulness"].init_kwargs == {"llm": llm}
    assert ragas_mocks.created["answer_relevancy"].init_kwargs["llm"] is llm


# --- Embeddings: HuggingFace, then OpenAI fallback ---------------------------------------


def test_answer_relevancy_uses_huggingface_embeddings(ragas_mocks):
    _make_adapter(embedding_model_name="my-hf-model")

    ragas_mocks.hf_cls.assert_called_once_with(model="my-hf-model")
    ragas_mocks.embedding_factory.assert_not_called()
    assert ragas_mocks.created["answer_relevancy"].init_kwargs["embeddings"] is ragas_mocks.hf_embeddings


def test_falls_back_to_openai_embeddings_when_huggingface_fails(ragas_mocks, caplog):
    ragas_mocks.hf_cls.side_effect = RuntimeError("hf down")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _make_adapter()

    ragas_mocks.embedding_factory.assert_called_once_with(
        "openai",
        model=OPENAI_FALLBACK_EMBEDDING_MODEL,
        client=ragas_mocks.openai_cls.return_value,
    )
    assert (
        ragas_mocks.created["answer_relevancy"].init_kwargs["embeddings"]
        is ragas_mocks.openai_embeddings
    )
    assert _messages(caplog) == [
        "ragas.evaluation.embeddings_init_failed",
        "ragas.evaluation.embeddings_fallback_openai",
    ]


def test_answer_relevancy_is_disabled_when_both_embeddings_fail(ragas_mocks, caplog):
    ragas_mocks.hf_cls.side_effect = RuntimeError("hf down")
    ragas_mocks.embedding_factory.side_effect = RuntimeError("no OPENAI_API_KEY")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        adapter = _make_adapter()

    assert "answer_relevancy" not in ragas_mocks.created
    assert adapter._answer_relevancy is None
    assert _messages(caplog) == [
        "ragas.evaluation.embeddings_init_failed",
        "ragas.evaluation.embeddings_fallback_failed",
    ]


# --- Scoring --------------------------------------------------------------------------------


def test_process_item_scores_both_metrics_and_sends_them_to_observability(ragas_mocks):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)

    asyncio.run(adapter._process_item(PendingEvaluation(_evaluation(), "trace-1", "obs-1")))

    assert ragas_mocks.created["faithfulness"].calls == [
        {"user_input": "Q ?", "response": "A.", "retrieved_contexts": ["ctx 1", "ctx 2"]}
    ]
    assert ragas_mocks.created["answer_relevancy"].calls == [{"user_input": "Q ?", "response": "A."}]
    assert obs.create_score.call_args_list == [
        call(trace_id="trace-1", name="faithfulness", value=0.9, observation_id="obs-1"),
        call(trace_id="trace-1", name="answer_relevancy", value=0.7, observation_id="obs-1"),
    ]


def test_process_item_only_scores_faithfulness_without_embeddings(ragas_mocks):
    ragas_mocks.hf_cls.side_effect = RuntimeError("hf down")
    ragas_mocks.embedding_factory.side_effect = RuntimeError("no OPENAI_API_KEY")
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)

    asyncio.run(adapter._process_item(PendingEvaluation(_evaluation(), "trace-1")))

    assert obs.create_score.call_args_list == [
        call(trace_id="trace-1", name="faithfulness", value=0.9, observation_id=None),
    ]


@pytest.mark.parametrize("raw_value", [None, float("nan")])
def test_missing_or_nan_scores_are_reported_as_zero(ragas_mocks, raw_value):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)
    adapter._faithfulness.value = raw_value

    asyncio.run(adapter._process_item(PendingEvaluation(_evaluation(), "trace-1")))

    sent = {c.kwargs["name"]: c.kwargs["value"] for c in obs.create_score.call_args_list}
    assert sent == {"faithfulness": 0.0, "answer_relevancy": 0.7}


def test_scores_are_not_sent_without_trace_id(ragas_mocks, caplog):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        asyncio.run(adapter._process_item(PendingEvaluation(_evaluation(), trace_id=None)))

    obs.create_score.assert_not_called()
    assert "ragas.evaluation.skip_scoring" in _messages(caplog, level=logging.INFO)


# --- Queue and background worker ----------------------------------------------------------


def test_enqueue_ignores_evaluations_without_contexts(ragas_mocks):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)

    adapter.enqueue(_evaluation(contexts=()), trace_id="trace-1")
    adapter._queue.join()

    assert ragas_mocks.created["faithfulness"].calls == []
    obs.create_score.assert_not_called()


def test_worker_processes_enqueued_evaluations(ragas_mocks):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)

    adapter.enqueue(_evaluation(), trace_id="trace-1", observation_id="obs-1")
    adapter.enqueue(_evaluation(), trace_id="trace-2")
    adapter._queue.join()

    assert [c.kwargs["trace_id"] for c in obs.create_score.call_args_list] == [
        "trace-1",
        "trace-1",
        "trace-2",
        "trace-2",
    ]


def test_worker_logs_failures_and_keeps_running(ragas_mocks, caplog):
    obs = MagicMock()
    adapter = _make_adapter(obs=obs)
    adapter._faithfulness.error = RuntimeError("LLM unavailable")

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        adapter.enqueue(_evaluation(), trace_id="trace-1")
        adapter._queue.join()

    assert _messages(caplog, level=logging.ERROR) == ["ragas.evaluation.failed"]
    obs.create_score.assert_not_called()

    adapter._faithfulness.error = None
    adapter.enqueue(_evaluation(), trace_id="trace-2")
    adapter._queue.join()

    assert {c.kwargs["trace_id"] for c in obs.create_score.call_args_list} == {"trace-2"}
