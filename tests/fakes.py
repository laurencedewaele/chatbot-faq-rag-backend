"""Hand-written fakes for the outbound ports (shared by the unit tests)."""
from dataclasses import replace

from src.application.ports.outbound.observability_port import ScoreInput
from src.domain.model.faq_models import FaqCandidate, JudgeVerdict
from src.domain.model.rag_models import GenerationResult


def make_candidate(theme="theme_a", score=None, formulation=None, response=None, similarity=0.9):
    return FaqCandidate(
        formulation=formulation or f"Question {theme}",
        theme=theme,
        response=response or f"Réponse {theme}",
        similarity_score=similarity,
        cross_encoder_score=score,
    )


class FakeFaqSearchPort:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = []

    def search(self, question, top_k, similarity_threshold):
        self.calls.append((question, top_k, similarity_threshold))
        return self.candidates[:top_k]

    def count(self):
        return len(self.candidates)

    def get_model_name(self):
        return "fake-embed"


class FakeRerankerPort:
    """Assigns the scores given per theme and sorts, like the real adapter."""

    def __init__(self, scores_by_theme):
        self.scores_by_theme = scores_by_theme
        self.calls = []

    def rerank(self, question, candidates):
        self.calls.append((question, candidates))
        scored = [replace(c, cross_encoder_score=self.scores_by_theme[c.theme]) for c in candidates]
        return sorted(scored, key=lambda c: c.cross_encoder_score, reverse=True)

    def get_model_name(self):
        return "fake-cross-encoder"


class FakeJudgePort:
    def __init__(self, is_answering=True, error=None):
        self.is_answering = is_answering
        self.error = error
        self.calls = []

    def judge(self, question, faq_question, faq_answer):
        self.calls.append((question, faq_question, faq_answer))
        if self.error is not None:
            raise self.error
        return JudgeVerdict(is_answering=self.is_answering, reason="fake reason")

    def get_model_name(self):
        return "fake-judge"


class FakeRetrievalPort:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, top_k, lambda_mult, similarity_threshold):
        self.calls.append((question, top_k, lambda_mult, similarity_threshold))
        return self.chunks


class FakeLlmPort:
    def __init__(self):
        self.calls = []

    def generate_answer(self, question, chunks, faq_candidates=()):
        self.calls.append((question, chunks, faq_candidates))
        return GenerationResult(
            text="answer-from-llm",
            prompt="prompt-from-llm",
            input_tokens=10,
            output_tokens=12,
            thoughts_tokens=3,
            finish_reason="STOP",
        )

    def get_model_name(self):
        return "gemini-2.5-flash"

    def get_generation_config(self):
        return {"thinking_budget": 200, "max_output_tokens": 512}


class FakeObservationHandle:
    def __init__(self, trace_id, span_id, name):
        self.trace_id = trace_id
        self.observation_id = span_id
        self.name = name
        self.updates = []
        self.scores = []
        self.closed = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score(self, scores):
        for score_input in scores.values():
            assert isinstance(score_input, ScoreInput)
            self.scores.append(score_input)

    def close(self):
        self.closed = True


class FakeObservabilityPort:
    def __init__(self):
        self.traces = []
        self.scores = []
        self.trace_handle = None
        self.all_handles = []

    def start_trace(self, name, payload, *, trace_name=None, user_id=None, session_id=None):
        self.traces.append((name, payload, trace_name, user_id, session_id))
        self.trace_handle = FakeObservationHandle("trace-123", "root-123", name)
        self.all_handles.append(self.trace_handle)
        return self.trace_handle

    def start_observation(self, name, as_type=None, model=None):
        handle = FakeObservationHandle("trace-123", f"{name}-123", name)
        self.all_handles.append(handle)
        return handle

    def handle(self, name):
        return next(h for h in self.all_handles if h.name == name)

    def span_names(self):
        return [h.name for h in self.all_handles]

    def finish_trace(self):
        pass

    def create_score(self, trace_id, name, value, comment=None, observation_id=None):
        self.scores.append((trace_id, name, value, comment, observation_id))


class FakeEvaluationPort:
    def __init__(self):
        self.items = []

    def enqueue(self, evaluation, trace_id, observation_id=None):
        self.items.append((evaluation, trace_id, observation_id))
