import os
from dataclasses import dataclass
from pathlib import Path

from src.application.dto.ask_question_dto import FaqLookupParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(frozen=True)
class FaqSettings:
    """FAQ + judge configuration, read once from the environment (after load_dotenv)."""

    faq_json_path: str
    embed_model_name: str
    cross_encoder_model_name: str
    top_k: int
    similarity_threshold: float
    accept_threshold: float
    judge_threshold: float
    judge_model_name: str
    judge_temperature: float
    judge_thinking_budget: int
    judge_thinking_level: str
    judge_max_output_tokens: int

    @classmethod
    def from_env(cls) -> "FaqSettings":
        return cls(
            faq_json_path=_resolve_path(os.getenv("FAQ_JSON_PATH", "data/260817_FAQ_Miss_Terry.json")),
            embed_model_name=os.getenv("FAQ_EMBED_MODEL_NAME", "intfloat/multilingual-e5-small"),
            cross_encoder_model_name=os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2"),
            top_k=int(os.getenv("FAQ_TOP_K", "3")),
            similarity_threshold=float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.85")),
            accept_threshold=float(os.getenv("FAQ_ACCEPT_THRESHOLD", "0.96")),
            judge_threshold=float(os.getenv("FAQ_JUDGE_THRESHOLD", "0.60")),
            judge_model_name=os.getenv("JUDGE_MODEL_NAME", "gemini-3.5-flash-lite"),
            judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
            judge_thinking_budget=int(os.getenv("JUDGE_THINKING_BUDGET", "0")),
            judge_thinking_level=os.getenv("JUDGE_THINKING_LEVEL", "minimal"),
            judge_max_output_tokens=int(os.getenv("JUDGE_MAX_OUTPUT_TOKENS", "512")),
        )

    def lookup_defaults(self) -> FaqLookupParams:
        return FaqLookupParams(
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            accept_threshold=self.accept_threshold,
            judge_threshold=self.judge_threshold,
        )
