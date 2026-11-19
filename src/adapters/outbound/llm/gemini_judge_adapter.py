import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from src.application.ports.outbound.answer_judge_port import AnswerJudgePort
from src.config.prompt_judge import JUDGE_SYSTEM_MESSAGE, build_judge_prompt
from src.domain.model.faq_models import JudgeVerdict

logger = logging.getLogger("chatbot.backend.judge")


class JudgeOutput(BaseModel):
    is_answering: bool
    reason: str


class JudgeOutputError(RuntimeError):
    """The judge did not return a usable verdict."""


class GeminiJudgeAdapter(AnswerJudgePort):
    """LLM-as-a-judge on Gemini, with structured JSON output and a dedicated config."""

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        thinking_budget: int = 0,
        max_output_tokens: int = 256,
        seed: int = 42,
        client: Any = None,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._thinking_budget = thinking_budget
        self._max_output_tokens = max_output_tokens
        self._seed = seed
        if client is None:
            from google import genai

            client = genai.Client()
        self._client = client

    def judge(self, question: str, faq_question: str, faq_answer: str) -> JudgeVerdict:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=build_judge_prompt(question, faq_question, faq_answer),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_MESSAGE,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=self._thinking_budget),
                seed=self._seed,
                response_mime_type="application/json",
                response_schema=JudgeOutput,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        output = self._parse(response)
        return JudgeVerdict(is_answering=output.is_answering, reason=output.reason)

    @staticmethod
    def _parse(response: Any) -> JudgeOutput:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, JudgeOutput):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise JudgeOutputError("Empty judge response")
        try:
            return JudgeOutput.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise JudgeOutputError(f"Invalid judge response: {text!r}") from exc

    def get_model_name(self) -> str | None:
        return self._model_name
