from abc import ABC, abstractmethod

from src.domain.model.faq_models import JudgeVerdict


class AnswerJudgePort(ABC):
    @abstractmethod
    def judge(self, question: str, faq_question: str, faq_answer: str) -> JudgeVerdict:
        """Decide whether `faq_answer` actually answers the user `question`.

        Implementations raise an exception when no reliable verdict can be produced.
        """
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None
