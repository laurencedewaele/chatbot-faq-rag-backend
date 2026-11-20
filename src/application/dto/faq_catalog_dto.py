from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FaqIntentView:
    intent_key: str
    category: Optional[str]
    canonical_question: Optional[str]
    utterances: List[str]
    response: str


@dataclass(frozen=True)
class ListFaqResult:
    intents: List[FaqIntentView]
