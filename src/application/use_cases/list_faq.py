from src.application.dto.faq_catalog_dto import FaqIntentView, ListFaqResult
from src.application.ports.inbound.list_faq_use_case import ListFaqUseCase
from src.application.ports.outbound.faq_catalog_port import FaqCatalogPort


class ListFaqService(ListFaqUseCase):
    def __init__(self, faq_catalog_port: FaqCatalogPort) -> None:
        self._catalog = faq_catalog_port

    def execute(self) -> ListFaqResult:
        return ListFaqResult(
            intents=[
                FaqIntentView(
                    intent_key=intent.intent_key,
                    category=intent.category,
                    canonical_question=intent.canonical_question,
                    utterances=list(intent.utterances),
                    response=intent.response,
                )
                for intent in self._catalog.list_intents()
            ]
        )
