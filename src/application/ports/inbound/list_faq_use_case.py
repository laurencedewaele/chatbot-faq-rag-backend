from abc import ABC, abstractmethod

from src.application.dto.faq_catalog_dto import ListFaqResult


class ListFaqUseCase(ABC):
    @abstractmethod
    def execute(self) -> ListFaqResult:
        raise NotImplementedError
