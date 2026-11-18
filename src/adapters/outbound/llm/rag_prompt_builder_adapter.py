from typing import List, Sequence

from src.application.ports.outbound.prompt_builder_port import PromptBuilderPort
from src.domain.model.faq_models import FaqCandidate
from src.domain.model.rag_models import Document


class RagPromptBuilderAdapter(PromptBuilderPort):
    def build_prompt(
        self,
        question: str,
        chunks: List[Document],
        faq_candidates: Sequence[FaqCandidate] = (),
    ) -> str:
        sections = []
        if faq_candidates:
            faq_block = "\n\n".join(
                f"Question FAQ : {candidate.formulation}\nRéponse FAQ : {candidate.response}"
                for candidate in faq_candidates
            )
            sections.append(f"RÉPONSES FAQ POTENTIELLEMENT LIÉES:\n{faq_block}")
        context = "\n\n".join(chunk.content for chunk in chunks)
        sections.append(f"CONTEXTE:\n{context}")
        sections.append(f"QUESTION:\n{question}")
        return "\n\n".join(sections)
