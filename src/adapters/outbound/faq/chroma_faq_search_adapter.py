import logging
import os

# Must be set before chromadb is imported to disable PostHog telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.adapters.outbound.faq.faq_json_loader import load_faq_entries
from src.adapters.outbound.faq.text_normalizer import normalize_text
from src.application.ports.outbound.faq_search_port import FaqSearchPort
from src.domain.model.faq_models import FaqCandidate, FaqEntry

logger = logging.getLogger("chatbot.backend.faq_search")

DEFAULT_FAQ_EMBED_MODEL = "intfloat/multilingual-e5-small"


class ChromaFaqSearchAdapter(FaqSearchPort):
    """Dense FAQ search: bi-encoder + in-memory ChromaDB (rebuilt at startup),
    keeping one candidate per FAQ theme."""

    def __init__(
        self,
        faq_json_path: str,
        model_name: str = DEFAULT_FAQ_EMBED_MODEL,
        collection_name: str = "faq",
    ) -> None:
        self._model_name = model_name
        if "-e5-" in model_name:
            self._prefix_p, self._prefix_q = "passage: ", "query: "
        else:
            self._prefix_p, self._prefix_q = "", ""

        logger.info("faq.embedding_model.loading %s", model_name)
        self._model = SentenceTransformer(model_name)

        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "ip"},  # inner product on normalized embeddings == cosine
        )
        self._index(load_faq_entries(faq_json_path))

    def _index(self, entries: list[FaqEntry]) -> None:
        documents = [self._prefix_p + normalize_text(entry.formulation) for entry in entries]
        embeddings = self._model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        self._collection.add(
            ids=[f"faq_{i}" for i in range(len(entries))],
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[{"theme": e.theme, "response": e.response} for e in entries],
        )
        self._count = len(entries)
        logger.info("faq.index.ready entries=%d", self._count)

    def search(self, question: str, top_k: int, similarity_threshold: float) -> list[FaqCandidate]:
        query = self._prefix_q + normalize_text(question)
        query_embedding = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(max(top_k * 5, 20), self._count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return self._filter_and_dedup_by_theme(documents, metadatas, distances, similarity_threshold, top_k)

    def _filter_and_dedup_by_theme(
        self,
        documents: list[str],
        metadatas: list[dict],
        distances: list[float],
        similarity_threshold: float,
        top_k: int,
    ) -> list[FaqCandidate]:
        seen_themes: set[str] = set()
        candidates: list[FaqCandidate] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=True):
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                break  # results are sorted by similarity
            theme = metadata["theme"]
            if theme in seen_themes:
                continue
            seen_themes.add(theme)
            candidates.append(
                FaqCandidate(
                    formulation=document.removeprefix(self._prefix_p),
                    theme=theme,
                    response=metadata["response"],
                    similarity_score=float(similarity),
                )
            )
            if len(candidates) == top_k:
                break
        return candidates

    def count(self) -> int:
        return self._count

    def get_model_name(self) -> str | None:
        return self._model_name
