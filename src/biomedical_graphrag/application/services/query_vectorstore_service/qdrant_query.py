from openai import AsyncOpenAI

from biomedical_graphrag.application.services.query_vectorstore_service.prompts.qdrant_prompts import (
    QDRANT_GENERATION_PROMPT,
)
from biomedical_graphrag.config import settings
from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.logger_util import setup_logging

from qdrant_client.models import models

logger = setup_logging()


class AsyncQdrantQuery:
    """Handles querying Qdrant vector store for natural language questions (async)."""

    def __init__(self) -> None:
        """
        Initialize the async Qdrant client with connection parameters.
        """
        self.url = settings.qdrant.url
        self.api_key = settings.qdrant.api_key
        self.collection_name = settings.qdrant.collection_name
        self.embedding_dimension = settings.qdrant.embedding_dimension
        self.reranker_embedding_dimension = settings.qdrant.reranker_embedding_dimension
        self.cloud_inference = settings.qdrant.cloud_inference

        self.openai_client = AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value())

        self.qdrant_client = AsyncQdrantVectorStore()

    async def close(self) -> None:
        """Close the async Qdrant client."""
        await self.qdrant_client.close()

    async def retrieve_documents_dense(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Query the Qdrant vector store for similar documents (async).
        Vanilla dense search on quantized openAI embeddings.

        Args:
            question (str): Input question to query.
            top_k (int): Number of top similar documents to retrieve.
        Returns:
            List of dictionaries containing the top_k similar documents.
        """
        dense_vector = await self.qdrant_client._get_openai_vectors(question)

        search_result = await self.qdrant_client.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="Dense",
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    oversampling=3.0,  # retrieve 3 * top_k quantized vectors
                    rescore=True,  # to rescore with original vectors
                )
            ),
            limit=top_k,
            with_payload=True,
        )
        results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in search_result.points
        ]
        return results

    async def retrieve_documents_hybrid(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Query the Qdrant vector search engine (async).
        Hybrid dense + lexical search, fused by reranking with text-embedding-3-large.

        Args:
            question (str): Input question - query.
            top_k (int): Number of top similar documents to retrieve.
        Returns:
            List of dictionaries containing the top_k similar documents.
        """
        retriever_vector = await self.qdrant_client._get_openai_vectors(
            question, dimensions=self.embedding_dimension
        )
        reranker_vector = await self.qdrant_client._get_openai_vectors(
            question, dimensions=self.reranker_embedding_dimension
        )
        sparse_vector = self.qdrant_client._define_bm25_vectors(question)

        search_result = await self.qdrant_client.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=retriever_vector,
                    using="Dense",
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            oversampling=3.0,  # retrieve 3 * top_k quantized vectors
                            rescore=True,  # to rescore with original vectors
                        )
                    ),
                    limit=top_k,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="Lexical",
                    limit=top_k,
                ),
            ],
            # query=models.RrfQuery(rrf=models.Rrf(k=60)),
            query=reranker_vector,
            using="Reranker",
            limit=top_k,
            with_payload=True,
        )

        results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in search_result.points
        ]
        return results

    async def get_answer(self, question: str) -> str:
        """
        Get an answer to the question by retrieving relevant documents from Qdrant (async).

        Args:
            question (str): The question to answer.
        Returns:
            str: The answer generated from the retrieved documents.
        """

        context = QDRANT_GENERATION_PROMPT.format(
            question=question,
            context="\n".join(
                f"Content: {doc['payload']}" for doc in await self.retrieve_documents_hybrid(question)
            ),
        )
        logger.debug(f"Qdrant context: {context}")

        response = await self.openai_client.chat.completions.create(
            model=settings.openai.model,
            messages=[{"role": "user", "content": context}],
            max_tokens=settings.openai.max_tokens,
        )
        return response.choices[0].message.content or ""
