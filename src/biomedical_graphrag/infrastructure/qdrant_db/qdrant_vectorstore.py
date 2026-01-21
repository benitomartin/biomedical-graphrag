from typing import Any

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointsList, models

from biomedical_graphrag.config import settings

# from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.gene import GeneRecord
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class AsyncQdrantVectorStore:
    """
    Async Qdrant client for managing collections and points.
    """

    def __init__(self) -> None:
        """
        Initialize the async Qdrant client with connection parameters.
        """
        self.url = settings.qdrant.url
        self.api_key = settings.qdrant.api_key
        self.collection_name = settings.qdrant.collection_name
        self.embedding_dimension = settings.qdrant.embedding_dimension
        self.reranker_embedding_dimension = settings.qdrant.reranker_embedding_dimension
        self.estimate_bm25_avg_len_on_x_docs = settings.qdrant.estimate_bm25_avg_len_on_x_docs
        self.cloud_inference = settings.qdrant.cloud_inference

        self.openai_client = AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value())

        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            cloud_inference=self.cloud_inference,
        )

    async def close(self) -> None:
        """Close the async Qdrant client."""
        await self.client.close()

    async def create_collection(self) -> None:
        """
        Create a new collection in Qdrant (async) for hybrid search with MRL-based reranking
        Retriever's embeddings are quantized using scalar quantization.
        Args:
                collection_name (str): Name of the collection.
                kwargs: Additional parameters for collection creation.
        """
        logger.info(f"🔧 Creating Qdrant collection: {self.collection_name}")
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "Dense": models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                        )
                    ),
                ),
                "Reranker": models.VectorParams(
                    size=self.reranker_embedding_dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={"Lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logger.info(f"✅ Collection '{self.collection_name}' created successfully")

    async def delete_collection(self) -> None:
        """
        Delete a collection from Qdrant (async).
        Args:
                collection_name (str): Name of the collection.
        """
        logger.info(f"🗑️ Deleting Qdrant collection: {self.collection_name}")
        await self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"✅ Collection '{self.collection_name}' deleted successfully")

    async def _get_openai_vectors(self, text: str, dimensions: int) -> list[float]:
        """
        Get the embedding vector for the given text (async).
        Args:
                text (str): Input text to embed.
                dimensions (int): Number of dimensions for the embedding. https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
        Returns:
                list[float]: The embedding vector.
        """
        try:
            embedding = await self.openai_client.embeddings.create(
                model=settings.qdrant.embedding_model, input=text, dimensions=dimensions
            )
            return embedding.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Failed to create embedding: {e}")
            raise

    def _define_openai_vectors(self, text: str, dimensions: int = 1536) -> models.Document:
        """
        Wrap text in models.Document to handle OpenAI embeddings inference
        through Qdrant's Cloud.

        Args:
                text (str): Input text.
                dimensions (int): Number of dimensions for the embedding. https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
        Returns:
                models.Document: Document object.
        """
        return models.Document(
            text=text,
            model=f"openai/{settings.qdrant.embedding_model}",
            options={
                "openai-api-key": settings.openai.api_key.get_secret_value(),
                "dimensions": dimensions,
            },
        )

    def _define_bm25_vectors(self, text: str, avg_len: int = 256) -> models.Document:
        """
        Wrap text in models.Document to handle BM25 sparse vectors inference
        in Qdrant's core.

        Args:
                text (str): Input text.
                avg_len (int): Average document length estimation for BM25 formula.
        Returns:
                models.Document: Document object.
        """
        return models.Document(
            text=text, model="qdrant/bm25", options={"avg_len": avg_len, "language": "english"}
        )

    async def upsert_points(
        self, pubmed_data: dict[str, Any], gene_data: dict[str, Any] | None = None, only_new: bool = False, batch_size: int = 30
    ) -> None:
        """
        Upsert points into a collection from pubmed_dataset.json structure,
        attaching related genes from gene_dataset.json into the payload (async).
        Args:
            pubmed_data (dict): Parsed JSON data from pubmed_dataset.json.
            gene_data (dict | None): Parsed JSON data from gene_dataset.json.
            only_new (bool): If True, only ingest papers NOT present in the collection (there's no papers with such PMID).
            batch_size (int): Number of points to process in each batch.
        """
        papers = pubmed_data.get("papers", [])
        # citation_network_dict = pubmed_data.get("citation_network", {}) # pretty much meaningless for Qdrant, if having Neo4j & agent

        logger.info(f"📚 Starting ingestion of {len(papers)} papers with batch size {batch_size}")

        # Build PMID -> [GeneRecord] index
        pmid_to_genes: dict[str, list[GeneRecord]] = {}
        if gene_data is not None:
            genes = gene_data.get("genes", [])
            logger.info(f"🧬 Processing {len(genes)} genes for paper-gene relationships")
            for gene in genes:
                record = GeneRecord(**gene)
                for linked_pmid in record.linked_pmids:
                    pmid_to_genes.setdefault(linked_pmid, []).append(record)
            logger.info(f"🔗 Built gene index for {len(pmid_to_genes)} papers")

        total_processed = 0
        total_skipped = 0

        # Estimate average abstract length for BM25
        total_words = 0
        sampled_count = 0
        for paper in papers:
            if paper.get("abstract"):
                total_words += len(paper["abstract"].split())
                sampled_count += 1
                if sampled_count >= self.estimate_bm25_avg_len_on_x_docs:
                    break
        avg_abstracts_len = (
            total_words // sampled_count
            if sampled_count > 0
            else None
        )

        if avg_abstracts_len is not None:
            logger.info(
                f"📏 Estimated average abstract length for BM25 formula on {sampled_count} abstracts: {avg_abstracts_len} words"
            )
        else:
            avg_abstracts_len = 256
            logger.info(
                "📏 Could not estimate average abstract length for BM25 formula, using default value of 256"
            )

        # Process papers in batches
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size

            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_papers)} papers)")

            batch_upsert_operations = [] # for only_new True
            batch_points = [] # for only_new False
            batch_skipped = 0

            for paper in batch_papers:
                pmid = paper.get("pmid")
                title = paper.get("title")
                abstract = paper.get("abstract")
                publication_date = paper.get("publication_date")
                journal = paper.get("journal")
                doi = paper.get("doi")
                authors = paper.get("authors", [])
                mesh_terms = paper.get("mesh_terms", [])

                if not abstract or not pmid:
                    batch_skipped += 1
                    logger.info(
                        f"⚠️ Skipping paper {pmid or 'unknown'} due to missing fields:"
                        f"Abstract: {not (bool(abstract))}, PMID: {not (bool(pmid))}"
                    )
                    continue

                try:    
                    point_id = int(pmid)
                    paper_model = Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        mesh_terms=mesh_terms,
                        publication_date=publication_date,
                        journal=journal,
                        doi=doi,
                    )

                    payload = {
                        "paper": paper_model.model_dump(),
                        # "citation_network": citation_network.model_dump() if citation_network else None,
                        "genes": [g.model_dump() for g in pmid_to_genes.get(pmid, [])],
                    }

                    if self.cloud_inference:
                        retriever_vector = self._define_openai_vectors(
                            abstract, dimensions=self.embedding_dimension
                        )
                        reranker_vector = self._define_openai_vectors(
                            abstract, dimensions=self.reranker_embedding_dimension
                        )
                    else:
                        openai_vector = await self._get_openai_vectors(abstract, dimensions=self.reranker_embedding_dimension)  # MRL, https://platform.openai.com/docs/guides/embeddings#use-cases
                        retriever_vector = openai_vector[:self.embedding_dimension] # Qdrant normalizes vectors used with COSINE automatically on upsert/query 
                        reranker_vector = openai_vector #reranking vector is more precise, hence, has more dimensions
                        
                    sparse_vector = self._define_bm25_vectors(abstract, avg_len=avg_abstracts_len)

                    point = models.PointStruct(
                        id=point_id,
                        vector={
                            "Dense": retriever_vector,
                            "Reranker": reranker_vector,
                            "Lexical": sparse_vector,
                        },
                        payload=payload,
                    )

                    # Get citation network for this paper if available
                    # citation_info = citation_network_dict.get(pmid, {})
                    # citation_network = CitationNetwork(**citation_info) if citation_info else None

                    if only_new:
                        batch_upsert_operations.append(
                            models.UpsertOperation(
                                upsert=PointsList(
                                    points=[point],
                                    update_filter=models.Filter(              
                                        must_not=[
                                            models.HasIdCondition(has_id=[point_id]),
                                        ]
                                    )
                                )
                            )
                        )
                    else:
                        batch_points.append(point)

                except Exception as e:
                    logger.error(f"❌ Failed to process paper {pmid}: {e}")
                    batch_skipped += 1
                    continue

            # Upsert batch if we have any valid papers

            if only_new: 
                if batch_upsert_operations:
                    try:
                        await self.client.batch_update_points(
                            collection_name=self.collection_name,
                            update_operations=batch_upsert_operations,
                        )
                        total_skipped += batch_skipped
                        logger.info(
                            f"✅ Batch {batch_num} completed: {len(batch_upsert_operations)} papers upserted, \
                                {batch_skipped} skipped"
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to upsert batch {batch_num}: {e}")
                        raise
                else:
                    logger.warning(f"⚠️ Batch {batch_num} had no valid papers to process")
            else:
                if batch_points:
                    try:
                        await self.client.upsert(
                            collection_name=self.collection_name,
                            points=PointsList(
                                points=batch_points,
                            ),
                        )
                        total_skipped += batch_skipped
                        logger.info(
                            f"✅ Batch {batch_num} completed: {len(batch_points)} papers upserted, \
                                {batch_skipped} skipped"
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to upsert batch {batch_num}: {e}")
                        raise
                else:
                    logger.warning(f"⚠️ Batch {batch_num} had no valid papers to process")

        logger.info(
            f"🎉 Ingestion complete! Total: {total_processed} papers processed, {total_skipped} skipped"
        )
