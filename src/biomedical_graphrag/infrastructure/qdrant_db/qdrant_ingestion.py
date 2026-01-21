import asyncio

from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.json_util import load_gene_json, load_pubmed_json
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def ingest_data(recreate: bool = False, only_new: bool = True) -> None:
    """
    Ingest data from datasets into the Qdrant vector store (async).
    Optionally recreate the collection before ingesting.

    Args:
        recreate: If True, recreate the collection (deletes all existing data).
        only_new: If True, only ingest papers NOT present in the collection (there's no papers with such PMID).
    """
    logger.info("Starting Qdrant data ingestion...")

    vector_store = AsyncQdrantVectorStore()
    try:
        if await vector_store.client.collection_exists(vector_store.collection_name):
            if recreate:
                logger.info("Recreating collection...")
                await vector_store.delete_collection()
                await vector_store.create_collection()
        else:
            logger.info(
                f"Collection with name {vector_store.collection_name} does not exist. Creating..."
            )
            await vector_store.create_collection()

        # Load datasets
        logger.info("Loading datasets...")
        pubmed_data = load_pubmed_json()
        gene_data = load_gene_json()
        total_papers = len(pubmed_data.get("papers", []))
        logger.info(
            f"Loaded {total_papers} papers and {len(gene_data.get('genes', []))} genes"
        )
        # Upsert papers with attached genes in payload
        logger.info("Starting data upsertion...")
        await vector_store.upsert_points(pubmed_data, gene_data, only_new=only_new)
        logger.info("Embeddings ingestion complete.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await vector_store.close()
        logger.info("Qdrant client connection closed")


if __name__ == "__main__":
    asyncio.run(ingest_data(recreate=False, only_new=True))
