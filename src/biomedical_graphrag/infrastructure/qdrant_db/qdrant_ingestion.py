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
        only_new: If True, only ingest papers not already in the collection.
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

        # Filter to only new papers if requested
        if only_new and not recreate:
            existing_ids = await vector_store.get_existing_ids()
            if existing_ids:
                original_papers = pubmed_data.get("papers", [])
                new_papers = [
                    p for p in original_papers
                    if p.get("pmid") and int(p["pmid"]) not in existing_ids
                ]
                pubmed_data["papers"] = new_papers
                logger.info(
                    f"Filtered to {len(new_papers)} new papers "
                    f"(skipping {total_papers - len(new_papers)} existing)"
                )

                if not new_papers:
                    logger.info("No new papers to ingest. Done.")
                    return

        # Upsert papers with attached genes in payload
        logger.info("Starting data upsertion...")
        await vector_store.upsert_points(pubmed_data, gene_data)
        logger.info("Embeddings ingestion complete.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await vector_store.close()
        logger.info("Qdrant client connection closed")


if __name__ == "__main__":
    asyncio.run(ingest_data(recreate=False, only_new=True))
