"""Script to create and populate Neo4j graph from biomedical papers dataset."""

import json
from pathlib import Path

from biomedical_graphrag.domain.dataset import GeneDataset, PaperDataset
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_graph_schema import Neo4jGraphIngestion
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


def load_paper_dataset(file_path: str | Path) -> PaperDataset:
    """
    Load dataset from JSON file.

    Args:
        file_path: Path to the JSON dataset file

    Returns:
        Dataset object with papers and citation network
    """
    with open(file_path) as f:
        data = json.load(f)

    return PaperDataset(**data)


def load_gene_dataset(file_path: str | Path) -> GeneDataset:
    """
    Load gene dataset from JSON file.

    Args:
        file_path: Path to the JSON dataset file

    Returns:
        Dataset object with genes
    """
    with open(file_path) as f:
        data = json.load(f)
    return GeneDataset(**data)


async def create_graph(
    dataset_path: str | Path | None = None, gene_dataset_path: str | Path | None = None
) -> None:
    """
    Create a new graph in the Neo4j graph database from biomedical papers dataset.

    Args:
        dataset_path: Optional path to dataset JSON file.
                     Defaults to data/pubmed_dataset.json in project root.

    Returns:
        None
    """
    # Use default path if none provided
    if dataset_path is None:
        project_root = Path(__file__).parents[4]
        dataset_path = project_root / "data" / "pubmed_dataset.json"
    if gene_dataset_path is None:
        project_root = Path(__file__).parents[4]
        gene_dataset_path = project_root / "data" / "gene_dataset.json"

    # Load dataset
    logger.info(f"Loading paper dataset from {dataset_path}...")
    dataset = load_paper_dataset(dataset_path)

    logger.info(f"Loaded {len(dataset.papers)} papers")

    # Load gene dataset (optional)
    gene_dataset: GeneDataset | None = None
    if Path(gene_dataset_path).exists():
        logger.info(f"Loading gene dataset from {gene_dataset_path}...")
        try:
            gene_dataset = load_gene_dataset(gene_dataset_path)
            logger.info(f"Loaded {len(gene_dataset.genes)} genes")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load gene dataset: {e}")
    else:
        logger.warning(f"No gene dataset found at {gene_dataset_path}; skipping gene ingestion")

    # Initialize Neo4j client
    client = await AsyncNeo4jClient.create()

    try:
        # Initialize ingestion handler
        ingestion = Neo4jGraphIngestion(client)

        # Ingest complete paper dataset
        await ingestion.ingest_paper_dataset(dataset)

        # Ingest genes if available
        if gene_dataset is not None:
            await ingestion.ingest_genes(gene_dataset)

        logger.info("✅ Graph created!")
        logger.info(f"   - {len(dataset.papers)} papers")
        logger.info(f"   - {dataset.metadata.total_authors} authors")
        logger.info(f"   - {dataset.metadata.total_mesh_terms} MeSH terms")
        logger.info("   - Citation network relationships created")
        if gene_dataset is not None:
            logger.info(f"   - {len(gene_dataset.genes)} genes and gene-paper relationships created")

    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_graph())
