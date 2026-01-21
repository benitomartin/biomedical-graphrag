"""
Paper enrichment module - finds related papers and adds them to the dataset.

For each paper in the dataset, searches PubMed using title + abstract,
and adds one related paper if not already present.
"""

import asyncio
import argparse
import json
import re

from biomedical_graphrag.config import settings
from biomedical_graphrag.data_sources.base import BaseDataSource
from biomedical_graphrag.data_sources.pubmed.pubmed_api_client import PubMedAPIClient
from biomedical_graphrag.data_sources.pubmed.pubmed_data_collector import PubMedDataCollector
from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.dataset import PaperDataset
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class PaperEnrichmentCollector(BaseDataSource):
    """Enriches existing PubMed dataset by finding related papers."""

    def __init__(self) -> None:
        super().__init__()
        self.api = PubMedAPIClient()
        self._parser = PubMedDataCollector()

    def _clean_query(self, text: str) -> str:
        """
        Clean text for use as PubMed search query.

        Args:
            text (str): Raw text to clean (typically title or abstract).

        Returns:
            str: Cleaned text with special characters removed and whitespace normalized.
        """
        # Remove special characters that break PubMed queries
        text = re.sub(r'["\[\]{}():;,\-\']', ' ', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_search_query(self, paper: Paper) -> str:
        """
        Extract key terms from paper title for PubMed search.

        Args:
            paper (Paper): Paper object to extract search terms from.

        Returns:
            str: Space-separated key terms (up to 5 significant words).
        """
        # Clean the title
        clean_title = self._clean_query(paper.title)
        # Take first 5-6 significant words (skip common stopwords)
        stopwords = {
            'a', 'an', 'the', 'of', 'in', 'on', 'for', 'and',
            'or', 'to', 'with', 'by', 'as', 'is', 'are', 'from',
        }
        words = [w for w in clean_title.split() if w.lower() not in stopwords]
        # Return first 5 significant words
        return ' '.join(words[:5])

    async def search(self, query: str, max_results: int = 5) -> list[str]:
        """
        Search PubMed for related papers.

        Args:
            query (str): Search query string.
            max_results (int): Maximum number of results to return. Defaults to 5.

        Returns:
            list[str]: List of PMIDs matching the search query.
        """
        await self._rate_limit()
        return await self.api.search(query, max_results=max_results)

    async def fetch_entities(self, entity_ids: list[str]) -> list[Paper]:
        """
        Fetch paper entities.

        Args:
            entity_ids (list[str]): List of PMIDs to fetch.

        Returns:
            list[Paper]: List of Paper objects for the requested PMIDs.
        """
        return await self.fetch_papers(entity_ids)

    async def fetch_papers(self, paper_ids: list[str], batch_size: int = 200) -> list[Paper]:
        """
        Fetch paper details from PubMed in batches with progress logging.

        Args:
            paper_ids (list[str]): List of PMIDs to fetch.
            batch_size (int): Number of papers to fetch per batch. Defaults to 200.

        Returns:
            list[Paper]: List of successfully fetched and parsed Paper objects.
        """
        if not paper_ids:
            return []

        papers = []
        total = len(paper_ids)

        for i in range(0, total, batch_size):
            batch = paper_ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(
                f"Fetching batch {batch_num}/{total_batches} "
                f"({len(batch)} papers, {len(papers)}/{total} done)"
            )

            await self._rate_limit()
            try:
                raw_papers = await self.api.fetch_papers(batch)
                for r in raw_papers:
                    parsed = self._parser._parse_paper(r)
                    if parsed:
                        papers.append(parsed)
            except Exception as e:
                logger.warning(f"Batch {batch_num} failed: {e}, retrying with smaller batch...")
                # Retry with smaller batches
                for j in range(0, len(batch), 50):
                    mini_batch = batch[j:j + 50]
                    await self._rate_limit()
                    try:
                        raw_papers = await self.api.fetch_papers(mini_batch)
                        for r in raw_papers:
                            parsed = self._parser._parse_paper(r)
                            if parsed:
                                papers.append(parsed)
                    except Exception as e2:
                        logger.error(f"Mini-batch failed: {e2}, skipping {len(mini_batch)} papers")

        logger.info(f"Fetched {len(papers)}/{total} papers successfully")
        return papers

    async def fetch_citations(self, paper_id: str) -> dict:
        """
        Fetch citations for a paper.

        Args:
            paper_id (str): PMID of the paper to fetch citations for.

        Returns:
            dict: Citation data including the PMID and citation relationships.
        """
        await self._rate_limit()
        citations = await self.api.fetch_citations(paper_id)
        return {"pmid": paper_id, **citations}

    async def _find_related_papers(
        self,
        source_paper: Paper,
        existing_pmids: set[str],
        max_new: int = 5,
    ) -> list[str]:
        """
        Find related papers not already in the dataset.

        Args:
            source_paper (Paper): Paper to find related papers for.
            existing_pmids (set[str]): Set of PMIDs already in the dataset.
            max_new (int): Maximum number of new papers to find. Defaults to 5.

        Returns:
            list[str]: List of PMIDs for related papers not in the existing set.
        """
        query = self._extract_search_query(source_paper)
        if not query:
            return []

        try:
            # Search for more results to find multiple new papers
            pmids = await self.search(query, max_results=50)
            new_pmids = []
            for pmid in pmids:
                if pmid != source_paper.pmid and pmid not in existing_pmids:
                    new_pmids.append(pmid)
                    if len(new_pmids) >= max_new:
                        break
            return new_pmids
        except Exception as e:
            logger.warning(f"Search failed for PMID={source_paper.pmid}: {e}")
            return []

    async def enrich_dataset(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        fetch_citations: bool = True,
        max_papers_to_process: int | None = None,
        related_per_paper: int = 5,
        start_index: int = 0,
    ) -> PaperDataset:
        """
        Enrich existing dataset with related papers.

        Args:
            input_path: Path to input dataset (defaults to settings path)
            output_path: Path to save enriched dataset (defaults to input path)
            fetch_citations: Whether to fetch citations for new papers
            max_papers_to_process: Limit number of papers to process (for testing)
            related_per_paper: How many related papers to find per source paper
            start_index: Index to start processing from (skip papers already used as sources)

        Returns:
            Enriched PaperDataset
        """
        input_path = input_path or settings.json_data.pubmed_json_path
        output_path = output_path or input_path

        logger.info(f"Loading dataset from {input_path}")
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        dataset = PaperDataset(**data)
        existing_pmids = {p.pmid for p in dataset.papers}
        initial_count = len(existing_pmids)

        # Skip papers already used as sources
        papers_to_process = dataset.papers[start_index:]
        if max_papers_to_process:
            papers_to_process = papers_to_process[:max_papers_to_process]

        logger.info(f"Skipping first {start_index} papers (already used as sources)")

        logger.info(
            f"Processing {len(papers_to_process)} papers for enrichment "
            f"(up to {related_per_paper} related each)"
        )

        # Find related papers with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Higher with API key
        new_pmids: list[str] = []

        async def process_paper(paper: Paper) -> list[str]:
            async with semaphore:
                return await self._find_related_papers(paper, existing_pmids, max_new=related_per_paper)

        tasks = [process_paper(p) for p in papers_to_process]
        results = await asyncio.gather(*tasks)

        for related_list in results:
            for related_pmid in related_list:
                if related_pmid not in existing_pmids:
                    new_pmids.append(related_pmid)
                    existing_pmids.add(related_pmid)

        logger.info(f"Found {len(new_pmids)} new related papers")

        # Fetch details for new papers in batches, saving after each batch
        if new_pmids:
            logger.info(f"Fetching details for {len(new_pmids)} new papers")
            batch_size = 200
            total_batches = (len(new_pmids) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(new_pmids), batch_size):
                batch_pmids = new_pmids[batch_idx:batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1

                logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch_pmids)} papers)")
                new_papers = await self.fetch_papers(batch_pmids)
                dataset.papers.extend(new_papers)

                # Fetch citations if requested
                if fetch_citations and new_papers:
                    async def fetch_with_semaphore(paper: Paper) -> dict:
                        async with semaphore:
                            return await self.fetch_citations(paper.pmid)

                    citation_tasks = [fetch_with_semaphore(p) for p in new_papers]
                    citations_list = await asyncio.gather(*citation_tasks)

                    for paper, citations in zip(new_papers, citations_list, strict=False):
                        dataset.citation_network[paper.pmid] = CitationNetwork(**citations)

                # Update metadata and save after each batch
                dataset.metadata.total_papers = len(dataset.papers)
                dataset.metadata.papers_with_citations = len(dataset.citation_network)
                dataset.metadata.total_authors = sum(len(p.authors) for p in dataset.papers)
                dataset.metadata.total_mesh_terms = sum(len(p.mesh_terms) for p in dataset.papers)

                # Save incrementally
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(dataset.model_dump_json(indent=2))
                logger.info(f"Saved {len(dataset.papers)} papers to {output_path}")

        # Final metadata update
        dataset.metadata.total_papers = len(dataset.papers)
        dataset.metadata.papers_with_citations = len(dataset.citation_network)
        dataset.metadata.total_authors = sum(len(p.authors) for p in dataset.papers)
        dataset.metadata.total_mesh_terms = sum(len(p.mesh_terms) for p in dataset.papers)

        logger.info(
            f"Enrichment complete: {initial_count} -> {len(dataset.papers)} papers "
            f"(+{len(new_pmids)} new)"
        )

        return dataset

    async def collect_dataset(self, query: str = "", max_results: int = 0) -> PaperDataset:
        """
        Collect dataset by enriching existing papers with related papers.

        Required by BaseDataSource interface - delegates to enrich_dataset.

        Args:
            query (str): Unused, kept for interface compatibility. Defaults to "".
            max_results (int): Maximum papers to process. Defaults to 0 (no limit).

        Returns:
            PaperDataset: Enriched dataset with related papers added.
        """
        return await self.enrich_dataset(max_papers_to_process=max_results or None, related_per_paper=5)


async def main() -> None:
    """
    Main function to run paper enrichment.

    Loads the dataset, enriches it with related papers, and saves the result.
    Starts from specific index to skip papers already used as sources.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Enrich an existing PubMed dataset by adding related papers.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing from (skip papers already used as sources).",
    )
    args = parser.parse_args()

    enricher = PaperEnrichmentCollector()
    enriched_dataset = await enricher.enrich_dataset(start_index=args.start_index)

    output_path = settings.json_data.pubmed_json_path
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(enriched_dataset.model_dump_json(indent=2))

    logger.info(f"Saved enriched dataset to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
