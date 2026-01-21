import asyncio
import argparse
from datetime import datetime
from typing import Any

from biomedical_graphrag.config import settings
from biomedical_graphrag.data_sources.base import BaseDataSource
from biomedical_graphrag.data_sources.pubmed.pubmed_api_client import PubMedAPIClient
from biomedical_graphrag.domain.author import Author
from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.dataset import PaperDataset, PaperMetadata
from biomedical_graphrag.domain.meshterm import MeSHTerm
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class PubMedDataCollector(BaseDataSource):
    """
    Data collector for PubMed, implements BaseDataSource (async).
    """

    def __init__(self) -> None:
        """
        Initialize the PubMedDataCollector with a PubMedAPIClient instance.

        Args:
            None
        Returns:
            None
        """

        super().__init__()
        self.api = PubMedAPIClient()

    async def search(self, query: str, max_results: int) -> list[str]:
        """
        Search PubMed for paper IDs matching the query (async).

        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
        Returns:
            list[str]: List of PubMed IDs (PMIDs) matching the query.
        """
        logger.info(f"Starting PubMed search for query='{query}' (max_results={max_results})")
        await self._rate_limit()
        ids = await self.api.search(query, max_results)
        logger.info(f"Found {len(ids)} PubMed IDs for query")
        return ids

    async def fetch_papers(self, paper_ids: list[str]) -> list[Paper]:
        """
        Fetch paper details from PubMed using a list of paper IDs (async).

        Args:
            paper_ids (list[str]): List of PubMed IDs (PMIDs) to fetch.
        Returns:
            list[Paper]: List of Paper objects containing details of the fetched papers.
        """
        logger.info(f"Fetching details for {len(paper_ids)} PubMed IDs")
        await self._rate_limit()
        raw_papers = await self.api.fetch_papers(paper_ids)
        logger.info(f"Fetched {len(raw_papers)} raw PubMed records; parsing into Paper models")
        papers: list[Paper] = []
        skipped = 0
        for r in raw_papers:
            parsed = self._parse_paper(r)
            if parsed is None:
                skipped += 1
                continue
            papers.append(parsed)
        logger.info(f"Parsed {len(papers)} papers")
        return papers

    async def fetch_citations(self, paper_id: str) -> dict:
        """
        Fetch citations (cited by and references) for a given paper ID (async).

        Args:
            paper_id (str): The PubMed ID (PMID) of the paper to fetch citations for.
        Returns:
            dict: A dictionary containing the citations for the paper.
        """
        logger.debug(f"Fetching citations for PMID={paper_id}")
        await self._rate_limit()
        citations = await self.api.fetch_citations(paper_id)
        return {"pmid": paper_id, **citations}

    # Common abstract method unification
    async def fetch_entities(self, entity_ids: list[str]) -> list[Any]:
        """
        Fetch paper details from PubMed using a list of paper IDs (async).

        Args:
            entity_ids (list[str]): List of PubMed IDs (PMIDs) to fetch.
        Returns:
            list[object]: List of Paper objects containing details of the fetched papers.
        """
        return await self.fetch_papers(entity_ids)

    # Keep explicit collect_dataset for symmetry with GeneDataCollector
    async def collect_dataset(self, query: str, max_results: int) -> PaperDataset:
        """Collect PubMed dataset for given query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to collect.

        Returns:
            PaperDataset containing collected papers and metadata.
        """
        logger.info("Collecting PubMed dataset...")
        paper_ids = await self.search(query, max_results)
        papers = await self.fetch_papers(paper_ids)

        # Fetch citations with controlled concurrency (max 10 at a time to match API limit)
        semaphore = asyncio.Semaphore(8)

        async def fetch_with_semaphore(paper: Paper) -> dict:
            """
            Fetch citations with semaphore.

            Args:
                paper (Paper): The paper to fetch citations for.
            Returns:
                dict: A dictionary containing the citations for the paper.
            """
            async with semaphore:
                return await self.fetch_citations(paper.pmid)

        citation_tasks = [fetch_with_semaphore(paper) for paper in papers]
        citations_list = await asyncio.gather(*citation_tasks)

        citation_network = {}
        for idx, (paper, citations) in enumerate(zip(papers, citations_list, strict=False), start=1):
            logger.debug(
                f"[{idx}/{len(papers)}] Processed citations for '{paper.title[:80]}' (PMID={paper.pmid})"
            )
            citation_network[paper.pmid] = CitationNetwork(**citations)

        total_authors = sum(len(paper.authors) for paper in papers)
        total_mesh_terms = sum(len(paper.mesh_terms) for paper in papers)
        logger.info(
            f"Computed totals: papers={len(papers)}, authors={total_authors}, \
            MeSH terms={total_mesh_terms}, with_citations={len(citation_network)}"
        )
        metadata = PaperMetadata(
            collection_date=datetime.now().isoformat(),
            query=query,
            total_papers=len(papers),
            papers_with_citations=len(citation_network),
            total_authors=total_authors,
            total_mesh_terms=total_mesh_terms,
        )
        logger.info("PubMed dataset collection complete")
        return PaperDataset(metadata=metadata, papers=papers, citation_network=citation_network)

    def _parse_paper(self, record: dict) -> Paper | None:
        """
        Parse a raw PubMed record into a Paper object.

        Args:
            record (dict): Raw PubMed record as returned by the API.
        Returns:
            Paper: Parsed Paper object with extracted details.
        """
        medline = record.get("MedlineCitation", {})
        article = medline.get("Article", {})
        pmid = str(medline.get("PMID", "")).strip()
        if not pmid:
            logger.debug("Skipping PubMed record with empty PMID")
            return None
        title = article.get("ArticleTitle", "")
        abstract = self._extract_abstract(article)
        authors = self._extract_authors(article)
        mesh_terms = self._extract_mesh_terms(medline)
        pub_date = self._extract_pub_date(article)
        journal = article.get("Journal", {}).get("Title", "")
        doi = None
        article_ids = record.get("PubmedData", {}).get("ArticleIdList", [])
        for aid in article_ids:
            if hasattr(aid, "attributes") and aid.attributes.get("IdType") == "doi":
                doi = str(aid)
        return Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            mesh_terms=mesh_terms,
            publication_date=pub_date,
            journal=journal,
            doi=doi or "",
        )

    def _extract_abstract(self, article: dict) -> str:
        """
        Extract the abstract text from a PubMed article record.

        Args:
            article (dict): The PubMed article record.

        Returns:
            str: The extracted abstract text.
        """
        abstract = ""
        if "Abstract" in article:
            abstract_parts = article["Abstract"].get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
        return abstract

    def _extract_authors(self, article: dict) -> list:
        """
        Extract the authors from a PubMed article record.

        Args:
            article (dict): The PubMed article record.

        Returns:
            list: A list of Author objects.
        """

        authors: list[Author] = []
        if "AuthorList" in article:
            for author in article["AuthorList"]:
                # Extract name fields
                if "LastName" in author and "ForeName" in author:
                    name = f"{author['ForeName']} {author['LastName']}"
                    last_name = author["LastName"]
                    first_name = author["ForeName"]
                elif "CollectiveName" in author:
                    name = author["CollectiveName"]
                    last_name = ""
                    first_name = ""
                else:
                    continue

                # Robustly handle AffiliationInfo as list, dict, str, or missing
                if "AffiliationInfo" in author:
                    aff_info = author["AffiliationInfo"]
                    if isinstance(aff_info, list):
                        affiliations = [str(aff.get("Affiliation", "")) for aff in aff_info]
                    elif isinstance(aff_info, dict):
                        affiliations = [str(aff_info.get("Affiliation", ""))]
                    elif isinstance(aff_info, str):
                        affiliations = [aff_info]
                    else:
                        affiliations = []
                else:
                    affiliations = []

                authors.append(
                    Author(
                        name=name,
                        first_name=first_name,
                        last_name=last_name,
                        affiliations=affiliations,
                    )
                )
        return authors

    def _extract_mesh_terms(self, medline: dict) -> list:
        """
        Extract the MeSH terms from a PubMed article record.

        Args:
            medline (dict): The MedlineCitation part of the PubMed record.

        Returns:
            list: A list of MeSHTerm objects.
        """

        mesh_terms = []
        if "MeshHeadingList" in medline:
            for mesh in medline["MeshHeadingList"]:
                descriptor = mesh.get("DescriptorName", {})
                mesh_info: dict = {
                    "term": str(descriptor),
                    "major_topic": descriptor.attributes.get("MajorTopicYN") == "Y"
                    if hasattr(descriptor, "attributes")
                    else False,
                    "ui": descriptor.attributes.get("UI", "")
                    if hasattr(descriptor, "attributes")
                    else "",
                    "qualifiers": [],
                }
                if "QualifierName" in mesh:
                    qualifiers = mesh["QualifierName"]
                    if not isinstance(qualifiers, list):
                        qualifiers = [qualifiers]
                    mesh_info["qualifiers"] = [str(q) for q in qualifiers]
                mesh_terms.append(MeSHTerm(**mesh_info))
        return mesh_terms

    def _extract_pub_date(self, article: dict) -> str:
        """
        Extract the publication date from a PubMed article record.

        Args:
            article (dict): The PubMed article record.

        Returns:
            str: The extracted publication date in YYYY-MM-DD format.
        """
        pub_date = ""
        if "Journal" in article and "JournalIssue" in article["Journal"]:
            pub_date_dict = article["Journal"]["JournalIssue"].get("PubDate", {})
            year = pub_date_dict.get("Year", "")
            month = pub_date_dict.get("Month", "01")
            day = pub_date_dict.get("Day", "01")
            month_map = {
                "Jan": "01",
                "Feb": "02",
                "Mar": "03",
                "Apr": "04",
                "May": "05",
                "Jun": "06",
                "Jul": "07",
                "Aug": "08",
                "Sep": "09",
                "Oct": "10",
                "Nov": "11",
                "Dec": "12",
            }
            month = month_map.get(month, month)
            if year:
                pub_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return pub_date


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect PubMed dataset and write it to JSON.")
    parser.add_argument(
        "--query",
        type=str,
        default="CRISPR gene editing Homo sapiens[Organism]",
        help="PubMed search query (ESearch term).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum number of PubMed papers (PMIDs) to collect.",
    )
    args = parser.parse_args()

    async def main() -> None:
        """
        Main function to collect PubMed dataset.

        Returns:
            None
        """
        api_key = settings.pubmed.api_key
        email = settings.pubmed.email
        print("Using email:", email)
        print("Using api_key:", api_key)
        collector = PubMedDataCollector()
        dataset = await collector.collect_dataset(
            query=args.query,
            max_results=args.max_results,
        )
        with open(settings.json_data.pubmed_json_path, "w") as f:
            f.write(dataset.model_dump_json(indent=2))

    asyncio.run(main())
