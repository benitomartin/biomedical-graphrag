from typing import Any

from neo4j import GraphDatabase

from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class Neo4jGraphQuery:
    """
    Handles querying Neo4j graph using predefined Cypher templates for biomedical enrichment.
    All query templates are static methods in this class.
    """

    def __init__(self) -> None:
        self.uri = settings.neo4j.uri
        self.username = settings.neo4j.username
        self.password = settings.neo4j.password.get_secret_value()
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a raw Cypher query against the graph.
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def get_schema(self) -> str:
        """
        Get the Neo4j graph schema for biomedical data.
        """
        return """
        Biomedical Graph Schema:

        Nodes:
        - Paper: {pmid, title, abstract, publication_date, doi}
        - Author: {name}
        - Institution: {name}
        - MeshTerm: {ui, term}
        - Journal: {name}
        - Gene: {gene_id, name, description, chromosome, map_location, organism, aliases, designations}

        Relationships:
        - (Author)-[:WROTE]->(Paper)
        - (Author)-[:AFFILIATED_WITH]->(Institution)
        - (Paper)-[:HAS_MESH_TERM {major_topic: boolean, qualifiers: [string]}]->(MeshTerm)
        - (Paper)-[:PUBLISHED_IN]->(Journal)
        - (Paper)-[:CITES]->(Paper)
        - (Gene)-[:MENTIONED_IN]->(Paper)
        """

    def get_collaborators_with_topics(
        self, author_name: str, topics: list[str], require_all: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get collaborators for an author filtered by MeSH topics.
        """
        if require_all:
            topic_matches = "\n".join(
                [
                    f"MATCH (p)-[:HAS_MESH_TERM]->(m{i}:MeshTerm) WHERE m{i}.term CONTAINS '{topic}'"
                    for i, topic in enumerate(topics)
                ]
            )
            cypher = f"""
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE a1.name CONTAINS '{author_name}' AND a1 <> a2
                WITH DISTINCT a2, p
                {topic_matches}
                RETURN DISTINCT a2.name as collaborator, COUNT(DISTINCT p) as papers
                ORDER BY papers DESC
                LIMIT 10
            """
        else:
            topic_conditions = " OR ".join([f"m.term CONTAINS '{topic}'" for topic in topics])
            cypher = f"""
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE a1.name CONTAINS '{author_name}' AND a1 <> a2
                WITH DISTINCT a2, p
                MATCH (p)-[:HAS_MESH_TERM]->(m:MeshTerm)
                WHERE {topic_conditions}
                RETURN DISTINCT a2.name as collaborator,
                       COUNT(DISTINCT p) as papers,
                       COLLECT(DISTINCT m.term)[0..3] as sample_topics
                ORDER BY papers DESC
                LIMIT 10
            """
        return self.query(cypher)

    def get_collaborating_institutions(self, min_collaborations: int = 2) -> list[dict[str, Any]]:
        """
        Get institutions that collaborate frequently.
        """
        cypher = f"""
            MATCH (i1:Institution)<-[:AFFILIATED_WITH]-(a1:Author)-[:WROTE]->(p:Paper)
                  <-[:WROTE]-(a2:Author)-[:AFFILIATED_WITH]->(i2:Institution)
            WHERE i1.name < i2.name
            WITH i1, i2, COUNT(DISTINCT p) as collaborations
            WHERE collaborations >= {min_collaborations}
            RETURN i1.name as institution1, i2.name as institution2, collaborations
            ORDER BY collaborations DESC
        """
        return self.query(cypher)

    def get_related_papers_by_mesh(self, pmid: str) -> list[dict[str, Any]]:
        """
        Get papers related by MeSH terms to a given PMID.
        """
        cypher = f"""
            MATCH (p1:Paper {{pmid: '{pmid}'}})-[:HAS_MESH_TERM]->(m:MeshTerm)
                  <-[:HAS_MESH_TERM]-(p2:Paper)
            WHERE p1 <> p2
            WITH p2, COUNT(DISTINCT m) as shared_terms
            RETURN p2.pmid as pmid, p2.title as title, shared_terms
            ORDER BY shared_terms DESC
            LIMIT 10
        """
        return self.query(cypher)

    def get_genes_in_same_papers(
        self, target_gene: str, mesh_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Find genes co-mentioned in the same papers as the target gene.
        Optionally filter by MeSH term substring (e.g., 'cancer', 'HIV').

        Examples:
            - "Which genes are mentioned in the same papers as gag?"
            - "Which genes co-occur with CCR5 in HIV-related papers?"
        """
        cypher = """
            MATCH (g:Gene)
            WHERE toLower(g.name) CONTAINS toLower($target_gene)
            OR toLower(g.aliases) CONTAINS toLower($target_gene)
            MATCH (g)-[:MENTIONED_IN]->(p:Paper)

            // Optional MeSH filter
            OPTIONAL MATCH (p)-[:HAS_MESH_TERM]->(m:MeshTerm)
            WHERE $mesh_filter IS NULL OR toLower(m.term) CONTAINS toLower($mesh_filter)

            MATCH (p)<-[:MENTIONED_IN]-(g2:Gene)
            WHERE g2 <> g
            RETURN g2.name AS gene,
                COUNT(DISTINCT p) AS shared_papers,
                COLLECT(DISTINCT p.pmid)[..5] AS example_pmids
            ORDER BY shared_papers DESC
            LIMIT 10
        """
        return self.query(cypher, {"target_gene": target_gene, "mesh_filter": mesh_filter})
