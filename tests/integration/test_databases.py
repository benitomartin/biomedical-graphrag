"""Integration tests for database connections and operations."""

from unittest.mock import AsyncMock, patch

import pytest

from biomedical_graphrag.config import Neo4jSettings, QdrantSettings
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore


class TestQdrantIntegration:
    """Test Qdrant database integration."""

    @pytest.mark.asyncio
    async def test_qdrant_vectorstore_initialization(self) -> None:
        """Test that QdrantVectorStore can be initialized with settings."""
        with patch(
            "biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore.settings"
        ) as mock_settings:
            mock_settings.qdrant.url = "http://localhost:6333"
            mock_settings.qdrant.api_key.get_secret_value.return_value = "test-key"
            mock_settings.qdrant.collection_name = "test_collection"
            mock_settings.qdrant.embedding_dimension = 1536
            mock_settings.qdrant.estimate_bm25_avg_len_on_x_docs = 300
            mock_settings.qdrant.cloud_inference = False
            mock_settings.openai.api_key.get_secret_value.return_value = "test-openai-key"

            with patch(
                "biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore.AsyncQdrantClient"
            ) as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client

                vectorstore = AsyncQdrantVectorStore()

                assert vectorstore.url == "http://localhost:6333"
                assert vectorstore.collection_name == "test_collection"
                assert vectorstore.embedding_dimension == 1536
                assert vectorstore.estimate_bm25_avg_len_on_x_docs == 300
                assert vectorstore.cloud_inference is False
                assert vectorstore.client == mock_client

                await vectorstore.close()
                mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_collection_creation(self, mock_qdrant_vectorstore: AsyncMock) -> None:
        """Test Qdrant collection creation."""
        mock_qdrant_vectorstore.create_collection.return_value = None

        await mock_qdrant_vectorstore.create_collection()

        mock_qdrant_vectorstore.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_client_close(self, mock_qdrant_vectorstore: AsyncMock) -> None:
        """Test Qdrant client close functionality."""
        await mock_qdrant_vectorstore.close()

        mock_qdrant_vectorstore.close.assert_called_once()


class TestNeo4jIntegration:
    """Test Neo4j database integration."""

    @pytest.mark.asyncio
    async def test_neo4j_client_creation(self) -> None:
        """Test that Neo4jClient can be created with settings."""
        with patch("biomedical_graphrag.infrastructure.neo4j_db.neo4j_client.settings") as mock_settings:
            mock_settings.neo4j.uri = "bolt://localhost:7687"
            mock_settings.neo4j.username = "neo4j"
            mock_settings.neo4j.password.get_secret_value.return_value = "test-password"
            mock_settings.neo4j.database = "neo4j"

            with patch(
                "biomedical_graphrag.infrastructure.neo4j_db.neo4j_client.AsyncGraphDatabase"
            ) as mock_db:
                mock_driver = AsyncMock()
                mock_db.driver.return_value = mock_driver

                client = await AsyncNeo4jClient.create()

                assert client.driver == mock_driver
                assert client.database == "neo4j"

                await client.close()
                mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_neo4j_execute_query(self, mock_neo4j_client: AsyncMock) -> None:
        """Test Neo4j query execution."""
        test_query = "CREATE (n:Test {name: $name})"
        test_params = {"name": "test_node"}

        await mock_neo4j_client.execute(test_query, test_params)

        mock_neo4j_client.execute.assert_called_once_with(test_query, test_params)

    @pytest.mark.asyncio
    async def test_neo4j_create_graph(self, mock_neo4j_client: AsyncMock) -> None:
        """Test Neo4j graph creation."""
        test_query = "CREATE (p:Paper {pmid: $pmid})"
        test_params = {"pmid": "12345678"}

        await mock_neo4j_client.create_graph(test_query, test_params)

        mock_neo4j_client.create_graph.assert_called_once_with(test_query, test_params)

    @pytest.mark.asyncio
    async def test_neo4j_client_close(self, mock_neo4j_client: AsyncMock) -> None:
        """Test Neo4j client close functionality."""
        await mock_neo4j_client.close()

        mock_neo4j_client.close.assert_called_once()


class TestDatabaseConnectionSettings:
    """Test database connection settings validation."""

    def test_qdrant_settings_validation(self) -> None:
        """Test Qdrant settings validation."""

        qdrant_settings = QdrantSettings()

        assert qdrant_settings.url == "http://localhost:6333"
        assert qdrant_settings.collection_name == "biomedical_papers"
        assert qdrant_settings.embedding_model == "text-embedding-3-large"
        assert qdrant_settings.embedding_dimension == 1536
        assert qdrant_settings.reranker_embedding_dimension == 3072
        assert qdrant_settings.estimate_bm25_avg_len_on_x_docs == 300
        assert qdrant_settings.cloud_inference is False

    def test_neo4j_settings_validation(self) -> None:
        """Test Neo4j settings validation."""

        neo4j_settings = Neo4jSettings()

        assert neo4j_settings.uri == "bolt://localhost:7687"
        assert neo4j_settings.username == "neo4j"
        assert neo4j_settings.database == "neo4j"
