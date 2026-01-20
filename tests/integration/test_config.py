"""Integration tests for configuration and settings."""

import os
from unittest.mock import patch

from biomedical_graphrag.config import Neo4jSettings, OpenAISettings, QdrantSettings, Settings


def test_settings_creation_with_defaults() -> None:
    """Test that Settings can be created with default values."""
    # Test individual settings classes with defaults
    openai_settings = OpenAISettings()
    assert openai_settings.model == "gpt-4o-mini"
    assert openai_settings.temperature == 0.0
    assert openai_settings.max_tokens == 1500

    neo4j_settings = Neo4jSettings()
    assert neo4j_settings.uri == "bolt://localhost:7687"
    assert neo4j_settings.username == "neo4j"
    assert neo4j_settings.database == "neo4j"

    qdrant_settings = QdrantSettings()
    assert qdrant_settings.url == "http://localhost:6333"
    assert qdrant_settings.collection_name == "biomedical_papers"
    assert qdrant_settings.embedding_model == "text-embedding-3-large"
    assert qdrant_settings.embedding_dimension == 1536
    assert qdrant_settings.reranker_embedding_dimension == 3072
    assert qdrant_settings.estimate_bm25_avg_len_on_x_docs == 300
    assert qdrant_settings.cloud_inference is False


def test_settings_with_environment_variables() -> None:
    """Test that Settings can be configured via environment variables."""
    with patch.dict(
        os.environ,
        {
            "OPENAI__API_KEY": "test-api-key",
            "OPENAI__MODEL": "gpt-4",
            "NEO4J__URI": "bolt://test:7687",
            "QDRANT__URL": "http://test:6333",
            "QDRANT__CLOUD_INFERENCE": "true",
        },
    ):
        settings = Settings()

        assert settings.openai.api_key.get_secret_value() == "test-api-key"
        assert settings.openai.model == "gpt-4"
        assert settings.neo4j.uri == "bolt://test:7687"
        assert settings.qdrant.url == "http://test:6333"
        assert settings.qdrant.cloud_inference is True


def test_openai_settings_validation() -> None:
    """Test OpenAI settings validation and defaults."""
    openai_settings = OpenAISettings()

    assert openai_settings.model == "gpt-4o-mini"
    assert openai_settings.temperature == 0.0
    assert openai_settings.max_tokens == 1500
    assert openai_settings.api_key.get_secret_value() == ""


def test_neo4j_settings_validation() -> None:
    """Test Neo4j settings validation and defaults."""
    neo4j_settings = Neo4jSettings()

    assert neo4j_settings.uri == "bolt://localhost:7687"
    assert neo4j_settings.username == "neo4j"
    assert neo4j_settings.database == "neo4j"
    assert neo4j_settings.password.get_secret_value() == ""


def test_qdrant_settings_validation() -> None:
    """Test Qdrant settings validation and defaults."""
    qdrant_settings = QdrantSettings()

    assert qdrant_settings.url == "http://localhost:6333"
    assert qdrant_settings.collection_name == "biomedical_papers"
    assert qdrant_settings.embedding_model == "text-embedding-3-large"
    assert qdrant_settings.embedding_dimension == 1536
    assert qdrant_settings.reranker_embedding_dimension == 3072
    assert qdrant_settings.estimate_bm25_avg_len_on_x_docs == 300
    assert qdrant_settings.cloud_inference is False
    assert qdrant_settings.api_key.get_secret_value() == ""
