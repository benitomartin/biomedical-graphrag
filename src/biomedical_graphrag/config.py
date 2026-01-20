import os
from typing import ClassVar

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class OpenAISettings(BaseModel):
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for OpenAI")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use for queries")
    temperature: float = Field(
        default=0.0, description="LLM temperature for OpenAI queries (0 for consistency)"
    )
    max_tokens: int = Field(default=1500, description="Maximum number of tokens for OpenAI queries")


class Neo4jSettings(BaseModel):
    uri: str = Field(default="bolt://localhost:7687", description="URI for Neo4j database")
    username: str = Field(default="neo4j", description="Username for Neo4j database")
    password: SecretStr = Field(default=SecretStr(""), description="Password for Neo4j database")
    database: str = Field(default="neo4j", description="Database name for Neo4j database")


class QdrantSettings(BaseModel):
    url: str = Field(default="http://localhost:6333", description="URL for Qdrant instance")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for Qdrant instance")
    collection_name: str = Field(
        default="biomedical_papers", description="Collection name for Qdrant instance"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large", description="OpenAI embedding model to use"
    )
    embedding_dimension: int = Field(
        default=1536, description="Dimension of the OpenAI embedding for retrieval (using OpenAI's MRL)"
    )
    reranker_embedding_dimension: int = Field(
        default=3072,
        description="Dimension of the OpenAI embedding for reranking (using OpenAI's MRL)",
    )
    estimate_bm25_avg_len_on_x_docs: int = Field(
        default=300,
        description="Number of documents to sample for estimating average length of documents, used in BM25 formula",
    )
    cloud_inference: bool = Field(
        default=False, description="Use Qdrant Cloud Inference for embeddings inference"
    )


class PubMedSettings(BaseModel):
    email: SecretStr = Field(default=SecretStr(""), description="Email for PubMed API")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for PubMed API")


class JsonDataSettings(BaseModel):
    pubmed_json_path: str = Field(
        default="data/pubmed_dataset.json", description="Path to the PubMed JSON dataset"
    )
    gene_json_path: str = Field(
        default="data/gene_dataset.json", description="Path to the Gene JSON dataset"
    )


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Use .env file or environment variables to configure.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    openai: OpenAISettings = OpenAISettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    qdrant: QdrantSettings = QdrantSettings()
    pubmed: PubMedSettings = PubMedSettings()
    json_data: JsonDataSettings = JsonDataSettings()

    @model_validator(mode="after")
    def validate_json_path(self) -> "Settings":
        """Validate that the JSON data path exists."""
        if not os.path.isfile(self.json_data.pubmed_json_path):
            logger.warning(
                f"PubMed JSON file not found at {self.json_data.pubmed_json_path}. "
                "Proceeding without it."
            )
        return self

    @model_validator(mode="after")
    def validate_gene_json_path(self) -> "Settings":
        """Validate that the Gene JSON data path exists."""
        if not os.path.isfile(self.json_data.gene_json_path):
            logger.warning(
                f"Gene JSON file not found at {self.json_data.gene_json_path}. Proceeding without it."
            )
        return self


settings = Settings()
