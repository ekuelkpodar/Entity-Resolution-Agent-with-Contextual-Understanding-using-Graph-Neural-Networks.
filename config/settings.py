"""
Configuration management for Entity Resolution Agent
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application Configuration
    app_name: str = Field(default="entity-resolution-agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="entity_resolution", alias="NEO4J_DATABASE")

    # Redis Configuration
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    cache_ttl: int = Field(default=86400, alias="CACHE_TTL")

    # LLM Configuration
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    default_llm_model: str = Field(default="claude-sonnet-4-5-20250929", alias="DEFAULT_LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4000, alias="LLM_MAX_TOKENS")

    # Vector Database Configuration
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp", alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="entity-embeddings", alias="PINECONE_INDEX_NAME")

    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, alias="WEAVIATE_API_KEY")

    chroma_persist_directory: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIRECTORY")

    # Database Configuration
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="entity_resolution", alias="MONGODB_DATABASE")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="entity_resolution", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="password", alias="POSTGRES_PASSWORD")

    # GNN Model Configuration
    gnn_hidden_dim: int = Field(default=256, alias="GNN_HIDDEN_DIM")
    gnn_embedding_dim: int = Field(default=128, alias="GNN_EMBEDDING_DIM")
    gnn_num_layers: int = Field(default=3, alias="GNN_NUM_LAYERS")
    gnn_dropout: float = Field(default=0.2, alias="GNN_DROPOUT")
    gnn_learning_rate: float = Field(default=0.001, alias="GNN_LEARNING_RATE")
    gnn_batch_size: int = Field(default=32, alias="GNN_BATCH_SIZE")
    gnn_num_epochs: int = Field(default=100, alias="GNN_NUM_EPOCHS")

    # Entity Resolution Thresholds
    definite_match_threshold: float = Field(default=0.85, alias="DEFINITE_MATCH_THRESHOLD")
    probable_match_threshold: float = Field(default=0.70, alias="PROBABLE_MATCH_THRESHOLD")
    possible_match_threshold: float = Field(default=0.50, alias="POSSIBLE_MATCH_THRESHOLD")

    # Signal Fusion Weights
    weight_name_similarity: float = Field(default=0.25, alias="WEIGHT_NAME_SIMILARITY")
    weight_address_similarity: float = Field(default=0.15, alias="WEIGHT_ADDRESS_SIMILARITY")
    weight_identifier_match: float = Field(default=0.20, alias="WEIGHT_IDENTIFIER_MATCH")
    weight_gnn_embedding: float = Field(default=0.25, alias="WEIGHT_GNN_EMBEDDING")
    weight_shared_directors: float = Field(default=0.05, alias="WEIGHT_SHARED_DIRECTORS")
    weight_shared_addresses: float = Field(default=0.05, alias="WEIGHT_SHARED_ADDRESSES")
    weight_llm_confidence: float = Field(default=0.05, alias="WEIGHT_LLM_CONFIDENCE")

    # Beneficial Ownership Configuration
    ubo_ownership_threshold: float = Field(default=25.0, alias="UBO_OWNERSHIP_THRESHOLD")
    ubo_max_depth: int = Field(default=10, alias="UBO_MAX_DEPTH")
    nominee_relationship_threshold: int = Field(default=5, alias="NOMINEE_RELATIONSHIP_THRESHOLD")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=4, alias="API_WORKERS")
    api_reload: bool = Field(default=True, alias="API_RELOAD")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="CORS_ORIGINS"
    )

    # Monitoring
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")

    # Feature Flags
    enable_gnn_matching: bool = Field(default=True, alias="ENABLE_GNN_MATCHING")
    enable_llm_reasoning: bool = Field(default=True, alias="ENABLE_LLM_REASONING")
    enable_caching: bool = Field(default=True, alias="ENABLE_CACHING")
    enable_async_processing: bool = Field(default=True, alias="ENABLE_ASYNC_PROCESSING")

    # Data Sources
    companies_house_api_key: Optional[str] = Field(default=None, alias="COMPANIES_HOUSE_API_KEY")
    sec_edgar_user_agent: Optional[str] = Field(default=None, alias="SEC_EDGAR_USER_AGENT")
    ofac_data_url: str = Field(
        default="https://www.treasury.gov/ofac/downloads/sdn.xml",
        alias="OFAC_DATA_URL"
    )

    # Spark Configuration
    spark_master: str = Field(default="local[*]", alias="SPARK_MASTER")
    spark_driver_memory: str = Field(default="4g", alias="SPARK_DRIVER_MEMORY")
    spark_executor_memory: str = Field(default="4g", alias="SPARK_EXECUTOR_MEMORY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Export for convenience
settings = get_settings()
