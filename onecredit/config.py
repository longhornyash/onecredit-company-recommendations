"""Environment configuration shared by the worker, API and operational commands."""

import hashlib
import os
from dataclasses import dataclass, field

MODEL_SPECS = {
    "microsoft/harrier-oss-v1-0.6b": (
        "microsoft/harrier-oss-v1-0.6b",
        "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        "model.safetensors",
    ),
    "intfloat/multilingual-e5-small": (
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "onnx/model.onnx",
    ),
    "BAAI/bge-small-en-v1.5": (
        "Qdrant/bge-small-en-v1.5-onnx-Q",
        "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        "model_optimized.onnx",
    ),
}


@dataclass(frozen=True)
class Settings:
    database_schema: str = field(default_factory=lambda: os.getenv("DATABASE_SCHEMA", "public"))
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"postgresql://postgres:postgres@localhost:{os.getenv('POSTGRES_PORT', '5432')}/companies"
        )
    )
    rabbitmq_url: str = field(
        default_factory=lambda: os.getenv(
            "RABBITMQ_URL", f"amqp://guest:guest@localhost:{os.getenv('RABBITMQ_PORT', '5672')}/"
        )
    )
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "microsoft/harrier-oss-v1-0.6b"))
    model_cache: str = field(default_factory=lambda: os.getenv("MODEL_CACHE", ".models"))
    statement_timeout_ms: int = 5000
    search_mode: str = field(default_factory=lambda: os.getenv("SEARCH_MODE", "auto"))

    @property
    def embedding_key(self) -> str:
        mode = "sts-last-token-cpu-f32-512-v1" if self.model.startswith("microsoft/harrier") else "symmetric-v1"
        return f"{self.model}@{MODEL_SPECS[self.model][1]}:nfkc-whitespace-v1:{mode}"

    @property
    def embedding_dimensions(self) -> int:
        return 1024 if self.model.startswith("microsoft/harrier") else 384

    @property
    def search_index_name(self):
        return "company_hnsw_" + hashlib.sha256(self.embedding_key.encode()).hexdigest()[:12]
