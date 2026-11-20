"""CPU-only, bounded in-memory cache over a persistent embedding cache."""

import math
from collections import OrderedDict
from threading import Lock

from onecredit.config import MODEL_SPECS, Settings

SUPPORTED_MODELS = set(MODEL_SPECS)


def validate_vector(vector, dimensions: int = 384) -> list[float]:
    if hasattr(vector, "to_list"):
        vector = vector.to_list()
    result = [float(x) for x in vector]
    if len(result) != dimensions or not all(math.isfinite(x) for x in result):
        raise ValueError(f"embedding must contain {dimensions} finite values")
    norm = math.sqrt(sum(x * x for x in result))
    if norm < 1e-12:
        raise ValueError("zero embedding is not searchable")
    return [x / norm for x in result]


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = OrderedDict()
        self.lock = Lock()
        self.calls = 0
        if settings.model.startswith("microsoft/harrier"):
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoTokenizer

            torch.set_num_threads(2)
            source, revision, _ = MODEL_SPECS[settings.model]
            snapshot = snapshot_download(
                source,
                revision=revision,
                cache_dir=f"{settings.model_cache}/hf",
                token=False,
                allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
            )
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, padding_side="left")
            self.model = (
                AutoModel.from_pretrained(
                    snapshot, trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa"
                )
                .to("cpu")
                .eval()
            )
            self.prompt = "Instruct: Retrieve semantically similar text\nQuery: "
            return
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        from huggingface_hub import snapshot_download

        if settings.model not in SUPPORTED_MODELS:
            raise ValueError("unsupported embedding model")
        if settings.model == "intfloat/multilingual-e5-small":
            if settings.model not in {x["model"] for x in TextEmbedding.list_supported_models()}:
                TextEmbedding.add_custom_model(
                    model=settings.model,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=settings.model),
                    dim=384,
                    model_file="onnx/model.onnx",
                )
        source, revision, model_file = MODEL_SPECS[settings.model]
        snapshot = snapshot_download(
            source,
            revision=revision,
            cache_dir=f"{settings.model_cache}/hf",
            token=False,
            allow_patterns=[
                model_file,
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
            ],
        )
        self.model = TextEmbedding(
            model_name=settings.model,
            cache_dir=settings.model_cache,
            specific_model_path=snapshot,
            threads=2,
            providers=["CPUExecutionProvider"],
        )

    def embed(self, text: str) -> list[float]:
        # A single inference lane bounds CPU/memory and prevents duplicate cache misses.
        with self.lock:
            if text in self.cache:
                self.cache.move_to_end(text)
                return self.cache[text]
            if self.settings.model.startswith("microsoft/harrier"):
                import torch

                batch = self.tokenizer(
                    [self.prompt + text], padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                with torch.inference_mode():
                    hidden = self.model(**batch).last_hidden_state
                    # Left padding places the last real token at the final position.
                    vector = hidden[:, -1, :].squeeze(0).float().cpu().tolist()
                result = validate_vector(vector, self.settings.embedding_dimensions)
            else:
                model_input = f"query: {text}" if "multilingual-e5" in self.settings.model else text
                result = validate_vector(
                    next(iter(self.model.embed([model_input]))), self.settings.embedding_dimensions
                )
            self.calls += 1
            self.cache[text] = result
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
            return result
