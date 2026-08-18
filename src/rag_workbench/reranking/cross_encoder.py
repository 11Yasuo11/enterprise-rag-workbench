from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.retrieval.vector_search import RetrievalResult

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    """Local Sentence Transformers adapter with deterministic rank ties."""

    model_id = MODEL_ID
    backend = "sentence-transformers/CrossEncoder (PyTorch)"
    external_calls = 0

    def __init__(
        self,
        *,
        device: str = "cpu",
        model: Any | None = None,
        model_path: str | None = None,
        resolved_revision: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.last_inference_latency_ms = 0.0
        self.last_sort_latency_ms = 0.0
        if model is None:
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder

            path = snapshot_download(
                repo_id=MODEL_ID,
                ignore_patterns=("onnx/*", "openvino/*", "*.onnx", "*.h5", "*.ot"),
            )
            model_path = path
            model = CrossEncoder(path, device=device)
        self._model = model
        self.model_path = model_path
        self.resolved_revision = resolved_revision or self._revision_from_path(model_path)

    @staticmethod
    def _revision_from_path(model_path: str | None) -> str:
        if not model_path:
            return "injected-test-model"
        path = Path(model_path)
        if path.parent.name == "snapshots":
            return path.name
        return "unknown"

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RerankedResult]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        inference_started = time.perf_counter()
        raw_scores: Sequence[float] = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        self.last_inference_latency_ms = (time.perf_counter() - inference_started) * 1000
        if len(raw_scores) != len(candidates):
            raise ValueError("cross-encoder returned an unexpected score count")
        scored = [
            (float(score), candidate)
            for score, candidate in zip(raw_scores, candidates, strict=True)
        ]
        sort_started = time.perf_counter()
        scored.sort(key=lambda item: (-item[0], item[1].rank, item[1].chunk_id))
        self.last_sort_latency_ms = (time.perf_counter() - sort_started) * 1000
        return [
            RerankedResult(
                result=candidate,
                original_dense_rank=candidate.rank,
                dense_score=candidate.dense_score or candidate.score,
                reranker_score=score,
                reranked_rank=rank,
            )
            for rank, (score, candidate) in enumerate(scored, start=1)
        ]
