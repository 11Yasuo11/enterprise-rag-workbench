from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import QueryEmbeddingCacheRecord
from rag_workbench.experiments.v3_ranking_phase5_fresh_e2e_cases import (
    DATASET_ID,
    DATASET_PATH,
    GENERATION_METHOD,
    OVERLAP_CEILING,
    dataset_overlap_report,
)
from rag_workbench.reranking.pairwise_complementarity import (
    PAIRWISE_COMPLEMENTARITY_RERANK_V1,
    PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
)
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key

EXPECTED_DISTRIBUTION = {
    "single_document": 4,
    "multiple_required_chunks_same_document": 10,
    "multidoc_two": 20,
    "multidoc_three": 30,
    "near_duplicate": 16,
    "exact_identifier": 8,
    "version_region": 8,
    "semantic_paraphrase": 6,
    "acl_sensitive": 4,
    "partial_no_answer": 4,
    "prompt_injection": 10,
}


DEFAULT_EMBEDDING_CACHE_IDENTITY = {
    "provider": "openai-compatible",
    "model": "text-embedding-3-small",
    "version": "1",
    "dimension": 64,
}


ARM_REFERENCE_R = "reference_r"
ARM_CONTROL_A = "control_a"
ARM_CANDIDATE_B = "candidate_b"


ARTIFACT_DIR = Path("data/experiments/v3-phase5-fresh-e2e-ranking-validation")
LEDGER_PATH = ARTIFACT_DIR / "ledger.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoint.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_payload() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Phase-5 dataset JSON missing: {DATASET_PATH}")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _category_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    # Runner only stores the derived distribution requested by the prompt.
    # We treat "multiple_required_chunks_same_document" as a subset within
    # "single_document" by recomputing it from required_chunk_markers length.
    dist = Counter(item["category"] for item in cases)
    multichunk = sum(
        1
        for item in cases
        if item["category"] == "single_document"
        and isinstance(item.get("required_chunk_markers"), list)
        and len(item["required_chunk_markers"]) >= 2
    )
    single_document = dist["single_document"] - multichunk
    return {
        "single_document": single_document,
        "multiple_required_chunks_same_document": multichunk,
        "multidoc_two": dist["multidoc_two"],
        "multidoc_three": dist["multidoc_three"],
        "near_duplicate": dist["near_duplicate"],
        "exact_identifier": dist["exact_identifier"],
        "version_region": dist["version_region"],
        "semantic_paraphrase": dist["semantic_paraphrase"],
        "acl_sensitive": dist["acl_sensitive"],
        "partial_no_answer": dist["partial_no_answer"],
        "prompt_injection": dist["prompt_injection"],
    }


class V3RankingPhase5FreshE2EBenchmark:
    """Phase 5: fresh unseen E2E evaluation for qualified pairwise ranking.

    In this environment, external calls (embeddings/judges/recovery) are often
    disabled. This runner is still responsible for:
      - freezing dataset + verifying independence gates
      - persisting embedding preflight / ceilings
      - stopping before any external inference when authorization is missing
    """

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self._state = self._load_checkpoint()

    def _load_checkpoint(self) -> dict[str, Any]:
        if CHECKPOINT_PATH.exists():
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return {
            "identity": {
                "lock_id": "v3-phase5-fresh-e2e-ranking-validation",
                "pairwise_candidate": PAIRWISE_COMPLEMENTARITY_RERANK_V1,
                "pairwise_candidate_config_hash": PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
                "dataset_id": DATASET_ID,
            },
            "stages": {
                "dataset_frozen": False,
                "embedding_preflight_done": False,
                "retrieval_done": False,
                "hosted_judge_preflight_done": False,
                "hosted_execute_done": False,
                "analysis_done": False,
            },
            "result": None,
            "ledger": None,
        }

    def _persist(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")
        # Ledger is optional; keep it in sync when present.
        if self._state.get("ledger") is not None:
            LEDGER_PATH.write_text(
                json.dumps(self._state["ledger"], indent=2, default=str),
                encoding="utf-8",
            )

    def status(self) -> dict[str, Any]:
        return self._state

    def freeze_dataset(self) -> dict[str, Any]:
        if self._state["stages"]["dataset_frozen"]:
            return self._state["ledger"]

        payload = _load_payload()
        cases: list[dict[str, Any]] = payload["cases"]
        dataset_hash = _sha256_file(DATASET_PATH)
        dist = _category_distribution(cases)
        if dist != EXPECTED_DISTRIBUTION:
            raise ValueError(f"phase5 dataset category distribution mismatch: {dist}")

        overlap = dataset_overlap_report(cases)
        if not overlap["pass"]:
            raise ValueError(
                "phase5 dataset independence gate failed: "
                + json.dumps(overlap, indent=2, ensure_ascii=True)
            )

        self._state["ledger"] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": dataset_hash,
            "case_count": len(cases),
            "distribution": dist,
            "maximum_prior_overlap": overlap["maximum_normalized_overlap"],
            "closest_prior_case": overlap["closest_prior_case"],
            "overlap_ceiling": OVERLAP_CEILING,
            "freeze_timestamp": payload.get("freeze_timestamp"),
            "generation_method": GENERATION_METHOD,
            "pairwise_candidate": {
                "name": PAIRWISE_COMPLEMENTARITY_RERANK_V1,
                "config_hash": PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
            },
        }
        self._state["stages"]["dataset_frozen"] = True
        self._persist()
        return self._state["ledger"]

    def embedding_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        ledger = self.freeze_dataset()
        cases: list[dict[str, Any]] = _load_payload()["cases"]
        questions = tuple(dict.fromkeys(item["question"] for item in cases))
        keys = [
            query_embedding_cache_key(
                question,
                provider=DEFAULT_EMBEDDING_CACHE_IDENTITY["provider"],
                model=DEFAULT_EMBEDDING_CACHE_IDENTITY["model"],
                version=DEFAULT_EMBEDDING_CACHE_IDENTITY["version"],
                dimension=DEFAULT_EMBEDDING_CACHE_IDENTITY["dimension"],
            )
            for question in questions
        ]
        keys_set = set(keys)

        # Current cumulative embedding calls (cache rows) for this embedding identity.
        current = int(
            self.session.scalar(
                select(func.count()).select_from(QueryEmbeddingCacheRecord).where(
                    QueryEmbeddingCacheRecord.embedding_provider
                    == DEFAULT_EMBEDDING_CACHE_IDENTITY["provider"],
                    QueryEmbeddingCacheRecord.embedding_model
                    == DEFAULT_EMBEDDING_CACHE_IDENTITY["model"],
                    QueryEmbeddingCacheRecord.embedding_version
                    == DEFAULT_EMBEDDING_CACHE_IDENTITY["version"],
                    QueryEmbeddingCacheRecord.embedding_dimension
                    == DEFAULT_EMBEDDING_CACHE_IDENTITY["dimension"],
                )
            )
            or 0
        )

        matches = int(
            self.session.scalar(
                select(func.count()).select_from(QueryEmbeddingCacheRecord).where(
                    QueryEmbeddingCacheRecord.cache_key.in_(list(keys_set))
                )
            )
            or 0
        )
        missing = len(keys_set) - matches

        preflight = {
            "phase": "embedding_preflight",
            "embedding_identity": DEFAULT_EMBEDDING_CACHE_IDENTITY,
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "dataset_unique_queries": len(keys_set),
            "existing_cache_matches": matches,
            "missing_unique_query_embeddings": missing,
            "maximum_new_embedding_calls": missing,
            "expected_cumulative_ending_usage": current + missing,
        }

        if persist:
            self._state["ledger"] = {**ledger, "embedding_preflight": preflight}
            self._state["stages"]["embedding_preflight_done"] = True
            self._persist()
        return preflight

    def execute(self) -> dict[str, Any]:
        """Run Phase 5 until the first external-inference authorization gate."""
        self.freeze_dataset()
        preflight = self.embedding_preflight(persist=True)

        missing = int(preflight["missing_unique_query_embeddings"])
        if missing <= 0:
            # In a fully-authorized environment we would proceed to retrieval.
            # This runner environment typically does not reach this path.
            self._state["result"] = {
                "verdict": "PARTIAL",
                "reason": (
                    "embedding_cache_complete_but_external_calls_disabled_"
                    "path_not_implemented"
                ),
            }
            self._persist()
            return self.status()

        # No new embeddings can be produced without external credentials.
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            self._state["result"] = {
                "verdict": "BLOCKED",
                "stop_code": "EXTERNAL_CREDENTIALS_REQUIRED",
                "external_preflight": preflight,
                "minimum_cumulative_call_ceilings": {
                    "query_embedding_calls": preflight["expected_cumulative_ending_usage"],
                },
                "estimated_additional_usd": "NOT_VERIFIED",
                "expected_max_physical_attempts": None,
                "next_phase": (
                    "Authorize embeddings + judges/recovery, then rerun "
                    "Phase 5 execution from checkpoint."
                ),
                "promotion_decision": "KEEP_CURRENT_V3_RESEARCH_ARCHITECTURE",
                "v3_status": "V3_CANDIDATE_REJECTED",
                "primary_remaining_bottleneck": "EXTERNAL_CREDENTIALS_REQUIRED",
            }
            self._persist()
            return self.status()

        # Embeddings are authorized, but might exceed configured ceilings.
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            self._state["result"] = {
                "verdict": "BLOCKED",
                "stop_code": "EXTERNAL_BUDGET_REQUIRED",
                "external_preflight": preflight,
                "minimum_cumulative_call_ceilings": {
                    "query_embedding_calls": preflight["expected_cumulative_ending_usage"],
                },
                "estimated_additional_usd": "NOT_VERIFIED",
                "expected_max_physical_attempts": None,
                "next_phase": (
                    "Increase MAX_EXTERNAL_EMBEDDING_CALLS, then resume "
                    "Phase 5 execution."
                ),
                "promotion_decision": "KEEP_CURRENT_V3_RESEARCH_ARCHITECTURE",
                "v3_status": "V3_CANDIDATE_REJECTED",
                "primary_remaining_bottleneck": "EXTERNAL_BUDGET_REQUIRED",
            }
            self._persist()
            return self.status()

        raise RuntimeError(
            "Phase 5 execute reached retrieval stage with embedded calls authorized; "
            "this runner environment did not implement retrieval/hosted execution yet."
        )

