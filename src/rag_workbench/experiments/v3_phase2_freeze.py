# ruff: noqa: E501
"""Frozen Experiment-1 identities. Do not retune after the first hosted result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    evidence_sufficiency_schema_identity,
    evidence_sufficiency_template_hash,
)
from rag_workbench.db.models import Chunk, Document
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.v2_final_benchmark import (
    DATASET_HASH as V2_FINAL_DATASET_HASH,
)
from rag_workbench.experiments.v2_final_benchmark import (
    DATASET_ID as V2_FINAL_DATASET_ID,
)
from rag_workbench.experiments.v2_final_benchmark import v2_architecture_configuration
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    stable_hash,
    verify_v1_file_identities,
)
from rag_workbench.experiments.v2_sufficiency_fn import SCHEMA_IDENTITY, SOL_MODEL
from rag_workbench.experiments.v3_generate_verify import verify_persisted_v2
from rag_workbench.experiments.v3_phase2_safety_cases import DATASET_ID, DATASET_PATH
from rag_workbench.ingestion.corpus_roots import (
    V3_RESEARCH_CORPUS_VERSION,
    collect_corpus_paths,
)
from rag_workbench.recovery.contracts import (
    CLAIM_VERIFIER_PROMPT_VERSION,
    RECOVERY_DRAFT_PROMPT_VERSION,
    recovery_draft_schema_identity,
    recovery_draft_template_hash,
    recovery_verifier_schema_identity,
    recovery_verifier_template_hash,
)
from rag_workbench.recovery.instruction_boundary import (
    BOUNDARY_VERSION,
    boundary_template_hash,
)

FROZEN_DATASET_HASH = "9a1fe2a4072c99e2ab7f483f739695ab373abcf3d485da98d005f3a668544f00"
FROZEN_V2_SEMANTIC_INDEX = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
FROZEN_V3_INDEX_IDENTITY = "2027d684f92af14a7dc6cd2362dda5d60233287e746f4aa856fb3c82fefe160b"
FROZEN_V3_CORPUS_HASH = "f646ffa12213277ae3bd2b167edbe2d5930ddfed96ce478a890154f54b2c8cc9"
FROZEN_DRAFT_PROMPT_HASH = "16bb25074a2915b95d629aaeb0e5139d466a95c71ad8fd63b4829f5771ea927c"
FROZEN_VERIFIER_PROMPT_HASH = "a81cdd3f9e572986cf8dca55344180c49359299272c9b1b566d31efeabfe2bbe"
FROZEN_DRAFT_SCHEMA_HASH = "46aecca613005a9476035468dfa7c342ff0fd77ae615f8500d1b43025e25d1b8"
FROZEN_VERIFIER_SCHEMA_HASH = "ad20ec05a4914b8787ee1003b4789251c451b2c90b1dc46e15f6c899c0163c05"
FROZEN_JUDGE_PROMPT_HASH = "d49994bc7a429e2cbbd07935a2ed4cbb5503098cd01cdd146e6417be55dc7d83"
FROZEN_JUDGE_SCHEMA_HASH = "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
FROZEN_BOUNDARY_HASH = "7eaa50e05ca3dbef872d7aa812b473384830da6dcd084ca5d91c258d122596f2"
FROZEN_EXP1_CONFIGURATION_HASH = "8f8ec19a195ef88ad15fd0699664984d4568f54336baf99320541237c8bf0ef9"
PUBLIC_BOUNDARY_NAME = "evidence-instruction-boundary-v1"
COST_CAP_USD = 1.0
AUTHORIZED_NEW_LOGICAL_SOL = 180
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER = "openai-compatible"
EMBEDDING_VERSION = "1"
EMBEDDING_DIMENSION = 64
EMBEDDING_USD_PER_MILLION = 0.02


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


FREEZE_DIR = _repo_root() / "data" / "experiments" / "v3-phase2-exp1"
FREEZE_PATH = FREEZE_DIR / "freeze.json"
QUALIFIED_CANDIDATE_PATH = FREEZE_DIR / "qualified-candidate.json"


def v3_corpus_file_hash() -> str:
    digest = hashlib.sha256()
    for path in collect_corpus_paths(
        V3_RESEARCH_CORPUS_VERSION,
        _repo_root() / "data" / "synthetic_company",
        require_manifest_complete=True,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def v3_index_identity_payload() -> dict[str, Any]:
    return {
        "parent_embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_version": EMBEDDING_VERSION,
        "dimension": EMBEDDING_DIMENSION,
        "corpus_version": V3_RESEARCH_CORPUS_VERSION,
        "corpus_hash": v3_corpus_file_hash(),
        "chunk_size": 180,
        "chunk_overlap": 30,
        "v2_semantic_index_not_used": FROZEN_V2_SEMANTIC_INDEX,
    }


def v3_index_identity() -> str:
    return stable_hash(v3_index_identity_payload())


def experiment_1_freeze() -> dict[str, Any]:
    from rag_workbench.experiments.v3_phase2_safety import (
        SELECTION_POLICY,
        exp1_configuration,
        exp1_configuration_hash,
    )

    dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    if dataset_hash != FROZEN_DATASET_HASH:
        raise ValueError("frozen validation dataset hash changed")
    if SEMANTIC_INDEX_IDENTITY != FROZEN_V2_SEMANTIC_INDEX:
        raise ValueError("frozen V2 semantic index identity changed")
    freeze = {
        "experiment_id": "V3_P2_EXP1_UNTRUSTED_INSTRUCTION_BOUNDARY",
        "public_mechanism_name": PUBLIC_BOUNDARY_NAME,
        "implementation_version": BOUNDARY_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_hash": dataset_hash,
        "selection_policy": SELECTION_POLICY,
        "draft_prompt_version": RECOVERY_DRAFT_PROMPT_VERSION,
        "verifier_prompt_version": CLAIM_VERIFIER_PROMPT_VERSION,
        "draft_prompt_hash": recovery_draft_template_hash(),
        "verifier_prompt_hash": recovery_verifier_template_hash(),
        "draft_schema_identity": recovery_draft_schema_identity(),
        "verifier_schema_identity": recovery_verifier_schema_identity(),
        "primary_judge_prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
        "primary_judge_prompt_hash": evidence_sufficiency_template_hash(),
        "primary_judge_schema_identity": evidence_sufficiency_schema_identity(),
        "boundary_hash": boundary_template_hash(),
        "exp1_configuration_hash": exp1_configuration_hash(),
        "exp1_configuration": exp1_configuration(),
        "v3_corpus_version": V3_RESEARCH_CORPUS_VERSION,
        "v3_corpus_hash": v3_corpus_file_hash(),
        "v3_index_identity": v3_index_identity(),
        "v2_semantic_index_identity": FROZEN_V2_SEMANTIC_INDEX,
        "v2_corpus_identity": CORPUS_IDENTITY,
        "v1_architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
        "v2_architecture_hash": v2_architecture_configuration()["architecture_hash"],
        "v2_final_dataset_id": V2_FINAL_DATASET_ID,
        "v2_final_dataset_hash": V2_FINAL_DATASET_HASH,
        "sol_model": SOL_MODEL,
        "judge_schema_identity": SCHEMA_IDENTITY,
        "quality_retries": False,
        "retune_after_hosted_results": False,
    }
    expected = {
        "v3_index_identity": FROZEN_V3_INDEX_IDENTITY,
        "v3_corpus_hash": FROZEN_V3_CORPUS_HASH,
        "draft_prompt_hash": FROZEN_DRAFT_PROMPT_HASH,
        "verifier_prompt_hash": FROZEN_VERIFIER_PROMPT_HASH,
        "draft_schema_identity": FROZEN_DRAFT_SCHEMA_HASH,
        "verifier_schema_identity": FROZEN_VERIFIER_SCHEMA_HASH,
        "primary_judge_prompt_hash": FROZEN_JUDGE_PROMPT_HASH,
        "primary_judge_schema_identity": FROZEN_JUDGE_SCHEMA_HASH,
        "boundary_hash": FROZEN_BOUNDARY_HASH,
        "exp1_configuration_hash": FROZEN_EXP1_CONFIGURATION_HASH,
        "judge_schema_identity": SCHEMA_IDENTITY,
    }
    drifted = {key: freeze[key] for key in expected if freeze[key] != expected[key]}
    if drifted:
        raise ValueError(f"Experiment 1 freeze identities drifted: {sorted(drifted)}")
    if freeze["v3_index_identity"] == FROZEN_V2_SEMANTIC_INDEX:
        raise ValueError("V3 index identity must not equal the frozen V2 index")
    return freeze


def persist_experiment_1_freeze() -> dict[str, Any]:
    freeze = experiment_1_freeze()
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(freeze, indent=2, sort_keys=True, default=str) + "\n"
    if FREEZE_PATH.exists() and FREEZE_PATH.read_text(encoding="utf-8") != encoded:
        raise ValueError("Experiment 1 freeze artifact would change after freeze")
    if not FREEZE_PATH.exists():
        FREEZE_PATH.write_text(encoded, encoding="utf-8")
    return freeze


def persist_qualified_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if QUALIFIED_CANDIDATE_PATH.exists() and QUALIFIED_CANDIDATE_PATH.read_text(encoding="utf-8") != encoded:
        raise ValueError("qualified candidate freeze artifact would change")
    if not QUALIFIED_CANDIDATE_PATH.exists():
        QUALIFIED_CANDIDATE_PATH.write_text(encoded, encoding="utf-8")
    return payload


def verify_v2_index_not_mutated(session: Session) -> dict[str, Any]:
    v2_chunks = int(
        session.scalar(
            select(func.count()).select_from(Chunk).where(
                Chunk.index_identity == FROZEN_V2_SEMANTIC_INDEX
            )
        )
        or 0
    )
    leaked = session.execute(
        select(Document.document_id)
        .join(Chunk, Chunk.document_fk == Document.id)
        .where(
            Chunk.index_identity == FROZEN_V2_SEMANTIC_INDEX,
            Document.document_id.like("v3-research-%"),
        )
    ).scalars().all()
    if leaked:
        raise ValueError("V3 research documents were written into the frozen V2 index")
    return {
        "v2_semantic_index_identity": FROZEN_V2_SEMANTIC_INDEX,
        "v2_chunk_count": v2_chunks,
        "v3_documents_in_v2_index": 0,
    }


def verify_preservation(session: Session) -> dict[str, Any]:
    files = verify_v1_file_identities()
    index = verify_v2_index_not_mutated(session)
    persisted: dict[str, Any]
    try:
        persisted = {**verify_persisted_v2(session), "present": True}
    except ValueError as exc:
        if "missing" not in str(exc):
            raise
        persisted = {
            "present": False,
            "reason": str(exc),
            "v2_architecture_hash_from_files": v2_architecture_configuration()[
                "architecture_hash"
            ],
        }
    frozen = _repo_root() / "data" / "synthetic_company"
    extra = _repo_root() / "data" / "v3_research_corpus"
    isolation = {
        "frozen_company_markdown_files": len(list(frozen.glob("*.md"))),
        "v3_research_markdown_files": len(list(extra.glob("*.md"))),
        "v3_files_in_synthetic_company": [
            path.name for path in frozen.glob("v3-research-*.md")
        ],
    }
    if isolation["frozen_company_markdown_files"] != 16:
        raise ValueError("frozen V1/V2 company corpus file count changed")
    if isolation["v3_files_in_synthetic_company"]:
        raise ValueError("V3 research documents must not live in synthetic_company")
    return {
        "v1_files": files,
        "v2_index": index,
        "v2_persisted": persisted,
        "corpus_isolation": isolation,
        "v1_architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
        "v2_architecture_hash": v2_architecture_configuration()["architecture_hash"],
    }
