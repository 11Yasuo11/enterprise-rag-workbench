"""Frozen, label-blind atomic requirement contract."""

from .contract import (
    AtomicRequirement,
    ContextQualifier,
    FrozenQuestionPlan,
    ValidatedMapping,
    ValidatedRequirement,
    assemble_frozen_plan,
    decompose_question,
    semantic_cardinality_audit,
    validate_verifier_result,
)
from .deterministic_support import (
    DeterministicSupportDecision,
    DirectSupportMapping,
    deterministic_support_complete,
    extract_direct_support_mappings,
)
from .evidence_mapping import (
    CanonicalEvidenceMapping,
    EvidenceMappingSchemaError,
    canonical_mappings_to_validation,
    normalize_evidence_mapping,
)
from .verifier import FrozenEvidenceVerifier, FrozenVerifierResult

__all__ = [
    "AtomicRequirement",
    "CanonicalEvidenceMapping",
    "ContextQualifier",
    "DeterministicSupportDecision",
    "DirectSupportMapping",
    "EvidenceMappingSchemaError",
    "FrozenEvidenceVerifier",
    "FrozenQuestionPlan",
    "FrozenVerifierResult",
    "ValidatedMapping",
    "ValidatedRequirement",
    "assemble_frozen_plan",
    "canonical_mappings_to_validation",
    "decompose_question",
    "deterministic_support_complete",
    "extract_direct_support_mappings",
    "normalize_evidence_mapping",
    "semantic_cardinality_audit",
    "validate_verifier_result",
]
