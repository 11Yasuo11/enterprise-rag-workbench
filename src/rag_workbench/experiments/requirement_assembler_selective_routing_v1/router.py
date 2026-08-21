"""Rule-based selective Luna/Sol router; no model or benchmark-label features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Route = Literal["SAFE_ABSTAIN", "DETERMINISTIC", "LUNA", "SOL"]


@dataclass(frozen=True)
class RoutingFeatures:
    security_precheck_requires_abstention: bool = False
    deterministic_support_complete: bool = False
    conflicting_evidence: bool = False
    version_sensitive: bool = False
    version_resolved: bool = False
    multiple_active_versions: bool = False
    version_metadata_complete: bool = True
    top15_evidence_complete: bool = False
    required_fact_count: int = 1


@dataclass(frozen=True)
class PostLunaFeatures:
    decision: Literal["GO", "ABSTAIN", "UNCERTAIN"]
    deterministic_validation_pass: bool = False
    deterministic_evidence_complete: bool = False
    genuine_required_evidence_absence: bool = False
    version_resolved: bool = False
    authorized_candidate_evidence_complete: bool = False


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    reason: str


class SelectiveRiskRouter:
    def initial(self, features: RoutingFeatures) -> RoutingDecision:
        if features.security_precheck_requires_abstention:
            return RoutingDecision("SAFE_ABSTAIN", "security_precheck")
        if features.deterministic_support_complete:
            return RoutingDecision("DETERMINISTIC", "complete_literal_support")
        if features.conflicting_evidence:
            return RoutingDecision("SOL", "conflicting_evidence")
        if features.multiple_active_versions and not features.version_resolved:
            return RoutingDecision("SOL", "multiple_active_versions")
        if features.version_sensitive and not features.version_resolved:
            return RoutingDecision("SOL", "unresolved_version")
        if features.version_sensitive and not features.version_metadata_complete:
            return RoutingDecision("SOL", "incomplete_version_metadata")
        return RoutingDecision("LUNA", "low_risk_semantic_verification")

    def after_luna(self, features: PostLunaFeatures) -> RoutingDecision:
        if features.decision == "GO":
            if features.deterministic_validation_pass:
                return RoutingDecision("DETERMINISTIC", "validated_luna_go")
            return RoutingDecision("SOL", "luna_go_failed_local_validation")
        if features.decision == "UNCERTAIN":
            return RoutingDecision("SOL", "luna_uncertain")
        if features.genuine_required_evidence_absence:
            return RoutingDecision("SAFE_ABSTAIN", "required_evidence_absent")
        if (
            features.deterministic_evidence_complete
            or features.version_resolved
            or features.authorized_candidate_evidence_complete
        ):
            return RoutingDecision("SOL", "suspicious_luna_abstention")
        return RoutingDecision("SAFE_ABSTAIN", "non_suspicious_luna_abstention")
