from __future__ import annotations

from rag_workbench.experiments.v2_quality_recovery import stable_hash

QUESTION_INJECTION_GUARD_V2_CONFIG = {
    "guard": "QUESTION_INJECTION_GUARD_V2",
    "version": 2,
    "benign_meta_policy_phrase_handling": "what does ... policy ... phrase",
    "intent_rules": [
        "higher-priority + follow/obey",
        "treat retrieved/document text as system/authoritative",
        "override/ignore/bypass previous/system/assistant rules (+ safety/policy tokens)",
        "follow/obey/comply/carry out/execute retrieved/instruction/embedded content",
        "reveal/expose/secrets/credentials/confidential/private",
        "drop citations / answer without citations / answer without evidence",
        "ignore/bypass safety/guardrails/restrictions",
        "answer despite lack of evidence",
    ],
}

QUESTION_INJECTION_GUARD_V3_CONFIG = {
    "guard": "QUESTION_INJECTION_GUARD_V3",
    "version": 3,
    "benign_analysis_handling": "allow policy/security analysis without imperative hostile intent",
    "intent_rules": [
        "authority override over system/assistant/rules/policy/guardrails",
        "forced compliance with command/instruction/directive",
        "instruction following from untrusted retrieved/quoted/document text",
        "role override (pretend you are admin/system/developer)",
        "secret/system prompt exfiltration directives",
        "output without evidence/citations directives",
    ],
}


ANSWERABILITY_CONSTRAINT_GUARD_V1_CONFIG = {
    "guard": "ANSWERABILITY_CONSTRAINT_GUARD_V1",
    "version": 1,
    "numeric_entailment_mode": "operator-strict-on-evidence-same-value",
    "fail_closed_when_constraint_detected_but_entailment_not_found": True,
    "supported_numeric_operators": ["EXACT", "ABOVE", "BELOW", "AT_LEAST", "AT_MOST"],
    "supported_date_ops": ["BEFORE", "AFTER", "ON"],
}


ANSWERABILITY_CONSTRAINT_GUARD_V2_CONFIG = {
    "guard": "ANSWERABILITY_CONSTRAINT_GUARD_V2",
    "version": 2,
    "numeric_entailment_mode": "open-closed-boundary-set-inclusion",
    "eq_on_strict_operator_match": True,
    "fail_closed_when_constraint_detected_but_entailment_not_found": True,
    "supported_numeric_operators": ["EQ", "GT", "GE", "LT", "LE"],
    "supported_date_ops": ["BEFORE", "AFTER", "ON"],
}


QUESTION_INJECTION_GUARD_V2_HASH = stable_hash(QUESTION_INJECTION_GUARD_V2_CONFIG)
QUESTION_INJECTION_GUARD_V3_HASH = stable_hash(QUESTION_INJECTION_GUARD_V3_CONFIG)
ANSWERABILITY_CONSTRAINT_GUARD_V1_HASH = stable_hash(
    ANSWERABILITY_CONSTRAINT_GUARD_V1_CONFIG
)

ANSWERABILITY_CONSTRAINT_GUARD_V2_HASH = stable_hash(ANSWERABILITY_CONSTRAINT_GUARD_V2_CONFIG)

SAFE_RECOVERY_BOUNDARY_V2_HASH = stable_hash(
    {
        "composition": "SAFE_RECOVERY_BOUNDARY_V2",
        "components": {
            "QUESTION_INJECTION_GUARD_V2": QUESTION_INJECTION_GUARD_V2_HASH,
            "ANSWERABILITY_CONSTRAINT_GUARD_V1": ANSWERABILITY_CONSTRAINT_GUARD_V1_HASH,
        },
    }
)

SAFE_RECOVERY_BOUNDARY_V3_HASH = stable_hash(
    {
        "composition": "SAFE_RECOVERY_BOUNDARY_V3",
        "components": {
            "QUESTION_INJECTION_GUARD_V2": QUESTION_INJECTION_GUARD_V2_HASH,
            "ANSWERABILITY_CONSTRAINT_GUARD_V2": ANSWERABILITY_CONSTRAINT_GUARD_V2_HASH,
        },
    }
)

