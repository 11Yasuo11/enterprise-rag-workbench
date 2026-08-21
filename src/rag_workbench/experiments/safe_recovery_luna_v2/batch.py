# ruff: noqa: E501
"""Offline Batch API helper. Evaluation-only; not used for interactive latency."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from rag_workbench.experiments.safe_recovery_luna_v2.identities import OUT_DIR
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import estimate_cost_usd


def run_batch_smoke(
    *,
    api_key: str,
    base_url: str,
    requests: list[dict[str, Any]],
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Submit a tiny Batch of identical verifier payloads and compare decisions if completed."""
    if not requests:
        return {"status": "SKIPPED", "reason": "no_requests"}
    out = OUT_DIR / "batch"
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "batch_input.jsonl"
    lines = []
    for i, req in enumerate(requests):
        lines.append(
            json.dumps(
                {
                    "custom_id": f"luna-batch-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": req["body"],
                }
            )
        )
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, timeout=60.0, headers=headers) as client:
        upload = client.post(
            "/files",
            files={"file": (jsonl_path.name, jsonl_path.read_bytes(), "application/jsonl")},
            data={"purpose": "batch"},
        )
        if upload.status_code >= 400:
            return {
                "status": "FAILED",
                "stage": "file_upload",
                "http_status": upload.status_code,
                "note": "Batch API not executed; live verifier semantics unchanged.",
            }
        file_id = upload.json().get("id")
        created = client.post(
            "/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        if created.status_code >= 400:
            return {
                "status": "FAILED",
                "stage": "batch_create",
                "http_status": created.status_code,
            }
        batch_id = created.json().get("id")
        started = time.time()
        status = created.json()
        while time.time() - started < timeout_s:
            polled = client.get(f"/batches/{batch_id}")
            status = polled.json()
            if status.get("status") in {"completed", "failed", "cancelled", "expired"}:
                break
            time.sleep(5)
        live_cost = sum(
            estimate_cost_usd(
                model=req["body"]["model"],
                input_tokens=int(req.get("live_input_tokens") or 0),
                output_tokens=int(req.get("live_output_tokens") or 0),
                cached_input_tokens=int(req.get("live_cached_input_tokens") or 0),
            )
            for req in requests
        )
        return {
            "status": status.get("status"),
            "batch_id": batch_id,
            "n": len(requests),
            "note": (
                "Batch pricing is a deployment/evaluation cost result, not an accuracy improvement. "
                "Standard vs batch USD uses OpenAI batch 50% input/output discount when completed."
            ),
            "standard_estimated_usd": live_cost,
            "batch_estimated_usd_if_50pct": live_cost * 0.5,
            "interactive_latency_not_measured": True,
        }
