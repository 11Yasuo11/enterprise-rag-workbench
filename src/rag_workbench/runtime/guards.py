"""Request-scoped Luna/Sol call guards (no cross-request leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelKind = Literal["LUNA", "SOL"]


class RequestGuardError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass
class RequestModelGuard:
    """Bounds model calls for a single user request."""

    max_luna_calls: int = 1
    max_sol_calls: int = 1
    luna_calls: int = 0
    sol_calls: int = 0

    def authorize(self, model: ModelKind) -> None:
        if model == "LUNA":
            if self.luna_calls >= self.max_luna_calls:
                raise RequestGuardError("REQUEST_LUNA_CALL_LIMIT")
            self.luna_calls += 1
            return
        if self.sol_calls >= self.max_sol_calls:
            raise RequestGuardError("REQUEST_SOL_CALL_LIMIT")
        self.sol_calls += 1
