import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SessionState:
    session_id: str
    initial_prompt: str
    current_prompt: Optional[str] = None
    iteration_count: int = 0
    clarification_history: list[dict] = field(default_factory=list)
    clarification_checklist: dict[str, str] = field(
        default_factory=lambda: {
            "audience": "pending",
            "goal": "pending",
            "inputs": "pending",
            "output": "pending",
            "constraints": "pending",
        }
    )
    change_log: list[dict] = field(default_factory=list)
    step_timestamps: Dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    status: str = "awaiting_question"  # awaiting_question | ready_for_refinement | finished |
    pending_question: Optional[dict] = None


class InMemorySessionStore:
    """Single-session store backing the FastAPI endpoints."""

    def __init__(self) -> None:
        self._session: Optional[SessionState] = None

    def create_session(self, initial_prompt: str) -> SessionState:
        # Drop any existing session (single-user requirement)
        self._session = SessionState(
            session_id=str(uuid.uuid4()),
            initial_prompt=initial_prompt,
        )
        return self._session

    def get_session(self) -> Optional[SessionState]:
        return self._session

    def require_session(self) -> SessionState:
        if not self._session:
            raise RuntimeError("No active session")
        return self._session

    def clear_session(self) -> None:
        self._session = None


__all__ = ["SessionState", "InMemorySessionStore"]
