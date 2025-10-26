import json
from typing import Optional

import httpx


class PromptRefinementClient:
    """Thin helper for the Streamlit frontend to talk to the FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        timeout = httpx.Timeout(300.0, connect=10.0)
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def create_session(self, prompt: str, skip_questions: bool = False) -> dict:
        resp = self._client.post("/sessions", json={"initial_prompt": prompt, "skip_questions": skip_questions})
        resp.raise_for_status()
        return resp.json()

    def next_question(self, session_id: str) -> dict:
        resp = self._client.get(f"/sessions/{session_id}/next-question")
        resp.raise_for_status()
        return resp.json()

    def submit_answer(
        self, session_id: str, answer: str, question_id: Optional[str] = None
    ) -> dict:
        payload = {"answer": answer}
        if question_id:
            payload["question_id"] = question_id
        resp = self._client.post(
            f"/sessions/{session_id}/answer",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def run_iteration(self, session_id: str) -> dict:
        resp = self._client.post(f"/sessions/{session_id}/run")
        resp.raise_for_status()
        return resp.json()

    def get_report(self, session_id: str) -> dict:
        resp = self._client.get(f"/sessions/{session_id}/report")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


__all__ = ["PromptRefinementClient"]
