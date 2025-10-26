import json
import logging
import time
from textwrap import dedent
from typing import Optional, Sequence

from agents import AgentBundle, ClarificationAssessment
from refinement_engine import PromptRefinementEngine
from session_store import InMemorySessionStore, SessionState


logger = logging.getLogger("prompt_manager")


CHECKLIST_KEYWORDS = {
    "audience": ["audience", "consumer", "users", "stakeholder"],
    "goal": ["goal", "objective", "success", "purpose", "outcome"],
    "inputs": ["input", "data", "source", "evidence"],
    "output": ["output", "deliverable", "format", "response"],
    "constraints": ["constraint", "rule", "limit", "edge case", "compliance"],
}


class PromptRefinementManager:
    """Handle clarification flow and bookkeeping for API sessions."""

    def __init__(
        self,
        agents: AgentBundle,
        store: InMemorySessionStore,
        *,
        clarification_max_rounds: int = 5,
        loop_max_iterations: int = 5,
    ) -> None:
        self._agents = agents
        self._store = store
        self._clarification_max_rounds = clarification_max_rounds
        self._engine = PromptRefinementEngine(
            agents,
            clarification_max_rounds=clarification_max_rounds,
            loop_max_iterations=loop_max_iterations,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def start_session(self, initial_prompt: str, skip_questions: bool = False) -> SessionState:
        session = self._store.create_session(initial_prompt)
        session.current_prompt = initial_prompt

        if skip_questions:
            session.status = "ready_for_refinement"
        else:
            session.status = "awaiting_question"
            self._ensure_next_question(session)

        return session

    def get_session(self, session_id: str) -> SessionState:
        session = self._store.get_session()
        if not session or session.session_id != session_id:
            raise KeyError("Session not found")
        return session

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------
    def current_question(self, session: SessionState) -> Optional[dict]:
        return session.pending_question

    def submit_answer(
        self,
        session: SessionState,
        *,
        answer: str,
        question_id: Optional[str] = None,
    ) -> None:
        if not session.pending_question:
            raise RuntimeError("No question awaiting answer")

        question_payload = session.pending_question

        session.clarification_history.append(
            {
                "question": question_payload["text"],
                "answer": answer,
                "question_id": question_payload["id"],
                "answered_at": time.time(),
            }
        )
        session.pending_question = None

        self._update_clarified_prompt(session)
        self._update_checklist(session, question_payload["text"], answer)
        self._ensure_next_question(session)

    # ------------------------------------------------------------------
    # Refinement loop
    # ------------------------------------------------------------------
    def run_iteration(self, session: SessionState) -> dict:
        if session.pending_question:
            raise RuntimeError("Clarification still in progress")

        prompt = session.current_prompt or session.initial_prompt
        result = self._engine.run(
            prompt,
            should_continue=lambda _: False,
            skip_clarification=True,
        )

        final_step = result.step_results[-1].content or {}
        final_prompt = final_step.get("latest_prompt") or prompt
        change_log = final_step.get("history_log") or []
        iteration_count = final_step.get("iteration_count") or session.iteration_count

        session.iteration_count = iteration_count
        session.current_prompt = final_prompt
        session.change_log = change_log
        session.status = "finished"
        session.completed_at = time.time()

        payload = {
            "final_prompt": final_prompt,
            "change_log": change_log,
            "iteration_count": session.iteration_count,
            "step_timestamps": session.step_timestamps,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
        }

        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_next_question(self, session: SessionState) -> None:
        if session.pending_question:
            return

        if all(v == "resolved" for v in session.clarification_checklist.values()):
            session.status = "ready_for_refinement"
            return

        if len(session.clarification_history) >= self._clarification_max_rounds:
            session.status = "ready_for_refinement"
            return

        assessment = self._run_question_agent(
            session, session.initial_prompt, session.clarification_history
        )

        if not assessment.needs_clarification:
            session.status = "ready_for_refinement"
            session.pending_question = None
            return

        next_question = self._select_next_question(
            assessment.questions or [], session.clarification_history
        )

        if not next_question:
            session.status = "ready_for_refinement"
            return

        session.pending_question = {
            "id": f"q{len(session.clarification_history) + 1}",
            "text": next_question,
        }
        session.status = "awaiting_question"

    def _run_question_agent(
        self, session: SessionState, initial_prompt: str, history: Sequence[dict]
    ) -> ClarificationAssessment:
        context_lines = [
            "USER PROMPT:",
            initial_prompt,
            "",
            "PRIOR CLARIFICATION:",
            self._format_history_for_agent(history),
            "",
            "CURRENT CHECKLIST STATUS:",
        ]
        for key, value in session.clarification_checklist.items():
            context_lines.append(f"- {key}: {value}")
        context = "\n".join(context_lines)
        system = dedent(
            """\
            You are the clarification specialist in a prompt refinement workflow.
            Ensure the prompt covers ALL of these dimensions:
              1. Target audience or consumers of the output
              2. Primary goal / success criteria
              3. Required inputs or data sources
              4. Expected output format or deliverable shape
              5. Constraints, domain rules, or edge cases to respect

            Return ONLY JSON matching:
            {
              "needs_clarification": bool,
              "questions": [str],
              "notes": str | null
            }

            If ANY dimension is missing or vague, set needs_clarification=true and ask focused questions until every item is satisfied.
            Explain remaining gaps inside `notes` so the user understands why more detail is required.
            """
        )

        result = self._agents.question.run(input=context, system=system)
        payload = self._parse_json(result.content)
        return ClarificationAssessment.model_validate(payload)

    def _parse_json(self, raw) -> dict:
        if isinstance(raw, list) and raw:
            raw = raw[0]

        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            cleaned = PromptRefinementEngine._clean_json_text(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Failed to decode clarification response. raw=%s cleaned=%s",
                    raw,
                    cleaned,
                )
                raise ValueError("Unable to parse agent response as JSON") from exc
        logger.error("Unsupported clarification response type: %s", type(raw))
        raise ValueError("Unable to parse agent response as JSON")

    def _select_next_question(
        self, candidates: Sequence[str], history: Sequence[dict]
    ) -> Optional[str]:
        asked = {entry.get("question") for entry in history}
        for question in candidates:
            if question not in asked:
                return question
        return None

    def _update_clarified_prompt(self, session: SessionState) -> None:
        if not session.clarification_history:
            session.current_prompt = session.initial_prompt
            return

        lines = [session.initial_prompt.strip(), "", "Refinement Notes:"]
        for idx, qa in enumerate(session.clarification_history, 1):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip() or "(no answer provided)"
            lines.append(f"{idx}. {q}")
            lines.append(f"   Answer: {a}")
        session.current_prompt = "\n".join(lines)

    def _update_checklist(self, session: SessionState, question_text: str, answer: str) -> None:
        blob = f"{question_text}\n{answer}".lower()
        for key, keywords in CHECKLIST_KEYWORDS.items():
            if session.clarification_checklist.get(key) == "resolved":
                continue
            if any(token in blob for token in keywords):
                session.clarification_checklist[key] = "resolved"

        unresolved = [k for k, v in session.clarification_checklist.items() if v != "resolved"]
        if not unresolved:
            session.status = "ready_for_refinement"


    @staticmethod
    def _format_history_for_agent(history: Sequence[dict]) -> str:
        if not history:
            return "(none)"
        lines = []
        for idx, qa in enumerate(history, 1):
            lines.append(f"{idx}. Q: {qa.get('question', '')}")
            lines.append(f"   A: {qa.get('answer', '')}")
        return "\n".join(lines)
