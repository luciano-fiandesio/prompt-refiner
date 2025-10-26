import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents import build_agents
from refinement_manager import PromptRefinementManager
from session_store import InMemorySessionStore


BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prompt_api")


app = FastAPI(title="Prompt Refinement API")
store = InMemorySessionStore()
manager: Optional[PromptRefinementManager] = None


class SessionCreateRequest(BaseModel):
    initial_prompt: str = Field(..., min_length=1)
    skip_questions: bool = False


class SessionCreateResponse(BaseModel):
    session_id: str
    status: str


@app.on_event("startup")
def _startup() -> None:
    global manager
    model_id = os.getenv("DEFAULT_MODEL")
    if not model_id:
        raise RuntimeError("DEFAULT_MODEL must be configured before starting the API")
    agents = build_agents(model_id)
    manager = PromptRefinementManager(agents=agents, store=store)


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(payload: SessionCreateRequest):
    if manager is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    try:
        session = manager.start_session(payload.initial_prompt, skip_questions=payload.skip_questions)
    except Exception as exc:
        logger.exception("Failed to start session")
        raise HTTPException(
            status_code=500,
            detail="Prompt refinement failed. Please restart the session.",
        ) from exc

    return SessionCreateResponse(session_id=session.session_id, status=session.status)


@app.get("/sessions/{session_id}/next-question")
def get_next_question(session_id: str):
    if manager is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    question = manager.current_question(session)
    if question:
        return {"status": "awaiting_answer", "question": question["text"], "question_id": question["id"]}

    return {"status": session.status}


class ClarificationAnswer(BaseModel):
    question_id: Optional[str] = None
    answer: str


@app.post("/sessions/{session_id}/answer")
def post_answer(session_id: str, payload: ClarificationAnswer):
    if manager is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        manager.submit_answer(
            session,
            answer=payload.answer,
            question_id=payload.question_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to process clarification answer")
        raise HTTPException(
            status_code=500,
            detail="Prompt refinement failed. Please restart the session.",
        ) from exc

    return {"status": session.status}


@app.post("/sessions/{session_id}/run")
def run_refinement(session_id: str):
    if manager is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        report = manager.run_iteration(session)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Iteration run failed")
        raise HTTPException(
            status_code=500,
            detail="Prompt refinement failed. Please restart the session.",
        ) from exc

    return report


@app.get("/sessions/{session_id}/report")
def get_report(session_id: str):
    if manager is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "finished":
        raise HTTPException(status_code=400, detail="Session has no completed report")

    return {
        "final_prompt": session.current_prompt,
        "change_log": session.change_log,
        "iteration_count": session.iteration_count,
        "step_timestamps": session.step_timestamps,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
    }
