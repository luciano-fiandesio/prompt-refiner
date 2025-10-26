from typing import List, Optional

from pydantic import BaseModel, Field


class ClarificationAssessment(BaseModel):
    needs_clarification: bool
    questions: List[str] = []
    notes: Optional[str] = None


class ClarificationTranscript(BaseModel):
    questions: List[str]
    answers: List[str]
    summary: str


class AnalysisSummary(BaseModel):
    primary_goal: str
    secondary_objectives: List[str]
    audience: str
    output_format: str


class PromptDraft(BaseModel):
    working_prompt: str


class EvaluationReport(BaseModel):
    clarity: str = Field(
        ...,
        description="How easily understandable is the prompt?",
    )
    conciseness: str = Field(
        ...,
        description="Does the prompt avoid unnecessary words or complexity?",
    )
    completeness: str = Field(
        ...,
        description="Does the prompt include all necessary information and instructions?",
    )
    goal_alignment: str = Field(
        ...,
        description="How well does the prompt align with the initial goal?",
    )
    context_awareness: str = Field(
        ...,
        description="Does the prompt consider and incorporate relevant context?",
    )
    expected_output: str = Field(
        ...,
        description="Is the prompt's outcome clear and well-formatted?",
    )


class TestCaseReport(BaseModel):
    report: str


class ChangeLogEntry(BaseModel):
    changes: str
    rationale: str


class RevisedPrompt(BaseModel):
    revised_prompt: str
    change_log_entry: ChangeLogEntry


__all__ = [
    "ClarificationAssessment",
    "ClarificationTranscript",
    "AnalysisSummary",
    "PromptDraft",
    "EvaluationReport",
    "TestCaseReport",
    "ChangeLogEntry",
    "RevisedPrompt",
]
