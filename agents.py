from dataclasses import dataclass
from typing import List, Optional

from agno.agent import Agent
from agno.models.google import Gemini
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


@dataclass(frozen=True)
class AgentBundle:
    analyzer: Agent
    question: Agent
    synthesizer: Agent
    evaluator: Agent
    testcase: Agent
    revisor: Agent


def build_agents(model_id: str) -> AgentBundle:
    """Construct all agents with a shared model identifier."""

    def make_agent(*, description: str, output_schema) -> Agent:
        return Agent(
            model=Gemini(id=model_id),
            description=description,
            output_schema=output_schema,
            structured_outputs=True,
            use_json_mode=True,
        )

    analyzer = make_agent(
        description="Analyze the current prompt to identify its core components.",
        output_schema=AnalysisSummary,
    )

    question = make_agent(
        description="Decide whether a prompt needs clarification and draft targeted questions.",
        output_schema=ClarificationAssessment,
    )

    synthesizer = make_agent(
        description="Generate a working prompt based on a prior analysis.",
        output_schema=PromptDraft,
    )

    evaluator = make_agent(
        description="Evaluate the generated prompt against defined criteria",
        output_schema=EvaluationReport,
    )

    testcase = make_agent(
        description="Create and evaluate hypothetical test cases for the generated prompt.",
        output_schema=TestCaseReport,
    )

    revisor = make_agent(
        description="Generate a revised prompt and document the changes.",
        output_schema=RevisedPrompt,
    )

    return AgentBundle(
        analyzer=analyzer,
        question=question,
        synthesizer=synthesizer,
        evaluator=evaluator,
        testcase=testcase,
        revisor=revisor,
    )


__all__ = [
    "ClarificationAssessment",
    "ClarificationTranscript",
    "AnalysisSummary",
    "PromptDraft",
    "EvaluationReport",
    "TestCaseReport",
    "ChangeLogEntry",
    "RevisedPrompt",
    "AgentBundle",
    "build_agents",
]
