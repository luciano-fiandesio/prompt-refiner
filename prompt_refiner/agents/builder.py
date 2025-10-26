from dataclasses import dataclass

from agno.agent import Agent
from agno.models.google import Gemini

from .models import (
    AnalysisSummary,
    ClarificationAssessment,
    EvaluationReport,
    PromptDraft,
    RevisedPrompt,
    TestCaseReport,
)


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


__all__ = ["AgentBundle", "build_agents"]
