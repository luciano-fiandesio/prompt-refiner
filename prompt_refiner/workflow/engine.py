import json
import logging
import re as pyre
from textwrap import dedent
from typing import Any, List, Optional

import regex
from agno.workflow.loop import Loop
from agno.workflow.step import Step
from agno.workflow.types import StepOutput
from agno.workflow.workflow import Workflow

from prompt_refiner.agents.builder import AgentBundle
from prompt_refiner.agents.models import (
    AnalysisSummary,
    ClarificationAssessment,
    ClarificationTranscript,
    EvaluationReport,
    PromptDraft,
    RevisedPrompt,
    TestCaseReport,
)


FENCE_OPEN_RE = pyre.compile(r"^```(?:json)?\s*", pyre.IGNORECASE)
FENCE_CLOSE_RE = pyre.compile(r"\s*```$")
JSON_BLOCK_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", regex.DOTALL)


log = logging.getLogger("prompt_refinement")


class PromptRefinementEngine:
    """Wrap the existing prompt refinement workflow for reuse."""

    def __init__(
        self,
        agents: AgentBundle,
        *,
        clarification_max_rounds: int = 3,
        loop_max_iterations: int = 5,
    ) -> None:
        self._agents = agents
        self._clarification_max_rounds = clarification_max_rounds
        self._loop_max_iterations = loop_max_iterations
        self._should_continue_handler = None
        self._clarify_enabled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, user_prompt: str, *, should_continue=None, skip_clarification: bool = False):
        """Execute the full workflow and return the Workflow result."""
        previous = self._should_continue_handler
        self._should_continue_handler = should_continue or self._default_should_continue
        previous_clarify = self._clarify_enabled
        self._clarify_enabled = not skip_clarification
        workflow = self._build_workflow()
        try:
            return workflow.run(input={"user_prompt": user_prompt})
        finally:
            self._should_continue_handler = previous
            self._clarify_enabled = previous_clarify

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------
    def _build_workflow(self) -> Workflow:
        steps = []
        if self._clarify_enabled:
            steps.append(Step(name="clarify", executor=self._clarify_step))

        steps.extend(
            [
                Step(name="analyze", executor=self._analyze_step),
                Step(name="synthesize", executor=self._synthesize_step),
                Step(name="evaluate", executor=self._evaluate_step),
                Step(name="test_case", executor=self._testcase_step),
                Step(name="revision", executor=self._revise_step),
                Step(name="update_and_loop", executor=self._update_and_loop_step),
                Step(name="ask_user", executor=self._ask_user_step),
            ]
        )

        refinement_loop = Loop(
            name="Prompt Refinement Loop",
            steps=steps,
            end_condition=self._end_condition,
            max_iterations=self._loop_max_iterations,
        )

        return Workflow(
            name="prompt_refinement",
            steps=[
                refinement_loop,
                Step(name="report_final", executor=self._report_final_step),
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = FENCE_OPEN_RE.sub("", s)
            s = FENCE_CLOSE_RE.sub("", s)
        return s.strip()

    @staticmethod
    def _truncate(v: Any, n: int = 800) -> str:
        try:
            if isinstance(v, (dict, list)):
                text = json.dumps(v, ensure_ascii=False)
            else:
                text = str(v)
        except Exception:
            text = repr(v)
        return (text[:n] + " …[truncated]") if len(text) > n else text

    @classmethod
    def _largest_json_object(cls, text: str) -> Optional[str]:
        try:
            matches = list(JSON_BLOCK_RE.finditer(text))
        except Exception as exc:
            log.debug("regex (?R) extract failed; err=%s", exc)
            matches = []

        if matches:
            match = max(matches, key=lambda m: (m.end() - m.start()))
            return text[match.start() : match.end()]

        best_start = best_end = -1
        stack = []
        start_idx = None
        for idx, ch in enumerate(text):
            if ch == "{":
                stack.append(idx)
                if len(stack) == 1:
                    start_idx = idx
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        end_idx = idx + 1
                        if (best_end - best_start) < (end_idx - start_idx):
                            best_start, best_end = start_idx, end_idx
                        start_idx = None
        if best_start != -1:
            return text[best_start:best_end]
        return None

    @classmethod
    def _clean_json_text(cls, value: str) -> str:
        stripped = cls._strip_code_fences(value)
        text = stripped.strip()
        if not (text.startswith("{") and text.endswith("}")):
            candidate = cls._largest_json_object(stripped)
            if candidate:
                text = candidate
        text = text.replace(""", '"').replace(""", '"').replace("'", "'")
        text = pyre.sub(r",\s*([}\]])", r"\1", text)
        return text.strip()

    def _run_with_logging(self, agent, *, user_input: str, system: str, step_name: str):
        try:
            run = agent.run(input=user_input, system=system)
        except Exception as exc:
            print(f"[{step_name}] agent.run failed: {exc}")
            raise
        try:
            preview = self._truncate(run.content, 800)
        except Exception:
            preview = "<unprintable>"
        print(f"[{step_name}] raw output preview:\n{preview}\n")
        return run

    def _to_model(self, model_cls, raw, *, step: str = "unknown"):
        try:
            if isinstance(raw, model_cls):
                return raw
            if isinstance(raw, dict):
                return model_cls.model_validate(raw)
            if isinstance(raw, str):
                cleaned = self._clean_json_text(raw)
                try:
                    data = json.loads(cleaned)
                except Exception as json_exc:
                    log.warning(
                        "JSON parse failed | step=%s model=%s\nRAW: %s\nCLEANED: %s\nERROR: %s",
                        step,
                        model_cls.__name__,
                        self._truncate(raw),
                        self._truncate(cleaned),
                        json_exc,
                    )
                    raise
                try:
                    return model_cls.model_validate(data)
                except Exception as validation_exc:
                    log.warning(
                        "Model validate failed | step=%s model=%s\nDATA: %s\nERROR: %s",
                        step,
                        model_cls.__name__,
                        self._truncate(data),
                        validation_exc,
                    )
                    raise
            log.warning(
                "Unsupported type | step=%s model=%s type=%s",
                step,
                model_cls.__name__,
                type(raw),
            )
            raise ValueError(f"Cannot coerce to {model_cls.__name__}: {type(raw)}")
        except Exception:
            raise

    # ------------------------------------------------------------------
    # Step executors
    # ------------------------------------------------------------------
    def _current_prompt(self, step_input, session_state) -> str:
        return (
            session_state.get("current_prompt")
            or session_state.get("clarified_prompt")
            or step_input.input["user_prompt"]
        )

    def _clarify_step(self, step_input, session_state) -> StepOutput:
        if session_state.get("iteration_count"):
            return StepOutput(content={"clarification_skipped": True})

        user_prompt = step_input.input.get("user_prompt", "")
        history = list(session_state.get("clarification_history", []))
        rounds = 0
        latest_assessment: Optional[ClarificationAssessment] = None

        while rounds < self._clarification_max_rounds:
            rounds += 1
            context = dedent(
                f"""\
                USER PROMPT:
                {user_prompt}

                PRIOR CLARIFICATION:
                {self._format_history_for_agent(history)}
                """
            )
            system = dedent(
                """\
                You are the first pass in a prompt refinement workflow.
                Decide if the prompt needs clarification before analysis.
                Return ONLY JSON matching:
                {
                  "needs_clarification": bool,
                  "questions": [str],
                  "notes": str | null
                }
                Ask the minimum number of domain-relevant questions.
                Base your decision on clarity, target audience, concrete deliverables, and constraints.
                """
            )

            try:
                run = self._run_with_logging(
                    self._agents.question,
                    user_input=context,
                    system=system,
                    step_name="clarify",
                )
            except Exception as exc:
                log.warning(
                    "Question agent failed; continuing without clarification: %s",
                    exc,
                )
                latest_assessment = ClarificationAssessment(
                    needs_clarification=False, questions=[], notes="agent failure"
                )
                break

            try:
                assessment = self._to_model(
                    ClarificationAssessment, run.content, step="clarify"
                )
            except Exception as exc:
                log.warning(
                    "Clarification parse failed; treating as no clarification needed. err=%s",
                    exc,
                )
                latest_assessment = ClarificationAssessment(
                    needs_clarification=False, questions=[], notes="parse failure"
                )
                break

            latest_assessment = assessment
            session_state["clarification_assessment"] = assessment.model_dump()

            if not assessment.needs_clarification or not assessment.questions:
                break

            input_blocked = False
            for question in assessment.questions:
                print("\n[clarify]", question, flush=True)
                try:
                    answer = input("Your answer: ").strip()
                except EOFError:
                    answer = ""
                    input_blocked = True
                history.append({"question": question, "answer": answer})

            session_state["clarification_history"] = history

            if input_blocked and assessment.needs_clarification:
                msg = "Unable to collect answers required for clarification."
                session_state["clarification_error"] = msg
                session_state["clarified_prompt"] = self._compose_clarified_prompt(
                    user_prompt, history
                )
                return StepOutput(content={"clarification_error": msg}, success=False)

        if latest_assessment and latest_assessment.needs_clarification:
            log.warning(
                "Clarification loop hit max rounds; proceeding with available context."
            )

        if latest_assessment and "clarification_assessment" not in session_state:
            session_state["clarification_assessment"] = latest_assessment.model_dump()

        clarified_prompt = self._compose_clarified_prompt(user_prompt, history)
        session_state["clarified_prompt"] = clarified_prompt
        if history:
            session_state["clarification_history"] = history
            summary_lines = []
            for idx, qa in enumerate(history, 1):
                summary_lines.append(
                    f"{idx}. {qa.get('question', '')} -> {qa.get('answer', '')}"
                )
            transcript = ClarificationTranscript(
                questions=[qa.get("question", "") for qa in history],
                answers=[qa.get("answer", "") for qa in history],
                summary="\n".join(summary_lines),
            )
            session_state["clarification_transcript"] = transcript.model_dump()
        else:
            session_state.pop("clarification_transcript", None)

        return StepOutput(
            content={
                "clarified_prompt": clarified_prompt,
                "clarification_rounds": rounds,
                "questions_asked": len(history),
            }
        )

    def _analyze_step(self, step_input, session_state) -> StepOutput:
        user_prompt = self._current_prompt(step_input, session_state)
        system = (
            "Extract a structured analysis of the user prompt. "
            "Return ONLY JSON matching this schema: "
            '{"primary_goal": str, "secondary_objectives": [str], "audience": str, "output_format": str}'
        )
        run = self._run_with_logging(
            self._agents.analyzer,
            user_input=user_prompt,
            system=system,
            step_name="analyze",
        )
        analysis = self._to_model(AnalysisSummary, run.content, step="analyze")
        session_state["analysis"] = analysis.model_dump()
        return StepOutput(content=analysis.model_dump())

    def _synthesize_step(self, step_input, session_state) -> StepOutput:
        user_prompt = self._current_prompt(step_input, session_state)
        analysis = session_state.get("analysis")
        if not analysis:
            return StepOutput(content={"error": "Missing analysis"}, success=False)

        system = dedent(
            """\
            You are a prompt engineer creating META-PROMPTS (instructions for other AI assistants).

            CRITICAL: You must create INSTRUCTIONS, not execute tasks.

            Example of CORRECT behavior:
            User need: "I need to write an email to my manager about a raise"
            Your output: "Write a professional email to a manager requesting a salary increase. The email should include: current contributions, market research, specific salary request, and proposed timeline. Maintain a respectful but confident tone."

            Example of INCORRECT behavior (DO NOT DO THIS):
            User need: "I need to write an email to my manager about a raise"
            Your output: "Dear Manager, I would like to discuss my compensation..."

            Rules:
            1. Create instructions that ANOTHER AI will follow
            2. Do NOT execute the task yourself
            3. Do NOT write content, generate examples, or provide solutions
            4. Output only the meta-prompt text, ready to hand to another assistant
            """
        )
        prompt = (
            "The user has expressed a need. Create META-INSTRUCTIONS for another AI to fulfill this need.\n\n"
            "DO NOT fulfill the need yourself. Create instructions for another AI to follow.\n\n"
            f"User's need analysis:\n"
            f"- Primary goal: {analysis['primary_goal']}\n"
            f"- Secondary objectives: {analysis['secondary_objectives']}\n"
            f"- Target audience: {analysis['audience']}\n"
            f"- Expected output format: {analysis['output_format']}\n\n"
            "Original user need that requires instructions:\n"
            f'"""{user_prompt}"""\n\n'
            "Now create a META-PROMPT (instructions for another AI) to address this need:"
        )
        run = self._run_with_logging(
            self._agents.synthesizer,
            user_input=prompt,
            system=system,
            step_name="synthesize",
        )
        try:
            draft = self._to_model(PromptDraft, run.content, step="synthesize")
            content = draft.model_dump()
        except Exception:
            content = {"working_prompt": str(run.content)}
        session_state["working_prompt"] = content["working_prompt"]
        return StepOutput(content=content)

    def _evaluate_step(self, step_input, session_state) -> StepOutput:
        working_prompt = session_state.get("working_prompt")
        if not working_prompt:
            return StepOutput(content={"error": "Missing working_prompt"}, success=False)

        system = dedent("""\
            Evaluate the generated prompt strictly as a META-PROMPT that will direct another AI assistant.

            FIRST CHECK: Is this instructions or executed content?
            - Instructions tell an AI what to do (CORRECT)
            - Executed content is the actual output/solution (INCORRECT - MUST FLAG)

            Example of CORRECT instructions:
            "Generate a professional email to request PTO. Include dates, reason, and coverage plan."

            Example of INCORRECT execution:
            "Dear Manager, I am writing to request PTO from..."

            Evaluation criteria:
            - clarity: Are the instructions clear? If it's executed content instead of instructions, mark as "FAILED - This is executed content, not instructions"
            - conciseness: Is it appropriately brief while complete?
            - completeness: Does it specify all necessary elements?
            - goal_alignment: Does it align with the user's intent? If it executes instead of instructs, mark as "FAILED - Executes task instead of providing instructions"
            - context_awareness: Does it understand the domain and requirements?
            - expected_output: Does it clearly describe what the AI should produce?

            Return ONLY JSON matching this schema:
            '{"clarity": str, "conciseness": str, "completeness": str, "goal_alignment": str, "context_awareness": str, "expected_output": str}'
            """)

        run = self._run_with_logging(
            self._agents.evaluator,
            user_input=working_prompt,
            system=system,
            step_name="evaluate",
        )
        try:
            evaluation = self._to_model(
                EvaluationReport, run.content, step="evaluate"
            ).model_dump()
            session_state["evaluation"] = evaluation
            return StepOutput(content=evaluation)
        except Exception:
            log.info(
                "Falling back to plain text | step=revision\nRAW: %s",
                self._truncate(run.content),
            )
            fallback = {"evaluation_fallback": str(run.content)}
            session_state["evaluation_fallback"] = fallback["evaluation_fallback"]
            return StepOutput(content=fallback, success=False)

    def _testcase_step(self, step_input, session_state) -> StepOutput:
        working_prompt = session_state.get("working_prompt")
        if not working_prompt:
            return StepOutput(content={"error": "Missing working_prompt"}, success=False)

        system = dedent("""\
            Create and evaluate hypothetical test cases for the generated prompt.
            Return ONLY JSON matching this schema:
            '{"report": str}'
            """)

        run = self._agents.testcase.run(input=working_prompt, system=system)
        try:
            report = self._to_model(
                TestCaseReport, run.content, step="test_case"
            ).model_dump()
            session_state["testcase_report"] = report
            return StepOutput(content=report)
        except Exception:
            log.info(
                "Falling back to plain text | step=revision\nRAW: %s",
                self._truncate(run.content),
            )
            fallback = {"testcase_report": str(run.content)}
            session_state["testcase_report"] = fallback["testcase_report"]
            return StepOutput(content=fallback, success=False)

    def _revise_step(self, step_input, session_state) -> StepOutput:
        working_prompt = session_state.get("working_prompt")
        evaluation = session_state.get("evaluation")
        evaluation_fallback = session_state.get("evaluation_fallback")
        testcase_report = session_state.get("testcase_report")

        if not working_prompt:
            return StepOutput(content={"error": "Missing working_prompt"}, success=False)
        if not (evaluation or evaluation_fallback):
            return StepOutput(content={"error": "Missing evaluation"}, success=False)
        if not testcase_report:
            return StepOutput(content={"error": "Missing testcase_report"}, success=False)

        if evaluation:
            eval_block = dedent(f"""\
                clarity: {evaluation['clarity']}
                conciseness: {evaluation['conciseness']}
                completeness: {evaluation['completeness']}
                goal_alignment: {evaluation['goal_alignment']}
                context_awareness: {evaluation['context_awareness']}
                expected_output: {evaluation['expected_output']}
            """)
        else:
            eval_block = str(evaluation_fallback)

        system = dedent("""\
                You are a prompt revisor creating improved META-PROMPTS (instructions for other AI assistants).

                CRITICAL VALIDATION: Check if the current prompt is instructions or executed content.
                - If it contains actual content (e.g., an email, a report, code), it's WRONG
                - If it contains instructions for creating content, it's CORRECT

                Example of CORRECT meta-prompt:
                "Create a formal business email requesting a meeting. Include: purpose, proposed dates, duration, and agenda items."

                Example of INCORRECT execution (MUST FIX):
                "Subject: Meeting Request
                Dear John,
                I would like to schedule a meeting..."

                Your revision must:
                  • Be instructions that tell another AI WHAT to do
                  • NOT contain the actual output/content/solution
                  • Clearly state the goal, inputs, constraints, and desired outputs
                  • Be actionable and specific enough for another AI agent to follow

                Return ONLY JSON matching this schema exactly:
                {
                  "revised_prompt": "string",
                  "change_log_entry": { "changes": "string", "rationale": "string" }
                }
            """)

        testcase_block = (
            testcase_report
            if isinstance(testcase_report, str)
            else json.dumps(testcase_report)
        )

        revision_input = dedent(
            f"""\
                CURRENT WORKING PROMPT:
                ```text
                {working_prompt}
                ```

                EVALUATION REPORT:
                ```text
                {eval_block}
                ```

                TEST CASE REPORT:
                ```text
                {testcase_block}
                ```
            """
        )

        run = self._agents.revisor.run(input=revision_input, system=system)
        try:
            revised = self._to_model(
                RevisedPrompt, run.content, step="revision"
            ).model_dump()
            session_state["revised_prompt"] = revised
            return StepOutput(content=revised)
        except Exception:
            log.info(
                "Falling back to plain text | step=revision\nRAW: %s",
                self._truncate(run.content),
            )
            fallback = {"revised_prompt": str(run.content)}
            session_state["revised_prompt"] = fallback["revised_prompt"]
            return StepOutput(content=fallback, success=False)

    def _update_and_loop_step(self, step_input, session_state) -> StepOutput:
        revised = session_state.get("revised_prompt")
        if isinstance(revised, dict):
            revised_prompt = revised.get("revised_prompt")
            change_log_entry = revised.get("change_log_entry")
        else:
            revised_prompt = revised
            change_log_entry = None

        if not revised_prompt:
            return StepOutput(content={"error": "Missing revised_prompt"}, success=False)

        session_state["current_prompt"] = revised_prompt
        session_state["iteration_count"] = session_state.get("iteration_count", 0) + 1
        history = session_state.get("history_log", [])
        if change_log_entry:
            history.append(change_log_entry)
        session_state["history_log"] = history

        for key in [
            "analysis",
            "working_prompt",
            "evaluation",
            "evaluation_fallback",
            "testcase_report",
            "clarified_prompt",
            "clarification_assessment",
            "clarification_history",
            "clarification_transcript",
            "clarification_error",
        ]:
            session_state.pop(key, None)

        return StepOutput(
            content={
                "current_prompt": revised_prompt,
                "iteration_count": session_state["iteration_count"],
                "history_len": len(history),
            }
        )

    def _report_final_step(self, step_input, session_state) -> StepOutput:
        latest = (
            session_state.get("current_prompt")
            or (session_state.get("revised_prompt") or {}).get("revised_prompt")
            or session_state.get("working_prompt")
            or "(no prompt)"
        )
        iterations = session_state.get("iteration_count", 0)
        history = session_state.get("history_log", [])
        lines = [
            "=== Final Prompt Refinement Report ===",
            f"Iterations: {iterations}",
            "",
            "Latest Prompt:",
            latest,
            "",
            "Change Log:",
        ]
        if not history:
            lines.append("- (empty)")
        else:
            for idx, entry in enumerate(history, 1):
                if isinstance(entry, dict):
                    lines.append(f"{idx}. Changes: {entry.get('changes', '')}")
                    lines.append(f"   Rationale: {entry.get('rationale', '')}")
                else:
                    lines.append(f"{idx}. {entry}")
        final_report = "\n".join(lines)
        session_state["final_report"] = final_report
        return StepOutput(
            content={
                "final_report": final_report,
                "latest_prompt": latest,
                "history_log": history,
                "iteration_count": iterations,
            }
        )

    def _ask_user_step(self, step_input, session_state) -> StepOutput:
        latest = (
            session_state.get("current_prompt")
            or (session_state.get("revised_prompt") or {}).get("revised_prompt")
            or session_state.get("working_prompt")
            or step_input.input.get("user_prompt")
            or "(no prompt)"
        )

        print("\n--- Latest prompt ---\n", latest, "\n", flush=True)
        handler = self._should_continue_handler or self._default_should_continue
        should_continue = bool(handler(session_state))
        stop = not should_continue
        return StepOutput(content={"stop": stop})

    def _end_condition(self, outputs) -> bool:
        for out in reversed(outputs or []):
            content = getattr(out, "content", None) or {}
            if isinstance(content, dict) and "stop" in content:
                stop = bool(content["stop"])
                print(f"[loop] user stop={stop}; break={stop}")
                return stop
        print("[loop] stop flag not found; breaking defensively")
        return True

    # ------------------------------------------------------------------
    # Continuation handlers
    # ------------------------------------------------------------------
    def _default_should_continue(self, session_state) -> bool:
        try:
            ans = input("Run another iteration? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        return ans in ("y", "yes")

    @staticmethod
    def _format_history_for_agent(history: List[dict]) -> str:
        if not history:
            return "(none)"
        lines = []
        for idx, qa in enumerate(history, 1):
            lines.append(f"{idx}. Q: {qa.get('question', '')}")
            lines.append(f"   A: {qa.get('answer', '')}")
        return "\n".join(lines)

    @staticmethod
    def _compose_clarified_prompt(original_prompt: str, history: List[dict]) -> str:
        if not history:
            return original_prompt
        lines = [original_prompt.strip(), "", "Refinement Notes:"]
        for idx, qa in enumerate(history, 1):
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            lines.append(f"{idx}. {question}")
            lines.append(f"   Answer: {answer or '(no answer provided)'}")
        return "\n".join(lines)
