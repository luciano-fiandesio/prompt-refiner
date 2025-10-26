import os
import sys
from dotenv import load_dotenv

from agents import build_agents
from refinement_engine import PromptRefinementEngine

BASEDIR = os.path.abspath(os.path.dirname(__file__))
_ = load_dotenv(os.path.join(BASEDIR, ".env"))


def _resolve_initial_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    try:
        prompt = input("Enter the prompt to refine: ").strip()
    except EOFError:
        prompt = ""
    return prompt


def main() -> None:
    model_id = os.getenv("DEFAULT_MODEL")
    if not model_id:
        print("Environment variable DEFAULT_MODEL is not set. Exiting.")
        return

    user_prompt = _resolve_initial_prompt()
    if not user_prompt:
        print("No prompt provided. Exiting.")
        return

    agents = build_agents(model_id)
    engine = PromptRefinementEngine(agents)

    result = engine.run(user_prompt)
    final_report = result.step_results[-1].content.get("final_report")

    print("Status:", result.status)
    print("Final report:", final_report)


if __name__ == "__main__":
    main()
