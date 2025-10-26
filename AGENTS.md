# Repository Guidelines

## Project Structure & Module Organization
The repository is intentionally flat while workflows stabilize. `main.py` is the minimal CLI entry point; `main2.py` exercises the Gemini-backed finance agent. Prompt iteration lives in `prompt.py` and `prompt_claude.py`; keep any shared helpers in dedicated modules once they outgrow prototypes. Store credentials in a local `.env` and mirror production-ready code under a future `app/` package. Add automated checks under `tests/`, mirroring source module names.

## Build, Test, and Development Commands
- `uv sync` installs the locked dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py` sanity-checks the basic CLI wiring.
- `uv run python main2.py` streams the Gemini agent response (requires a configured `DEFAULT_MODEL`).
- `uv run python prompt.py` executes the multi-step workflow with logging enabled.
- `uv run ruff check .` enforces linting and formatting expectations; add `--fix` before committing.
- `uv run python run_api.py` launches the FastAPI service on `http://localhost:8000`.
- `uv run streamlit run streamlit_app.py` starts the single-user Streamlit front end.

## Coding Style & Naming Conventions
Target Python 3.13 with 4-space indentation. Use snake_case for functions and variables, CapWords for classes, and keep filenames lowercase with underscores. Prefer pure functions and typed dataclasses for shared utilities; isolate agent configuration in modules named `*_agent.py`. Let Ruff be the source of truth for styling—run it locally before pushing.

## Testing Guidelines
No automated suite exists yet; grow one with `pytest` placed in `tests/` and named `test_<module>.py`. Validate new workflows with fixture data and assert on both structured outputs and side effects (e.g., logging). Once tests land, run them with `uv run pytest` and aim for 80% coverage on new modules. Document any manual validation steps in PR descriptions until automation catches up.

## Commit & Pull Request Guidelines
Git history favors short, imperative messages (`add ruff`, `Better logging`); keep commits similarly scoped and descriptive. Open PRs with a clear summary, validation commands, linked issues, and screenshots or terminal excerpts when behavior changes. Call out environment or dependency updates explicitly and request review before merging, even for prompt-only adjustments.

## Environment & Secrets
Create a local `.env` (ignored by Git) with `DEFAULT_MODEL` and related API keys before running agent scripts. Never commit secrets or sample keys; redact sensitive output in logs and PRs. Rotate credentials regularly and rely on the platform secret manager for shared deployments.
