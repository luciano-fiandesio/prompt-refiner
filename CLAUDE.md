# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Install dependencies using uv package manager
uv sync

# Run the FastAPI backend service
uv run python run_api.py
# Service runs on http://localhost:8000

# Run the Streamlit frontend UI
uv run streamlit run streamlit_app.py
# UI runs on http://localhost:8501
```

### Code Quality
```bash
# Run linting and formatting checks
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix
```

### Testing & Debugging
```bash
# Run the prompt refinement workflow directly
uv run python prompt.py

# Test Gemini agent integration
uv run python main2.py
```

## Architecture Overview

### Core Components

**Prompt Refinement Engine** (`refinement_engine.py`)
- Orchestrates a multi-agent workflow using the `agno` library
- Manages iterative refinement through clarification, analysis, synthesis, evaluation, testing, and revision phases
- Supports configurable max rounds for clarification (default: 3) and loop iterations (default: 5)

**Agent System** (`agents.py`)
- Defines specialized agents for each workflow phase:
  - **Analyzer**: Extracts core components from initial prompts
  - **Questioner**: Generates clarifying questions for user context
  - **Synthesizer**: Creates working prompts from user inputs
  - **Evaluator**: Assesses prompts against quality criteria
  - **Test Case Generator**: Validates prompts with hypothetical scenarios
  - **Revisor**: Iteratively improves prompts based on evaluations
- Uses Pydantic models for structured agent outputs

**API Layer** (`api.py`)
- FastAPI application managing refinement sessions
- Session-based workflow with endpoints for creating sessions, retrieving questions, and submitting answers
- Uses `InMemorySessionStore` for session persistence

**Refinement Manager** (`refinement_manager.py`)
- Bridges the API and refinement engine
- Manages session state, clarification flow, and iteration control
- Handles asynchronous question/answer interaction

**Frontend** (`streamlit_app.py`)
- Streamlit UI for interactive refinement sessions
- Supports clarification Q&A, iteration control, and result viewing
- Communicates with backend via `streamlit_client.py`

### Key Dependencies
- **agno**: Workflow orchestration and agent management
- **google-genai**: Gemini model integration (configured via DEFAULT_MODEL env var)
- **fastapi/uvicorn**: REST API framework
- **streamlit**: Interactive web UI
- **ruff**: Code linting and formatting

### Configuration
- Requires `.env` file with `DEFAULT_MODEL` set to a Gemini model ID
- Python 3.13+ required
- Uses `uv` for dependency management via `pyproject.toml`

### Development Notes
- Flat repository structure while workflows stabilize
- All agent configurations isolated in `agents.py`
- Session management currently in-memory (not persistent across restarts)
- Structured outputs use Pydantic models with JSON extraction fallbacks