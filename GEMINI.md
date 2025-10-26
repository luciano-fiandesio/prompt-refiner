# Project Overview

This project is a web application designed for refining prompts through an iterative process involving multiple AI agents. The application consists of a FastAPI backend that manages the prompt refinement workflow and a Streamlit frontend that provides a user interface for interacting with the system.

The core of the application is a prompt refinement engine that uses a series of AI agents, each with a specific role:

*   **Analyzer:** Analyzes the initial prompt to identify its core components.
*   **Questioner:** Asks clarifying questions to gather more information about the user's goals.
*   **Synthesizer:** Generates a working prompt based on the initial prompt and the user's answers to the clarifying questions.
*   **Evaluator:** Evaluates the generated prompt against a set of predefined criteria.
*   **Test Case Generator:** Creates and evaluates hypothetical test cases for the generated prompt.
*   **Revisor:** Revises the prompt based on the evaluation and test case results.

The application uses the `agno` library to build and manage the AI agents, and the `Gemini` model for the underlying language model.

# Building and Running

## Prerequisites

*   Python 3.13+
*   `uv` package manager

## Installation

1.  Create a virtual environment:
    ```bash
    uv venv
    ```
2.  Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

## Running the Application

1.  **Run the API:**
    ```bash
    uv run python run_api.py
    ```
    The API will be available at `http://localhost:8000`.

2.  **Run the Streamlit UI:**
    ```bash
    uv run streamlit run streamlit_app.py
    ```
    The Streamlit UI will be available at `http://localhost:8501`.

# Development Conventions

*   **Code Style:** The project uses `ruff` for linting and formatting.
*   **Type Hinting:** The project uses type hints extensively.
*   **Dependencies:** The project uses `uv` for dependency management. The dependencies are listed in the `pyproject.toml` file.
*   **Configuration:** The application is configured through a `.env` file. The `DEFAULT_MODEL` environment variable must be set to the ID of the Gemini model to use.
