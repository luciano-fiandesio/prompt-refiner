- **Run the API**
  ```bash
  uv run python run_api.py
  ```
  The service starts on `http://localhost:8000`. Ensure `.env` contains `DEFAULT_MODEL`.

- **Launch the Streamlit UI**
  ```bash
  uv run streamlit run streamlit_app.py
  ```
  Use the app to walk through clarification questions and iterate on prompts.
