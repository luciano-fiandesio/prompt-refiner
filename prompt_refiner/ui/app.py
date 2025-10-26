import streamlit as st

from prompt_refiner.ui.client import PromptRefinementClient


st.set_page_config(page_title="Prompt Refinement", layout="wide")
st.title("Prompt Refinement Console")

if "client" not in st.session_state:
    st.session_state.client = PromptRefinementClient()

client: PromptRefinementClient = st.session_state.client

st.session_state.setdefault("refinement_running", False)
st.session_state.setdefault("pending_iteration", False)
st.session_state.setdefault("iteration_message", None)

# Handle pending iteration before rendering UI
if st.session_state.get("pending_iteration") and st.session_state.get("session_id"):
    st.session_state.refinement_running = True
    session_id = st.session_state.session_id
    with st.spinner("Running refinement..."):
        try:
            report = client.run_iteration(session_id)
            st.session_state.report = report
            st.session_state.status = "finished"
            st.session_state.iteration_message = ("success", "Iteration complete.")
        except Exception as exc:
            st.session_state.iteration_message = ("error", f"Refinement failed: {exc}")
        finally:
            st.session_state.pending_iteration = False
            st.session_state.refinement_running = False
    st.rerun()


# Session storage helpers
def _reset_session_state():
    st.session_state.pop("session_id", None)
    st.session_state.pop("status", None)
    st.session_state.pop("current_question", None)
    st.session_state.pop("answers", None)
    st.session_state.pop("report", None)
    st.session_state.pop("answer_submit_locked", None)
    st.session_state.pop("refinement_running", None)
    st.session_state.pop("pending_iteration", None)
    st.session_state.pop("iteration_message", None)
    st.session_state.pop("modal_active", None)
    st.session_state.pop("show_restart_modal", None)
    st.session_state.pop("pending_prompt", None)
    st.session_state.pop("pending_skip_questions", None)
    st.session_state.pop("skipped_questions", None)


def _fetch_next_question(session_id: str):
    try:
        result = client.next_question(session_id)
    except Exception as exc:
        st.error(f"Failed to fetch question: {exc}")
        return

    status = result.get("status")
    st.session_state.status = status

    if status == "awaiting_answer":
        st.session_state.current_question = {
            "text": result.get("question"),
            "id": result.get("question_id"),
        }
        st.session_state.answer_submit_locked = False
    else:
        st.session_state.current_question = None
        st.session_state.answer_submit_locked = False


with st.form("initialize_session"):
    prompt_input = st.text_area("Initial Prompt", height=200)
    skip_questions = st.checkbox("Skip clarifying questions", help="When selected, the system will skip the clarification phase and proceed directly to refinement.")
    refinement_running = st.session_state.get("refinement_running", False) or st.session_state.get("pending_iteration", False)
    submitted = st.form_submit_button(
        "Start Session",
        disabled=refinement_running,
        help="Finish the current refinement before starting a new session." if refinement_running else None,
    )

if submitted:
    if not prompt_input.strip():
        st.warning("Please provide a prompt before starting.")
    else:
        if st.session_state.get("session_id"):
            st.session_state.pending_prompt = prompt_input.strip()
            st.session_state.pending_skip_questions = skip_questions
            st.session_state.show_restart_modal = True
            st.session_state.modal_active = True
            st.rerun()
        else:
            try:
                response = client.create_session(prompt_input.strip(), skip_questions=skip_questions)
                st.session_state.session_id = response["session_id"]
                st.session_state.status = response["status"]
                st.session_state.current_question = None
                st.session_state.answers = []
                st.session_state.report = None
                st.session_state.answer_submit_locked = False
                st.session_state.refinement_running = False
                st.session_state.skipped_questions = skip_questions
                if not skip_questions:
                    _fetch_next_question(st.session_state.session_id)
                st.success("Session created." + (" Skipping clarification phase." if skip_questions else ""))
            except Exception as exc:
                st.error(f"Failed to create session: {exc}")

if st.session_state.get("show_restart_modal"):

    @st.dialog("Start New Session")
    def restart_dialog():
        st.write(
            "A session is already in progress. Starting a new one will discard the current conversation."
        )
        skip_questions_modal = st.checkbox(
            "Skip clarifying questions",
            value=st.session_state.get("pending_skip_questions", False),
            key="modal_skip_questions",
            help="When selected, the system will skip the clarification phase and proceed directly to refinement."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Cancel", key="restart_cancel"):
                st.session_state.show_restart_modal = False
                st.session_state.pending_prompt = None
                st.session_state.pending_skip_questions = None
                st.session_state.modal_active = False
                st.rerun()
        with col_b:
            if st.button("Start New Session", key="restart_confirm"):
                prompt_value = st.session_state.get("pending_prompt", "")
                st.session_state.show_restart_modal = False
                st.session_state.pending_prompt = None
                st.session_state.pending_skip_questions = None
                st.session_state.modal_active = False
                if prompt_value:
                    try:
                        response = client.create_session(prompt_value, skip_questions=skip_questions_modal)
                        st.session_state.session_id = response["session_id"]
                        st.session_state.status = response["status"]
                        st.session_state.current_question = None
                        st.session_state.answers = []
                        st.session_state.report = None
                        st.session_state.answer_submit_locked = False
                        st.session_state.refinement_running = False
                        st.session_state.skipped_questions = skip_questions_modal
                        if not skip_questions_modal:
                            _fetch_next_question(st.session_state.session_id)
                        st.success("Session created." + (" Skipping clarification phase." if skip_questions_modal else ""))
                    except Exception as exc:
                        st.error(f"Failed to create session: {exc}")
                st.rerun()

    restart_dialog()

if "session_id" in st.session_state:
    st.sidebar.write(f"Session ID: {st.session_state.session_id}")
    st.sidebar.write(f"Status: {st.session_state.status}")

    with st.sidebar.expander("Session Actions"):
        if st.button("Reset Session"):
            _reset_session_state()
            st.rerun()

# Dynamically order tabs based on whether questions were skipped
if st.session_state.get("skipped_questions", False):
    iteration_tab, clarification_tab = st.tabs(["Iteration", "Clarification (Skipped)"])
else:
    clarification_tab, iteration_tab = st.tabs(["Clarification", "Iteration"])

with clarification_tab:
    if "session_id" not in st.session_state:
        st.info("Create a session to start the clarification flow.")
    elif st.session_state.get("skipped_questions", False):
        st.info("Clarification phase was skipped for this session. Proceed directly to the Iteration tab.")
    else:
        session_id = st.session_state.session_id
        st.write(f"Current status: {st.session_state.status}")

        question = st.session_state.get("current_question")
        if question:
            st.subheader("Current Question")
            st.write(question["text"])        
            form_key = f"question_form_{question['id']}"
            answer_input_key = f"answer_input_{question['id']}"
            submit_disabled = st.session_state.get("answer_submit_locked", False)

            with st.form(form_key):
                answer = st.text_area("Your answer", key=answer_input_key)
                submitted_answer = st.form_submit_button(
                    "Submit Answer",
                    disabled=submit_disabled,
                )

            if submitted_answer:
                if not answer.strip():
                    st.warning("Please enter an answer before submitting.")
                else:
                    try:
                        st.session_state.answer_submit_locked = True
                        resp = client.submit_answer(
                            session_id,
                            answer.strip(),
                            question_id=question["id"],
                        )
                        st.session_state.status = resp.get("status", st.session_state.status)
                        st.session_state.answers.append(
                            {"question": question["text"], "answer": answer.strip()}
                        )
                        st.success("Answer submitted.")
                        _fetch_next_question(session_id)
                        st.rerun()
                    except Exception as exc:
                        st.session_state.answer_submit_locked = False
                        st.error(f"Failed to submit answer: {exc}")
        else:
            st.info("No more clarification required.")

        if st.session_state.answers:
            st.subheader("Clarification History")
            for idx, item in enumerate(st.session_state.answers, 1):
                st.write(f"{idx}. Q: {item['question']}")
                st.write(f"   A: {item['answer']}")

with iteration_tab:
    if "session_id" not in st.session_state:
        st.info("Create a session to run refinements.")
    else:
        session_id = st.session_state.session_id
        st.write(f"Current status: {st.session_state.status}")

        if st.button(
            "Run Iteration",
            disabled=st.session_state.get("pending_iteration", False)
            or st.session_state.get("refinement_running", False),
        ):
            st.session_state.report = None
            st.session_state.iteration_message = None
            st.session_state.pending_iteration = True
            st.rerun()

        if st.session_state.get("iteration_message"):
            level, message = st.session_state.iteration_message
            if level == "success":
                st.success(message)
            elif level == "error":
                st.error(message)
            st.session_state.iteration_message = None

        if st.session_state.get("report"):
            report = st.session_state.report
            st.subheader("Final Prompt")
            st.code(report.get("final_prompt", ""))

            st.subheader("Change Log")
            change_log = report.get("change_log") or []
            if not change_log:
                st.write("(no change log)")
            else:
                for idx, entry in enumerate(change_log, 1):
                    st.write(f"{idx}. Changes: {entry.get('changes', '')}")
                    st.write(f"   Rationale: {entry.get('rationale', '')}")

            st.subheader("Metadata")
            st.write(f"Iterations: {report.get('iteration_count')}")
            st.json(report.get("step_timestamps", {}))
            st.write(
                f"Started: {report.get('started_at')} | Completed: {report.get('completed_at')}"
            )
