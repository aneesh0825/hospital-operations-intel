from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    return app


def test_top_navigation_switches_active_page():
    app = _app()
    assert app.session_state["active_page"] == "Operations Console"

    app.radio[0].set_value("Patient Workup").run(timeout=30)
    assert app.session_state["active_page"] == "Patient Workup"
    assert not app.exception

    app.radio[0].set_value("Risk Assessment").run(timeout=30)
    assert app.session_state["active_page"] == "Risk Assessment"
    assert not app.exception


def test_queue_selection_opens_selected_patient_workup():
    app = _app()
    app.button[0].click().run(timeout=30)

    assert app.session_state["active_page"] == "Patient Workup"
    assert app.session_state["selected_patient_id"]
    assert app.session_state["workup_patient"]
    assert not app.exception
