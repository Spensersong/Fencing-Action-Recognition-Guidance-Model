from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_real_streamlit_demo_produces_pdf_and_result_table():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    assert not app.exception
    app.radio[0].set_value("Sample drill").run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 1
    assert app.session_state["report"]["files"]["pdf"].startswith(b"%PDF-")
    assert app.session_state["report"]["result"]["simulated"]
