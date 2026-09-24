from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_interactive_plan_in_both_tabs():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert len(app.get("plotly_chart")) >= 2
    reset_buttons = [
        button
        for button in app.button
        if button.label == "Vider le plan et réinitialiser les observations"
    ]
    assert len(reset_buttons) == 1
    assert reset_buttons[0].disabled
    assert any(toggle.label == "Analyser la couverture entre Avant et Après" for toggle in app.toggle)
