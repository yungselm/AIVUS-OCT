import importlib
import pytest
from PyQt5.QtWidgets import QWidget

def test_master_widget_constructs_and_shows(qtbot, monkeypatch, simple_config):
    """
    Construct gui.Master while stubbing heavy subcomponents (LeftHalf, RightHalf,
    ContourBasedGating, Predict, init_menu, init_shortcuts, write_contours).
    Then show the window and assert basic state.
    """
    gui_mod = importlib.import_module("gui.gui")

    # LeftHalf and RightHalf in your code are used as LeftHalf(self)() -> widget
    class StubContainer:
        def __init__(self, parent):
            self.parent = parent
        def __call__(self):
            return QWidget()

    # Replace heavy components with light stubs
    monkeypatch.setattr(gui_mod, "LeftHalf", StubContainer)
    monkeypatch.setattr(gui_mod, "RightHalf", StubContainer)
    monkeypatch.setattr(gui_mod, "ContourBasedGating", lambda parent: None)
    monkeypatch.setattr(gui_mod, "Predict", lambda parent: None)
    monkeypatch.setattr(gui_mod, "init_menu", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_mod, "init_shortcuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_mod, "write_contours", lambda parent: None)

    # Now construct the Master window
    Master = gui_mod.Master
    w = Master(simple_config)   # uses the patched stubs
    qtbot.addWidget(w)

    w.show()
    qtbot.waitExposed(w)

    # Basic sanity asserts
    assert w.isVisible()
    # default file_name is set in init_gui()
    assert getattr(w, "file_name", None) == "default_file_name"
    # the autosave timer should be present (we can't easily inspect Qt timers here,
    # but the attribute autosave_interval comes from config)
    assert hasattr(w, "autosave_interval")
