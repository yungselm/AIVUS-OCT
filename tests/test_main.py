import sys
import pytest
import importlib
from types import SimpleNamespace

def test_main_initializes_master(monkeypatch, simple_config):
    mod = importlib.import_module("main")

    constructed = {}
    def fake_master(cfg):
        constructed["cfg"] = cfg

    class DummyApp:
        def __init__(self, argv):
            self.argv = argv
        def setApplicationVersion(self, v):
            pass
        def exec_(self):
            return 0

    # Patch heavy/Qt-y things so the entrypoint doesn't try to access the real GUI
    monkeypatch.setattr(mod, "Master", fake_master)
    monkeypatch.setattr(mod, "QApplication", DummyApp)

    # Stub qdarktheme functions used in main() so they don't require a real QApplication
    # mod.qdarktheme is the module object imported in src/main.py, so patch its members
    if hasattr(mod, "qdarktheme"):
        monkeypatch.setattr(mod.qdarktheme, "enable_hi_dpi", lambda *a, **k: None)
        monkeypatch.setattr(mod.qdarktheme, "setup_theme", lambda *a, **k: None)
    else:
        # fallback: replace with a dummy namespace (rare)
        monkeypatch.setattr(mod, "qdarktheme", SimpleNamespace(enable_hi_dpi=lambda *a, **k: None,
                                                              setup_theme=lambda *a, **k: None))

    # Prevent real sys.exit from killing pytest: make it raise SystemExit so we can assert it.
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc:
        mod.main.__wrapped__(simple_config)

    assert exc.value.code == 0
    assert "cfg" in constructed
