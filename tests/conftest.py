import os
import sys
import pytest
from omegaconf import DictConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

@pytest.fixture
def simple_config():
    # Minimal config used by Master.__init__ (it reads config.save.autosave_interval)
    return DictConfig({"save": {"autosave_interval": 1000}})
