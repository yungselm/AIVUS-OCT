# docs/conf.py

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'aivus-caa'
author = 'AI-in-Cardiovascular-Medicine'
copyright = '2025, AI-in-Cardiovascular-Medicine'
ns = {}
exec(Path("../src/version.py").read_text(), ns)
release = ns.get('__version__', '0.0.0')

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',            # support Markdown (for CONTRIBUTING.md etc)
    'sphinx.ext.autodoc',     # automatic doc generation from docstrings
    'sphinx.ext.napoleon',    # support for NumPy/Google style docstrings
    'sphinx.ext.viewcode',    # add links to source code
    'sphinx.ext.todo'         # support todo directives
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']

# (Optional: add any theme options or HTML title here)
html_title = 'AIVUS-CAA Documentation'
html_logo = ''
