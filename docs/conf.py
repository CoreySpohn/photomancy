"""Sphinx configuration file."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

project = "photomancy"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"

try:
    release = _get_version("photomancy")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/_version.py"]
autodoc_typehints = "description"

myst_enable_extensions = ["amsmath", "dollarmath"]
myst_fence_as_directive = ["mermaid"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "photomancy"
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/photomancy",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {"default_mode": "dark"}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "off"
nb_execution_timeout = 300
