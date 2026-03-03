# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RLDocs'
copyright = '2026, Zahiruddin Mahammad'
author = 'Zahiruddin Mahammad'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# --- Zahir ------
extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = [
    "amsmath",      # supports \begin{align} ... \end{align}
    "dollarmath",   # supports $...$ for inline and $$...$$ for block
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# -----------------

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# --- Zahir ------
html_theme_options = {
    "collapse_navigation": False,  # keep True by default; set to False to disable dropdown
    "sticky_navigation": True,     # optional: keep sidebar visible on scroll
    "navigation_depth": 1,         # show only top-level pages in sidebar
}
# -----------------