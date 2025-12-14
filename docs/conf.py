"""Sphinx documentation build configuration file."""  # noqa: INP001

import importlib.metadata


project = "allisbns"
author = "Maxim Stolyarchuk"
version = importlib.metadata.version("allisbns")
release = version

html_theme = "alabaster"
html_show_copyright = False
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}
html_theme_options = {
    "fixed_sidebar": True,
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_mdinclude",
]

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

autosectionlabel_prefix_document = True
suppress_warnings = ["autosectionlabel.*"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "autoclass_content": "both",
}

autodoc_expand_types = False
autodoc_typehints_format = "short"

autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_rtype = False
napoleon_use_ivars = False
