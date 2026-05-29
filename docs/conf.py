project = "ZShooter ScopeSim"
copyright = "2026, Caltech Optical Observatories"
author = "ZShooter Team"

extensions = [
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

try:
    import sphinx_book_theme  # noqa: F401
except ModuleNotFoundError:
    html_theme = "alabaster"
else:
    html_theme = "sphinx_book_theme"
html_title = "ZShooter ScopeSim"
html_static_path = ["_static"]

source_suffix = {
    ".md": "markdown",
}
