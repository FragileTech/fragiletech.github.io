.PHONY: build docs serve clean install venv

# Use local venv if it exists, otherwise use system python
VENV_DIR := .venv
PYTHON := $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python)
JUPYTER_BOOK := $(if $(wildcard $(VENV_DIR)/bin/jupyter-book),$(VENV_DIR)/bin/jupyter-book,jupyter-book)

# Build the Jupyter Book
build: venv
	$(JUPYTER_BOOK) build .

# Alias for build
docs: build

# Serve the built site
serve: build
	$(PYTHON) -m http.server -d _build/html 8000

# Build PDF output
pdf: venv
	$(JUPYTER_BOOK) build . --builder pdflatex

# Clean build artifacts
clean:
	rm -rf _build .jupyter_cache

# Create virtual environment with dependencies
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv $(VENV_DIR); \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
		$(VENV_DIR)/bin/pip install "jupyter-book>=1.0.0,<2.0.0" sphinx-tippy sphinx-proof sphinxcontrib-bibtex sphinxcontrib-mermaid; \
	fi

# Force reinstall dependencies
install:
	rm -rf $(VENV_DIR)
	$(MAKE) venv
