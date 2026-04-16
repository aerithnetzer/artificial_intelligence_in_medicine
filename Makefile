#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = artificial_intelligence_in_medicine
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

.PHONY: clean_figures
clean_figures:
	rm -rf reports/figures/GENE_EXPRESSION/*
	rm -rf reports/figures/ARTIFICIAL_INTELLIGENCE/*
	rm -rf reports/figures/NULL/*
	rm -rf reports/figures/comparative/*

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	uv run artificial_intelligence_in_medicine/dataset.py

## Generate all visualizations (per-mode + comparative)
.PHONY: visualizations
visualizations: requirements
	uv run python -m artificial_intelligence_in_medicine.generate_all all

## Generate only comparative (cross-field) visualizations
.PHONY: comparative
comparative: requirements
	uv run python -m artificial_intelligence_in_medicine.generate_all comparative

## Generate temporal visualizations only
.PHONY: temporal
temporal: requirements
	uv run python -m artificial_intelligence_in_medicine.generate_all temporal

## Generate geographic visualizations only
.PHONY: geographic
geographic: requirements
	uv run python -m artificial_intelligence_in_medicine.generate_all geographic

## Generate funding visualizations only
.PHONY: funding
funding: requirements
	uv run python -m artificial_intelligence_in_medicine.generate_all funding

## Run graph analysis pipeline
.PHONY: graphs
graphs: requirements
	uv run python -m artificial_intelligence_in_medicine.modeling.graphs


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
