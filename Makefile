#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cifar10_classification
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements: create_environment
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt
	venv/bin/python scripts/downloads.py

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	venv/bin/flake8 cifar10_classification
	venv/bin/isort --check --diff --profile black cifar10_classification
	venv/bin/black --check --config pyproject.toml cifar10_classification

## Format source code with black
.PHONY: format
format:
	venv/bin/isort --profile black cifar10_classification
	venv/bin/black --config pyproject.toml cifar10_classification

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> New virtualenv created. Activate with:\nsource venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Prepare Data
.PHONY: prepare_data
prepare_data: requirements
	venv/bin/python scripts/prepare_data.py

## Extract Features
.PHONY: extract_features
extract_features: prepare_data
	venv/bin/python scripts/extract_features.py

## Visualize Features
.PHONY: visualize_features
visualize_features: extract_features
	venv/bin/python scripts/visualize_features.py

## Train Models
.PHONY: train_models
train_models: extract_features
	venv/bin/python scripts/train_models.py

## Evaluate Metrics
.PHONY: evaluate_metrics
evaluate_metrics: train_models
	venv/bin/python scripts/evaluate_metrics.py

## Plot Results
.PHONY: plot_results
plot_results: evaluate_metrics
	venv/bin/python scripts/plot_results.py

## Run All Steps
.PHONY: all
all: prepare_data extract_features visualize_features train_models evaluate_metrics plot_results

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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
