# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

# GLOBAL VARIABLES
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
VENV = "venv"

DEMCOMPARE_VERSION = $(shell python3 setup.py --version)
DEMCOMPARE_VERSION_MIN = $(shell echo ${DEMCOMPARE_VERSION} | cut -d . -f 1,2,3)

.PHONY: help venv install lint format test-ci test doc docker clean

help: ## this help
	@echo "      DEMCOMPARE MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv: ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || python3 -m venv ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${VENV}/bin/activate

install: venv  ## install environment for development target (depends venv)
	@test -f ${VENV}/bin/demcompare || ${VENV}/bin/pip install -e .[dev,doc]
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${VENV}/bin/pre-commit install -t pre-push
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "Demcompare ${DEMCOMPARE_VERSION} installed in dev mode in virtualenv ${VENV} with Sphinx docs"
	@echo "Demcompare venv usage : source ${VENV}/bin/activate; demcompare -h"

lint: install  ## run lint tools (depends install)
	@echo "Demcompare linting isort check"
	@${VENV}/bin/isort --check demcompare tests
	@echo "Demcompare linting black check"
	@${VENV}/bin/black --check demcompare tests
	@echo "Demcompare linting flake8 check"
	@${VENV}/bin/flake8 demcompare tests
	@echo "Demcompare linting pylint check"
	@set -o pipefail; ${VENV}/bin/pylint demcompare tests --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

format: install  ## run black and isort formatting (depends install)
	@${VENV}/bin/isort demcompare tests
	@${VENV}/bin/black demcompare tests

test-ci: install ## tox run all tests with python3.7 and python3.8 + coverage
	# Run tox (recreate venv (-r) and parallel mode (-p auto)) for CI
	@${VENV}/bin/tox -r -p auto

test: install ## run all tests + coverage with one python version
	# Run pytest directly
	@${VENV}/bin/pytest -o log_cli=true --cov-config=.coveragerc --cov --cov-report=term-missing

doc: install ## build sphinx documentation
	@${VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${VENV}/bin/sphinx-build -M html docs/source/ docs/build

docker: ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image Demcompare ${DEMCOMPARE_VERSION_MIN}"
	@docker build -t cnes/demcompare:${DEMCOMPARE_VERSION_MIN} -t cnes/demcompare:latest .

clean: ## clean: remove venv and all generated files
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@rm -rf .eggs/
	@rm -rf demcompare.egg-info
	@rm -rf dist/
	@rm -rf build/
	@rm -rf ${VENV}
	@rm -rf .tox/
	@rm -rf .pytest_cache/
	@rm -f pytest-report.xml
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -f coverage.xml
	@rm -rf htmlcov/
	@rm -f pylint-report.txt
	@rm -f pylint-report.xml
	@rm -f debug.log
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/
	@rm -rf docs/source/apidoc/
