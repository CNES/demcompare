# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv
# Some Makefile global variables can be set in make command line
# Recall: .PHONY  defines special targets not associated with files

############### GLOBAL VARIABLES ######################
.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Exemple: VENV="other-venv/" make install
ifndef VENV
	VENV = "venv"
endif

# Software version from setup.py and setuptools_scm
VERSION = $(shell python3 setup.py --version)
VERSION_MIN = $(shell echo ${VERSION} | cut -d . -f 1,2,3)

# Browser definition
define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

## Install section

.PHONY: git
git: ## init local git repository if not present
	@test -d .git/ || git init .

.PHONY: venv
venv: ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || python3 -m venv ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${VENV}/bin/activate

.PHONY: install
install: venv git  ## install the package in dev mode in virtualenv
	@test -f ${VENV}/bin/mesh_3d || echo "Install mesh_3d package from local directory"
	@test -f ${VENV}/bin/mesh_3d || ${VENV}/bin/python -m pip install -e .[dev,docs]
	@test -f .git/hooks/pre-commit || echo "Install pre-commit"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "mesh_3d ${VERSION} installed in dev mode in virtualenv ${VENV} with documentation"
	@echo " mesh_3d venv usage : source ${VENV}/bin/activate; mesh_3d -h"

## Test section
	
.PHONY: test
test: install ## run tests and coverage quickly with the default Python
	@${VENV}/bin/pytest -o log_cli=true --cov-config=.coveragerc --cov --cov-report=term-missing

.PHONY: test-all
test-all: install ## run tests on every Python version with tox
	@${VENV}/bin/tox -r -p auto  ## recreate venv (-r) and parallel mode (-p auto)
	
.PHONY: coverage
coverage: install ## check code coverage quickly with the default Python
	@${VENV}/bin/coverage run --source mesh_3d -m pytest
	@${VENV}/bin/coverage report -m
	@${VENV}/bin/coverage html
	$(BROWSER) htmlcov/index.html

## Code quality, linting section

### Format with isort and black

.PHONY: format
format: install format/isort format/black  ## run black and isort formatting (depends install)

.PHONY: format/isort
format/isort: install  ## run isort formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/isort mesh_3d tests

.PHONY: format/black
format/black: install  ## run black formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/black mesh_3d tests

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint: install lint/isort lint/black lint/flake8 lint/pylint ## check code quality and linting

.PHONY: lint/isort
lint/isort: ## check imports style with isort
	@echo "+ $@"
	@${VENV}/bin/isort --check mesh_3d tests
	
.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${VENV}/bin/black --check mesh_3d tests

.PHONY: lint/flake8
lint/flake8: ## check linting with flake8
	@echo "+ $@"
	@${VENV}/bin/flake8 mesh_3d tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${VENV}/bin/pylint mesh_3d tests --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash
	
## Documentation section

.PHONY: docs
docs: install ## generate Sphinx HTML documentation, including API docs
	@${VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${VENV}/bin/sphinx-apidoc -o docs/source/apidoc/ mesh_3d
	@${VENV}/bin/sphinx-build -M html docs/source/ docs/build
	$(BROWSER) docs/build/html/index.html

## Notebook section

notebook: install ## Install Jupyter notebook kernel with venv
	@echo "\nInstall Jupyter Kernel and launch Jupyter notebooks environment"
	@${VENV}/bin/python -m ipykernel install --sys-prefix --name=mesh_3d$(VENV) --display-name=mesh_3d$(VERSION)
	@echo "\n --> After virtualenv activation, please use following command to launch local jupyter notebook to open Notebooks:"
	@echo "jupyter notebook"

## Docker section

docker: git ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image mesh_3d ${VERSION_MIN}"
	@docker build -t chloe thenoz (magellium), lisa vo thanh (magellium)/mesh_3d:${VERSION_MIN} -t chloe thenoz (magellium), lisa vo thanh (magellium)/mesh_3d:latest .

## Release section
	
.PHONY: dist
dist: clean install ## clean, install, builds source and wheel package
	@${VENV}/bin/python -m pip install --upgrade build
	@${VENV}/bin/python -m build
	ls -l dist

.PHONY: release
release: dist ## package and upload a release
	@twine check dist/*
	@twine upload dist/* --verbose ##  update your .pypirc accordingly

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${VENV}

.PHONY: clean-build
clean-build: ## remove build artifacts
	@echo "+ $@"
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -name '*~' -delete

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f pylint-report.txt
	@rm -f debug.log

.PHONY: clean-docs
clean-docs:
	@echo "+ $@"
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/
	@rm -rf docs/source/apidoc/

.PHONY: clean-docker
clean-docker:
		@echo "+ $@"
		@echo "Clean Docker image mesh_3d ${VERSION_MIN}"
		@docker image rm chloe thenoz (magellium), lisa vo thanh (magellium)/mesh_3d:${VERSION_MIN}
		@docker image rm chloe thenoz (magellium), lisa vo thanh (magellium)/mesh_3d:latest
