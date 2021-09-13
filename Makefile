# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

# GLOBAL VARIABLES
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
VENV = "venv"

CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_GIT = $(shell command -v git 2> /dev/null)

CHECK_CYTHON = $(shell ${VENV}/bin/python -m pip list|grep cython)
CHECK_NUMPY = $(shell ${VENV}/bin/python -m pip list|grep numpy)
# Uncomment Rasterio lines for local GDAL installation (typically OTB install)
#CHECK_RASTERIO = $(shell ${VENV}/bin/python -m pip list|grep rasterio)

DEMCOMPARE_VERSION = $(shell python3 setup.py --version)
DEMCOMPARE_VERSION_MIN = $(shell echo ${DEMCOMPARE_VERSION} | cut -d . -f 1,2,3)

.PHONY: help check venv install install-deps lint format tests docs docker clean

help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


check: ## check if cmake, git are installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_GIT}" ] || ( echo ">> git not found"; exit 1 )

venv: check ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || virtualenv -p `which python3` ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${VENV}/bin/activate

install-deps: venv ## install demcompare dependencies
	@[ "${CHECK_CYTHON}" ] ||${VENV}/bin/python -m pip install --upgrade cython
	@[ "${CHECK_NUMPY}" ] ||${VENV}/bin/python -m pip install --upgrade numpy
#	@[ "${CHECK_RASTERIO}" ] ||${VENV}/bin/python -m pip install --no-binary rasterio rasterio

install: install-deps  ## install environment for development target (depends venv)
	@test -f ${VENV}/bin/demcompare || ${VENV}/bin/pip install -e .[dev]
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "Demcompare ${DEMCOMPARE_VERSION} installed in dev mode in virtualenv ${VENV}"
	@echo "Demcompare venv usage : source ${VENV}/bin/activate; demcompare -h"

lint: install  ## run lint tools (depends install)
	@echo "Demcompare linting isort check"
	@${VENV}/bin/isort --check demcompare
	@echo "Demcompare linting black check"
	@${VENV}/bin/black --check demcompare
	@echo "Demcompare linting flake8 check"
	@${VENV}/bin/flake8 demcompare
	@echo "Demcompare linting pylint check"
	@set -o pipefail; ${VENV}/bin/pylint demcompare --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

format: install  ## run black and isort formatting (depends install)
	@${VENV}/bin/isort demcompare
	@${VENV}/bin/black demcompare

tests: install ## run tests
	@cd tests;../${VENV}/bin/demcompare @opts.txt
	@cd tests;../${VENV}/bin/demcompare_with_baseline

docker: ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image Demcompare ${DEMCOMPARE_VERSION_MIN}"
	@docker build -t cnes/demcompare:${DEMCOMPARE_VERSION_MIN} -t cnes/demcompare:latest .

clean: ## clean: remove venv
	@rm -rf ${VENV}
	@rm -rf dist
	@rm -rf build
	@rm -rf demcompare.egg-info
	@rm -rf .eggs/
	@rm -rf demcompare/__pycache__
	@rm -rf pylint-report.txt
	@rm -rf tests/test_output/
