# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

# GLOBAL VARIABLES
# Set Virtualenv directory name
VENV = "venv"

CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_GIT = $(shell command -v git 2> /dev/null)

CHECK_CYTHON = $(shell ${VENV}/bin/python -m pip list|grep cython)
CHECK_NUMPY = $(shell ${VENV}/bin/python -m pip list|grep numpy)
# Uncomment Rasterio lines for local GDAL installation (typically OTB install)
#CHECK_RASTERIO = $(shell ${VENV}/bin/python -m pip list|grep rasterio)

.PHONY: help venv install lint format tests docs docker clean

help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


check: ## check if cmake, git are installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_GIT}" ] || ( echo ">> git not found"; exit 1 )

venv: check ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || virtualenv -p `which python3` ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip
	@touch ${VENV}/bin/activate

install: venv  ## install environment for development target (depends venv)
	@[ "${CHECK_CYTHON}" ] ||${VENV}/bin/python -m pip install --upgrade cython
	@[ "${CHECK_NUMPY}" ] ||${VENV}/bin/python -m pip install --upgrade numpy
#	@[ "${CHECK_RASTERIO}" ] ||${VENV}/bin/python -m pip install --no-binary rasterio rasterio
	@test -f ${VENV}/bin/demcompare || ${VENV}/bin/pip install -e .[dev]
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit

lint: install  ## run lint tools (depends install)
	@${VENV}/bin/isort --check **/*.py
	@${VENV}/bin/black --check **/*.py
	@${VENV}/bin/flake8 **/*.py
	@${VENV}/bin/pylint **/*.py

format: install  ## run black and isort (depends install)
	@${VENV}/bin/isort **/*.py
	@${VENV}/bin/black **/*.py

tests: install ## run tests
	@cd tests;../${VENV}/bin/demcompare @opts.txt
	@cd tests;../${VENV}/bin/demcompare_with_baseline

docker: ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image"
	@docker build -t demcompare .

clean: ## clean: remove venv
	@rm -rf ${VENV}
	@rm -rf dist
	@rm -rf build
	@rm -rf demcompare.egg-info
	@rm -rf demcompare/__pycache__
