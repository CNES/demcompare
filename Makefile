# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

.PHONY: help venv install lint format tests docs clean

help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv: ## create virtualenv in "venv" dir if not exists
	@test -d venv || virtualenv -p `which python3` venv
	@venv/bin/python -m pip install --upgrade pip
	@touch venv/bin/activate

install: venv  ## install environment for development target (depends venv)
	@test -f venv/bin/demcompare || venv/bin/pip install -e .[dev]
	@test -f .git/hooks/pre-commit || venv/bin/pre-commit install -t pre-commit

lint: install  ## run lint tools (depends install)
	@venv/bin/isort --check **/*.py
	@venv/bin/black --check **/*.py
	@venv/bin/flake8 **/*.py
	@venv/bin/pylint **/*.py

format: install  ## run black and isort (depends install)
	@venv/bin/isort **/*.py
	@venv/bin/black **/*.py

clean: ## clean: remove venv
	@rm -rf venv
	@rm -rf dist
	@rm -rf build
	@rm -rf demcompare.egg-info
	@rm -rf demcompare/__pycache__
