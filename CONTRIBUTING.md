# **DEMcompare** **Contributing guide**.

1. [Bug report](#bug-report)
2. [Contributing workflow](#contributing-workflow)
3. [Coding guide](#coding-guide)
4. [Pre-commit validation](#pre-commit-validation)
5. [Pylint usage](#pylint-usage)
6. [Merge request acceptation process](#merge-request-acceptation-process)

# Bug report

Any proven or suspected malfunction should be traced in a bug report, the latter being an issue in the DEMcompare repository.

**Don't hesitate to do so: It is best to open a bug report and quickly resolve it than to let a problem remains in the project.**
**Notifying the potential bugs is the first way for contributing to a software.**

In the problem description, be as accurate as possible. Include:
* The procedure used to initialize the environment
* The incriminated command line or python function
* The content of the input and output configuration files

# Contributing workflow

Any code modification requires a Merge Request. It is forbidden to push patches directly into master (this branch is protected).

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add `WIP:` before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding `Closes xx` where xx is the reference number of the issue.

Likewise, if you work on a branch (which is recommended), prefix the branch's name by `xx-` in order to link it to the xx issue.

DEMcompare Classical workflow is :
* Create an issue (or begin from an existing one)
* Create a Merge Request from the issue: a MR is created accordingly with "WIP:", "Closes xx" and associated "xx-name-issue" branch
* Hack code from a local working directory or from the forge (less possibilities)
* Git add, commit and push from local working clone directory or from the forge directly
* Install pre-commit validation process (using black, isort, flake8 and pylint) and check errors
* Follow [Conventional commits](https://www.conventionalcommits.org/) specifications for commit messages
* Launch the [test](./README.md) on your modifications.
* When finished, change your Merge Request name (erase "WIP:" in title ) and ask to review the code (see below Merge request acceptation process)

# Coding guide

Here are some rules to apply when developing a new functionality:
* Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* If possible, limit the use of classes as much as possible and opt for a functional approach. The classes are reserved for data modelling if it is impossible to do so using `xarray`.
* Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* All functions shall be documented (object, parameters, return values).
* Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the python modules.
* If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* Do not add new dependencies unless it is absolutely necessary, and only if it has a permissive license.
* Use the type hints provided by the `typing` python module.
* Correct project quality code errors (see below) : isort, black, flake8, pylint
* The line length is 80

# Pre-commit validation

A Pylint pre-commit validation is installed in Continuous Integration.
Here is the way to install it:

```
pre-commit install
```
This installs the pre-commit hook in `.git/hooks/pre-commit` and `.git/hooks/pre-push`  from `.pre-commit-config.yaml` file configuration.
The pre-commit checks different validation process: isort
```
It is possible to test pre-commit before commiting:
* pre-commit run --all-files                      # Run all hooks on all files
* pre-commit run --files demcompare/__init__.py   # Run all hooks on one file
* pre-commit run pylint                           # Run only pylint hook
```

# Isort
[Isort](https://pypi.org/project/isort/) checks python imports validity in source code.
The configuration is in isort section [pyproject.toml](./pyproject.toml) file.
It is configured with black profile.

# Black
[Black](https://pypi.org/project/black/) is the uncompromising Python code formatter.
The configuration is in a black section in [pyproject.toml](./pyproject.toml) file.
The default configuration is used.

# Flake8
[Flake8](https://pypi.org/project/flake8/) is a wrapper around PyFlakes, pycodestyle,  Ned Batchelder’s McCabe script.
The configuration is put in a flake8 section in [setup.cfg](./setup.cfg)

[flake8-copyright](https://pypi.org/project/flake8-copyright/) is installed to check copyright in added file.
Flake8 messages can be avoided (if necessary !) adding "# noqa error-number"

[flake8-bugbear](https://pypi.org/project/flake8-bugbear/) adds several rules to flake8

[flake8-comprehensions](https://pypi.org/project/flake8-comprehensions/) is set for checking dict, set, list structures usage.


# Pylint
[pylint](https://pypi.org/project/pylint/) is well known global rules checker, complementary with flake8.
The configuration is set in [.pylintrc](./.pylintrc) file.
It is possible to run only pylint tool to check code modifications:
```
* cd DEMCOMPARE_HOME
* pylint *.py demcompare/*.py             # Run all pylint tests
* pylint --list-msgs                      # Get pylint detailed errors informations
```
Pylint messages can be avoided (in particular cases !) adding "#pylint: disable=error-message-name" in the file or line.
Look at examples in code.


# Merge request acceptation process

Two Merge Requests types to help the workflow :
- Simple merge requests (bugs, documentation) can be merged directly by the author with rights on master.
- Advanced merge requests (typically a big change in code) are flagged with "To be Reviewed" by the author

This mechanism is to help quick modifications and avoid long reviews on unneeded simple merge requests.
The author has to be responsible in the need or  not to be reviewed.

## Advanced Merge Request
The Advanced Merge Request will be merged into master after being reviewed by a DEMcompare steering committee (core committers) composed of:
* Emmanuelle Sarrazin (CNES)
* Emmanuel Dubois (CNES)

Only the members of this committee can merge into master.

The checklist of a Advanced Merge Request acceptance is the following:
* [ ] At least one code review have been done by members of the group above (who check among others the correct application of the rules listed in the [Coding guide](# Coding guide)).
* [ ] All comments of the reviewers has been dealt with and are closed
* [ ] The reviewers have signaled their approbation (thumb up)
* [ ] No developer is against the Merge Request (thumb down)
