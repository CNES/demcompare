.. _coding_guide:

Coding guide: Quality, Tests, Documentation, Pre-commit
=======================================================

Please beware that a **make** command is available to ease main maintenance workflows. 
Run **make help** to see all possibilities.

General rules
*************

Here are some rules to apply when developing a new functionality:

* **Comments:** Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* **Test**: Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* **Documentation**: All functions shall be documented (object, parameters, return values).
* **Use type hints**: Use the type hints provided by the `typing` python module.
* **Use doctype**: Follow sphinx default doctype for automatic API
* **Quality code**: Correct project quality code errors with pre-commit automatic workflow (see below)
* **Factorization**: Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the **demcompare** python modules.
* **Be careful with user interface upgrade:** If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* **Logging and no print**: The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* **Limit classes**: If possible, limit the use of classes as much as possible and opt for a functional approach. The classes are reserved for data modelling if it is impossible to do so using `xarray` and for the good level of modularity.
* **Limit new dependencies**: Do not add new dependencies unless it is absolutely necessary, and only if it has a **permissive license**.

Code quality
************

**Demcompare** uses `Isort`_, `Black`_, `Flake8`_, `Pylint`_ and `Mypy`_ quality code checking.

Use the following command in **demcompare** `virtualenv`_ to check the code with these tools:

.. code-block:: console

    $ make lint

Use the following command to automatically format the code with isort and black:

.. code-block:: console

    $ make format

.. warning::
  Use the auto formatting with caution and check before committing.

Isort
-----
`Isort`_ is a Python utility / library to sort imports alphabetically, and automatically separates into sections and by type.

**Demcompare** ``isort`` configuration is done in `.pyproject.toml`_.
`Isort`_ manual usage examples:

.. code-block:: console

    $ cd DEMCOMPARE_HOME
    $ isort --check demcompare tests  # Check code with isort, does nothing
    $ isort --diff demcompare tests   # Show isort diff modifications
    $ isort demcompare tests          # Apply modifications

`Isort`_ messages can be avoided when really needed with *"# isort:skip"* on the incriminated line.

Black
-----
`Black`_ is a quick and deterministic code formatter to help focus on the content.

**Demcompare**'s ``black`` configuration is done in `.pyproject.toml`_.

If necessary, Black doesn’t reformat blocks that start with "# fmt: off" and end with # fmt: on, or lines that ends with "# fmt: skip". "# fmt: on/off" have to be on the same level of indentation.

`Black`_ manual usage examples:

.. code-block:: console

    $ cd DEMCOMPARE_HOME
    $ black --check demcompare tests  # Check code with black with no modifications
    $ black --diff demcompare tests   # Show black diff modifications
    $ black demcompare tests          # Apply modifications

Flake8
------
`Flake8`_ is a command-line utility for enforcing style consistency across Python projects. By default it includes lint checks provided by the PyFlakes project, PEP-0008 inspired style checks provided by the PyCodeStyle project, and McCabe complexity checking provided by the McCabe project. It will also run third-party extensions if they are found and installed.

**Demcompare**'s ``flake8`` configuration is done in `setup.cfg <https://raw.githubusercontent.com/CNES/Demcompare/master/setup.cfg>`_

`Flake8`_ messages can be avoided (in particular cases !) adding "# noqa" in the file or line for all messages.
It is better to choose filter message with "# noqa: E731" (with E371 example being the error number).
Look at examples in source code.

Flake8 manual usage examples:

.. code-block:: console

  $ cd DEMCOMPARE_HOME
  $ flake8 demcompare tests           # Run all flake8 tests


Pylint
------
`Pylint`_ is a global linting tool which helps to have many information on source code.

**Demcompare**'s ``pylint`` configuration is done in dedicated `.pylintrc <https://raw.githubusercontent.com/CNES/demcompare/master/.pylintrc>`_ file.

`Pylint`_ messages can be avoided (in particular cases !) adding "# pylint: disable=error-message-name" in the file or line.
Look at examples in source code. For instance, member attributes are ignored for the different factory classes on the .pylintrc file since the no-member pylint error raises due to the factory dynamics.

Pylint manual usage examples:

.. code-block:: console

  $ cd DEMCOMPARE_HOME
  $ pylint tests demcompare       # Run all pylint tests
  $ pylint --list-msgs          # Get pylint detailed errors informations


Mypy
----
`Mypy`_ is a static type checker for Python.

**Demcompare**'s ``Mypy`` configuration is done in `.pyproject.toml`_ file.

`Mypy`_ messages can be avoided (in particular cases !) adding "# type: ignore" in the file or line.

Mypy manual usage examples:

.. code-block:: console

  $ cd DEMCOMPARE_HOME
  $ mypy demcompare       # Run mypy tests


Tests
*****

Demcompare includes a set of tests executed with `pytest <https://docs.pytest.org/>`_ tool.

To run tests, use:

.. code-block:: bash

    make test

It runs the unit tests present in `demcompare/tests` displaying the traces generated by the tests and the tests code coverage level.

During some tests execution, demcompare will write the output data in a */tmp* directory.

It is possible also to run the test through tox tool on several python versions (see tox.ini configuration file):

.. code-block:: bash

    make test-all

Documentation
*************

Demcompare documentation can be generated with the following command:

.. code-block:: bash

    make docs

It cleans documentation from *docs/build/* directory and builds the sphinx documentation from *docs/source/* into *docs/build/*:

.. code-block:: bash

    sphinx-build -M clean docs/source/ docs/build
    sphinx-build -M html docs/source/ docs/build

Demcompare :doc:`/api_reference/index` is generated through the autoAPI tool.

Pre-commit validation
*********************

A `Pre-commit`_ validation is installed with code quality tools (see below).
It is installed automatically by `make install` command.

Here is the way to install it manually:

.. code-block:: console

  $ pre-commit install -t pre-commit
  $ pre-commit install -t pre-push

This installs the pre-commit hook in `.git/hooks/pre-commit` and `.git/hooks/pre-push` from `.pre-commit-config.yaml <https://raw.githubusercontent.com/CNES/demcompare/master/.pre-commit-config.yaml>`_ file configuration.

It is possible to test pre-commit before committing:

.. code-block:: console

  $ pre-commit run --all-files                # Run all hooks on all files
  $ pre-commit run --files demcompare/__init__.py   # Run all hooks on one file
  $ pre-commit run pylint                     # Run only pylint hook
  $ pre-commit run --hook-stage push --all-files # Run with push hook




.. _`virtualenv`: https://docs.python.org/fr/3/library/venv.html
.. _`Isort`: https://pycqa.github.io/isort/
.. _`Black`: https://black.readthedocs.io/
.. _`Flake8`: https://flake8.pycqa.org/
.. _`Pylint`: http://pylint.pycqa.org/
.. _`Mypy`: https://mypy-lang.org/
.. _`Pre-commit`: https://pre-commit.com/
.. _`.pyproject.toml`: https://raw.githubusercontent.com/CNES/demcompare/master/pyproject.toml