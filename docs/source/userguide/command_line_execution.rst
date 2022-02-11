.. _command_line_execution:

.. role:: bash(code)
   :language: bash

Command line execution
**********************

Execution from the command line
===============================

**Demcompare** is executed from the command line with an input configuration file:

.. code-block:: bash

    demcompare config_file.json #run demcompare

All the parameters on *config_file.json* are described on :ref:`inputs`.

Step by step execution from the command line
============================================

**Demcompare** allows one to execute only a subset of the whole process. As such, a :bash:`--step` command line argument is
provided. It accepts the values `coregistration`, `stats` and `report` :

.. code-block:: bash

    [user@machine] $ demcompare
    usage: demcompare [-h]
      [--step step_name [step_name ...]]
      [--display] [--version]
      config.json

- All the steps but **stats** are optional

- **Demcompare** can start at any step as long as the previously required steps have been launched.

  - This means that one can launch the report step only as long as the stats step has already been performed from a previous **demcompare** launch and the *config.json* remains the same.
  - The steps are space-separated (no comma).

.. note::  Coregistration step is **not** mandatory as one may consider that the DEMs are already coregistered.


Optional parameters file
========================

**Demcompare** can be launched with a file containing its parameters (one per line) with "@" character:

.. code-block:: bash

    demcompare @opts.txt

Where a possible *opts.txt* file would contain:

.. code-block:: bash

    test_config.json
    --display

