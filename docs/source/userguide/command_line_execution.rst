.. _command_line_execution:

.. role:: bash(code)
   :language: bash

Step by step execution from the command line
********************************************

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



