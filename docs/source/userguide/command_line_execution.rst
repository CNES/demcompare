.. _command_line_execution:

.. role:: bash(code)
   :language: bash

Step by step execution from the command line
********************************************

**Demcompare** allows one to execute only a subset of the whole process. As such, a :bash:`--step` command line argument is
provided. It accepts values in `coregistration` `stats` `report` :

.. code-block:: bash

    [user@machine] $ demcompare
    usage: demcompare [-h]
      [--step step_name [step_name ...]]
      [--display] [--version]
      config.json

- All the steps but **stats** are optional

- **Demcompare** can start at any step as long as previously required steps have been launched.

  - This means that one can launch the report step only as long as the stats step has already been performed from a previous **demcompare** launch and the *config.json* remains the same. The steps are space-separated (no comma).

.. note::  Coregistration step is not mandatory for stats and following steps as one can decide its DEMs are already coregistered.

.. note::  Be careful: the positional argument for the configuration file can be wrongly considered as a step if used after the :bash:`--step` option.

    .. code-block:: bash

        demcompare --step stats config.json : KO
        demcompare config.json --step stats : OK

For the Nuth & Kääb universal coregistration algorithm :

.. [NuthKaab] Nuth, C. Kääb, 2011. A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. Cryosphere 5, 271290.
